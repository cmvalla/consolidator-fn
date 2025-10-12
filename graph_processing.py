# Graph processing for the consolidator function
import logging
import time
import igraph as ig
import numpy as np
import uuid
import json
import concurrent.futures
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chains import LLMChain
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from cdlib import algorithms

from config import Config
from llm_operations import SUMMARY_PROMPT, CLASS_SCHEMA
import hashlib

# Define the state for the LangGraph workflow
class ClusteringState(TypedDict):
    input_graph: ig.Graph
    similarity_threshold: float
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    clusters: List[List[str]]
    id_to_entity: Dict[str, Any]
    entity_id_to_source_text: Dict[str, str]
    cluster_info_list: List[Dict[str, Any]]
    all_generated_properties: List[Dict[str, Any]]
    name_to_class_entity: Dict[str, Any]
    class_id_map: Dict[str, str]
    summaries_to_embed: List[str]
    class_eids_to_embed: List[str]
    final_graph: ig.Graph
    error: str | None

def _invoke_llm_for_cluster(llm_ops, cluster_info):
    try:
        instances_text_parts = []
        source_text_parts = set()
        
        id_to_entity = cluster_info['id_to_entity']
        entity_id_to_source_text = cluster_info['entity_id_to_source_text']

        for member_id in cluster_info["member_ids"]:
            member_entity = id_to_entity.get(member_id)
            if member_entity:
                properties = member_entity.get('properties', {})
                instances_text_parts.append(f"- {json.dumps(properties)}")
                source_text = entity_id_to_source_text.get(member_id)
                if source_text:
                    source_text_parts.add(source_text)

        instances_text = "\n".join(instances_text_parts)
        source_text_context = "\n\n---\n\n".join(source_text_parts)
        schema_str = json.dumps(CLASS_SCHEMA, indent=2)

        llm_input = {
            "instances_text": instances_text,
            "schema_str": schema_str,
            "source_text_context": source_text_context
        }
        
        generated_properties_str = llm_ops.generate_class_properties([llm_input], CLASS_SCHEMA)
        extracted_json_str = llm_ops.extract_json_from_llm_response(generated_properties_str)
        
        generated_properties = json.loads(extracted_json_str)
        if isinstance(generated_properties, list) and generated_properties:
            return generated_properties[0]
        elif isinstance(generated_properties, dict):
             return generated_properties
        else:
            logging.warning(f"Could not parse properties for a cluster. Raw response: {generated_properties_str}")
            return {}
    except Exception as e:
        logging.error(f"Failed to process a cluster in parallel: {e}", exc_info=True)
        return {}

class GraphProcessor:
    def __init__(self, llm_operations):
        self.llm_ops = llm_operations
        self.summarization_chain = LLMChain(llm=self.llm_ops.llm, prompt=SUMMARY_PROMPT)

    def _initial_clustering(self, state: ClusteringState) -> ClusteringState:
        logging.info("Starting initial clustering with Faiss...")
        graph = state['input_graph']
        similarity_threshold = state.get('similarity_threshold', 0.9)

        if not graph.vcount():
            state['error'] = "Input graph has no vertices."
            return state

        entities = []
        for v in graph.vs:
            entity = v.attributes()
            entity['id'] = v['name']
            entities.append(entity)

        relationships = []
        for e in graph.es:
            rel = e.attributes()
            rel["source"] = graph.vs[e.source]['name']
            rel["target"] = graph.vs[e.target]['name']
            relationships.append(rel)

        id_to_entity = {entity['id']: entity for entity in entities}
        entities_without_description = [e for e in entities if not e.get("description")]
        if entities_without_description:
            logging.info(f"Found {len(entities_without_description)} entities without description. Generating them now.")
            texts = [f"Generate a short description for an entity of type '{e.get('type', '')}' with name '{e.get('name', '')}'" for e in entities_without_description]
            summaries = self.summarization_chain.batch(texts)
            for i, entity in enumerate(entities_without_description):
                entity["description"] = summaries[i].get('text', '')

        entities_missing_embedding = [e for e in entities if not (e.get('embedding') and any(e.get('embedding', [])))]
        if entities_missing_embedding:
            logging.info(f"Found {len(entities_missing_embedding)} entities missing an embedding. Backfilling now...")
            texts = [e.get("description", e['id']) for e in entities_missing_embedding]
            ids = [e['id'] for e in entities_missing_embedding]
            embeddings_map = self.llm_ops.get_embeddings(texts, ids)
            for entity in entities_missing_embedding:
                entity_embeddings = embeddings_map.get(entity['id'])
                if entity_embeddings:
                    entity["embedding"] = entity_embeddings.get("semantic_search")
                    entity["cluster_embedding"] = entity_embeddings.get("clustering")
        
        logging.info(f"Generating embeddings for {len(relationships)} relationships...")
        rel_texts = [f"Relationship from {r['source']} to {r['target']} of type {r.get('type')}" for r in relationships]
        rel_ids = [hashlib.sha256(text.encode()).hexdigest() for text in rel_texts]
        rel_embeddings_map = self.llm_ops.get_embeddings(rel_texts, rel_ids)
        for i, rel in enumerate(relationships):
            rel_id = rel_ids[i]
            rel_embeddings = rel_embeddings_map.get(rel_id)
            if rel_embeddings:
                rel["embedding"] = rel_embeddings.get("semantic_search")

        clusterable_entities = [e for e in entities if e.get("type") not in ["Chunk", "Community"]]
        valid_indices = [i for i, e in enumerate(clusterable_entities) if e.get("cluster_embedding") and any(e.get('cluster_embedding', []))]
        
        if len(valid_indices) < 2:
            logging.warning("Not enough entities with embeddings to perform clustering.")
            state['final_graph'] = graph
            return state

        valid_entities = [clusterable_entities[i] for i in valid_indices]
        valid_embeddings = np.array([e["cluster_embedding"] for e in valid_entities]).astype('float32')
        faiss.normalize_L2(valid_embeddings)

        dimension = valid_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(valid_embeddings)
        logging.info(f"Performing Faiss range search for {len(valid_embeddings)} vectors...")
        lims, _, I = index.range_search(valid_embeddings, similarity_threshold)

        visited = set()
        clusters = []
        for i in range(len(valid_embeddings)):
            if i in visited:
                continue
            
            neighbors_indices = I[lims[i]:lims[i+1]]
            new_cluster_indices = set(neighbors_indices)
            
            cluster_entity_ids = {valid_entities[idx]['id'] for idx in new_cluster_indices}
            clusters.append(list(cluster_entity_ids))
            
            visited.update(new_cluster_indices)

        logging.info(f"Initial clustering with Faiss found {len(clusters)} clusters.")
        state['clusters'] = clusters
        state['id_to_entity'] = id_to_entity
        state['entity_id_to_source_text'] = {}
        state['entities'] = entities
        state['relationships'] = relationships
        return state

    def _prepare_llm_inputs(self, state: ClusteringState) -> ClusteringState:
        if state.get('error') or not state.get('clusters'):
            return state
        logging.info("Preparing inputs for parallel LLM calls...")
        cluster_info_list = []
        for cluster_member_ids in state['clusters']:
            cluster_text_parts = []
            for member_id in cluster_member_ids:
                member_entity = state['id_to_entity'].get(member_id)
                if member_entity:
                    properties = member_entity.get('properties', {})
                    entity_type = member_entity.get('type', '')
                    description = properties.get('description', '')
                    cluster_text_parts.append(f"Type: {entity_type}, Description: {description}, Properties: {json.dumps(properties)}")
            
            cluster_info_list.append({
                "member_ids": cluster_member_ids,
                "cluster_text": " ".join(cluster_text_parts),
                "id_to_entity": state['id_to_entity'],
                "entity_id_to_source_text": state['entity_id_to_source_text']
            })
        
        state['cluster_info_list'] = cluster_info_list
        return state

    def _generate_properties_parallel(self, state: ClusteringState) -> ClusteringState:
        if state.get('error') or not state.get('cluster_info_list'):
            return state
        
        cluster_info_list = state['cluster_info_list']
        total_clusters = len(cluster_info_list)
        logging.info(f"Generating properties for {total_clusters} clusters in parallel...")
        
        all_generated_properties = []
        max_workers = getattr(Config, 'MAX_WORKERS', 20)
        
        processed_count = 0
        start_time = time.time()
        ten_percent_step = total_clusters // 10 if total_clusters > 10 else 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_cluster = {executor.submit(_invoke_llm_for_cluster, self.llm_ops, info): info for info in cluster_info_list}
            
            for future in concurrent.futures.as_completed(future_to_cluster):
                try:
                    properties = future.result()
                    all_generated_properties.append(properties)
                except Exception as exc:
                    logging.error(f'A cluster generation raised an exception: {exc}')
                    all_generated_properties.append({})
                finally:
                    processed_count += 1
                    if processed_count % ten_percent_step == 0 or processed_count == total_clusters:
                        progress_percent = (processed_count / total_clusters) * 100
                        elapsed_time = time.time() - start_time
                        logging.info(
                            f"Cluster property generation progress: {progress_percent:.0f}% "
                            f"({processed_count}/{total_clusters} clusters processed in {elapsed_time:.2f} seconds)"
                        )

        state['all_generated_properties'] = all_generated_properties
        logging.info(f"Finished generating properties. Got results for {len(all_generated_properties)} clusters.")
        return state

    def _create_class_entities(self, state: ClusteringState) -> ClusteringState:
        if state.get('error'):
            return state
        logging.info("Creating and merging 'Class' entities...")
        name_to_class_entity = {}
        class_id_map = {}
        summaries_to_embed = []
        class_eids_to_embed = []

        new_class_proposals = []
        for idx, generated_properties in enumerate(state['all_generated_properties']):
            cluster_info = state['cluster_info_list'][idx]
            cluster_member_ids = cluster_info["member_ids"]
            class_name = generated_properties.get("name")

            if not class_name or not class_name.strip():
                logging.warning(f"Skipping class creation for cluster due to empty class name.")
                continue

            if "name" in generated_properties:
                generated_properties["class_name"] = generated_properties.pop("name")

            if class_name in name_to_class_entity:
                existing_class_eid = name_to_class_entity[class_name]["id"]
                for member_id in cluster_member_ids:
                    class_id_map[member_id] = existing_class_eid
            else:
                class_eid = generate_class_eid(class_name)
                if not class_eid:
                    continue
                
                proposal = {
                    "class_name": class_name,
                    "class_eid": class_eid,
                    "generated_properties": generated_properties,
                    "cluster_info": cluster_info,
                    "member_ids": cluster_member_ids
                }
                new_class_proposals.append(proposal)
                name_to_class_entity[class_name] = {"id": class_eid}

        if new_class_proposals:
            logging.info(f"Generating summaries for {len(new_class_proposals)} new classes in a single batch...")
            summarization_inputs = [{"text_chunk": prop["cluster_info"]["cluster_text"]} for prop in new_class_proposals]
            try:
                summaries = self.summarization_chain.batch(summarization_inputs)
            except Exception as e:
                logging.error(f"Batch summarization failed: {e}", exc_info=True)
                from langchain_core.messages import AIMessage
                summaries = [AIMessage(content="")] * len(new_class_proposals)

            for i, proposal in enumerate(new_class_proposals):
                summary = summaries[i].get('text', '')
                if not proposal["generated_properties"].get("description"):
                    proposal["generated_properties"]["description"] = summary
                
                summaries_to_embed.append(summary)
                class_eids_to_embed.append(proposal["class_eid"])
                
                class_entity = {
                    "id": proposal["class_eid"],
                    "type": "Class",
                    "properties": proposal["generated_properties"]
                }
                name_to_class_entity[proposal["class_name"]] = class_entity
                for member_id in proposal["member_ids"]:
                    class_id_map[member_id] = proposal["class_eid"]

        state['name_to_class_entity'] = name_to_class_entity
        state['class_id_map'] = class_id_map
        state['summaries_to_embed'] = summaries_to_embed
        state['class_eids_to_embed'] = class_eids_to_embed
        return state

    def _embed_class_summaries(self, state: ClusteringState) -> ClusteringState:
        if state.get('error') or not state.get('summaries_to_embed'):
            return state
        logging.info("Embedding new class summaries...")
        all_embeddings = self.llm_ops.get_embeddings(state['summaries_to_embed'], state['class_eids_to_embed'])
        name_to_class_entity = state['name_to_class_entity']
        
        for class_name, class_entity in name_to_class_entity.items():
            class_eid = class_entity["id"]
            embeddings = all_embeddings.get(class_eid, {})
            class_entity["clustering_embedding"] = embeddings.get("clustering", [0.0] * Config.EMBEDDING_DIMENSION)
            class_entity["retrieval_document_embedding"] = embeddings.get("semantic_search", [0.0] * Config.EMBEDDING_DIMENSION)
        
        state['name_to_class_entity'] = name_to_class_entity
        return state

    def _rebuild_graph_with_classes(self, state: ClusteringState) -> ClusteringState:
        if state.get('error'):
            return state
        logging.info("Rebuilding graph with new class structure...")
        
        new_entities = state['entities'].copy()
        new_entities.extend(state['name_to_class_entity'].values())

        new_graph = ig.Graph(directed=True)
        new_graph.add_vertices(len(new_entities))

        logging.info("--- DIAGNOSTIC: Rebuilding graph vertices. ---")
        for i, entity in enumerate(new_entities):
            v = new_graph.vs[i]
            if i < 5:
                logging.info(f"Entity {i} dictionary keys: {list(entity.keys())}")

            v['name'] = entity.get('id')
            v['type'] = entity.get('type')

            if 'properties' in entity and isinstance(entity.get('properties'), dict):
                for prop_key, prop_value in entity['properties'].items():
                    v[prop_key] = prop_value

            if entity.get('embedding') and any(entity.get('embedding', [])):
                v['embedding'] = entity['embedding']
            if entity.get('cluster_embedding') and any(entity.get('cluster_embedding', [])):
                v['cluster_embedding'] = entity['cluster_embedding']
            if entity.get('retrieval_document_embedding') and any(entity.get('retrieval_document_embedding', [])):
                v['retrieval_document_embedding'] = entity['retrieval_document_embedding']

        name_to_vertex = {v["name"]: v for v in new_graph.vs if v["name"] is not None}
        
        relationships = state['relationships']
        class_id_map = state['class_id_map']

        for rel in relationships:
            if rel.get("source") in class_id_map:
                rel["source"] = class_id_map[rel["source"]]
            if rel.get("target") in class_id_map:
                rel["target"] = class_id_map[rel["target"]]

        logging.info("--- DIAGNOSTIC: Rebuilding graph edges. ---")
        for rel in relationships:
            try:
                source_vertex = name_to_vertex[rel["source"]]
                target_vertex = name_to_vertex[rel["target"]]
                edge = new_graph.add_edge(source_vertex, target_vertex)
                
                for key, value in rel.items():
                    if key not in ['source', 'target']:
                        edge[key] = value

            except KeyError:
                logging.warning(f"Skipping relationship due to missing source/target in name_to_vertex map: {rel}")

        for v in new_graph.vs:
            if v["type"] not in ["Chunk", "Community", "Class"]:
                v["type"] = "Instance"
            
            class_id = class_id_map.get(v["name"])
            if class_id:
                try:
                    source_vertex = name_to_vertex[v["name"]]
                    target_vertex = name_to_vertex[class_id]
                    edge = new_graph.add_edge(source_vertex, target_vertex)
                    edge["type"] = "INSTANCE_OF"
                    edge["description"] = "Indicates that an entity is an instance of a specific class."
                except KeyError:
                     logging.warning(f"Could not create INSTANCE_OF link for {v['name']} to {class_id}")

        for rel in relationships:
            source_class_id = class_id_map.get(rel.get("source"))
            target_class_id = class_id_map.get(rel.get("target"))
            if source_class_id and target_class_id and source_class_id != target_class_id:
                try:
                    source_vertex = name_to_vertex[source_class_id]
                    target_vertex = name_to_vertex[target_class_id]
                    edge = new_graph.add_edge(source_vertex, target_vertex)
                    edge["type"] = rel.get("type")
                    if rel.get("embedding"):
                        edge["embedding"] = rel.get("embedding")
                    if rel.get("properties"):
                        for k, v_prop in rel.get("properties", {}).items():
                            edge[k] = v_prop
                except KeyError:
                    pass

        logging.info(f"Clustering complete. Result: {new_graph.vcount()} entities, {new_graph.ecount()} relationships.")
        state['final_graph'] = new_graph
        return state

    def cluster_and_merge_entities(self, graph: ig.Graph, similarity_threshold=0.9) -> ig.Graph:
        logging.info("Starting LangGraph workflow for clustering and merging entities.")
        workflow = StateGraph(ClusteringState)
        workflow.add_node("initial_clustering", self._initial_clustering)
        workflow.add_node("prepare_llm_inputs", self._prepare_llm_inputs)
        workflow.add_node("generate_properties_parallel", self._generate_properties_parallel)
        workflow.add_node("create_class_entities", self._create_class_entities)
        workflow.add_node("embed_class_summaries", self._embed_class_summaries)
        workflow.add_node("rebuild_graph_with_classes", self._rebuild_graph_with_classes)
        workflow.set_entry_point("initial_clustering")
        workflow.add_edge("initial_clustering", "prepare_llm_inputs")
        workflow.add_edge("prepare_llm_inputs", "generate_properties_parallel")
        workflow.add_edge("generate_properties_parallel", "create_class_entities")
        workflow.add_edge("create_class_entities", "embed_class_summaries")
        workflow.add_edge("embed_class_summaries", "rebuild_graph_with_classes")
        workflow.add_edge("rebuild_graph_with_classes", END)
        app = workflow.compile()
        initial_state: ClusteringState = {
            "input_graph": graph,
            "similarity_threshold": similarity_threshold,
            "entities": [], "relationships": [], "clusters": [], "id_to_entity": {},
            "entity_id_to_source_text": {},
            "all_generated_properties": [], "name_to_class_entity": {}, "class_id_map": {},
            "summaries_to_embed": [], "class_eids_to_embed": [],
            "final_graph": None, "error": None
        }
        final_state = app.invoke(initial_state)
        if final_state.get("error") or final_state.get("final_graph") is None:
            logging.error(f"LangGraph workflow failed or did not produce a final graph. Error: {final_state.get('error')}")
            return graph
        logging.info("LangGraph workflow for clustering completed successfully.")
        return final_state['final_graph']

    def deduplicate_entities(self, data):
        logging.info("Starting entity de-duplication process...")
        entities = data.get("entities", [])
        relationships = data.get("relationships", [])
        id_to_entity = {e["id"]: e for e in entities}
        eid_groups = {}
        for entity in entities:
            eid = entity["id"]
            if eid not in eid_groups:
                eid_groups[eid] = []
            eid_groups[eid].append(entity)
        final_entities = {}
        eids_to_remap = {}
        for eid, group in eid_groups.items():
            if len(group) == 1:
                final_entities[eid] = group[0]
                continue
            logging.warning(f"Found duplicate EID: '{eid}' for {len(group)} entities.")
            if all(e.get("type") == "Class" for e in group):
                logging.info(f"Handling duplicate Class EID: {eid}")
                instance_counts = {e["id"]: 0 for e in group}
                for rel in relationships:
                    if rel.get("type") == "INSTANCE_OF" and rel.get("target") in instance_counts:
                        instance_counts[rel.get("target")] += 1
                sorted_classes = sorted(group, key=lambda e: instance_counts[e["id"]], reverse=True)
                winner = sorted_classes[0]
                losers = sorted_classes[1:]
                final_entities[winner["id"]] = winner
                for loser in losers:
                    eids_to_remap[loser["id"]] = winner["id"]
                    winner["properties"].get("communities", []).extend(loser["properties"].get("communities", []))
                    logging.info(f"Merging class '{loser['id']}' into '{winner['id']}'.")
            elif all(e.get("type") == "Instance" for e in group):
                logging.info(f"Handling duplicate Instance EID: {eid}")
                winner = group[0]
                final_entities[eid] = winner
                for i, duplicate in enumerate(group[1:]):
                    original_id = duplicate["id"]
                    while True:
                        new_eid = f"{original_id}_{uuid.uuid4().hex[:6]}"
                        if new_eid not in id_to_entity and new_eid not in final_entities:
                            break
                    eids_to_remap[original_id] = new_eid
                    duplicate["id"] = new_eid
                    final_entities[new_eid] = duplicate
                    winner["properties"].get("communities", []).extend(duplicate["properties"].get("communities", []))
                    logging.info(f"Renamed duplicate instance '{original_id}' to '{new_eid}'.")
            else:
                logging.warning(f"Unhandled duplicate EID case for eid '{eid}'. Renaming duplicates.")
                if group:
                    winner = group[0]
                    final_entities[winner["id"]] = winner
                    for duplicate in group[1:]:
                        original_id = duplicate["id"]
                        while True:
                            new_eid = f"{original_id}_{uuid.uuid4().hex[:6]}"
                            if new_eid not in id_to_entity and new_eid not in final_entities:
                                break
                        eids_to_remap[original_id] = new_eid
                        duplicate["id"] = new_eid
                        final_entities[new_eid] = duplicate
                        winner["properties"].get("communities", []).extend(duplicate["properties"].get("communities", []))
                        logging.info(f"Renamed duplicate entity '{original_id}' of type '{duplicate.get('type')}' to '{new_eid}'.")
        logging.info(f"EIDs to remap: {eids_to_remap}")
        for rel in relationships:
            if isinstance(rel, dict) and "source" in rel and "target" in rel:
                try:
                    if rel["source"] in eids_to_remap:
                        rel["source"] = eids_to_remap[rel["source"]]
                    if rel["target"] in eids_to_remap:
                        rel["target"] = eids_to_remap[rel["target"]]
                except KeyError as e:
                    logging.error(f"KeyError processing relationship {rel}: {e}", exc_info=True)
                    continue
            else:
                logging.warning(f"Skipping malformed relationship: {rel}.")
        data["entities"] = list(final_entities.values())
        data["relationships"] = relationships
        logging.info(f"De-duplication complete. Result: {len(data['entities'])} entities.")
        return data

    def run_igraph_community_detection(self, graph: ig.Graph):
        if not graph.vcount():
            return graph
        communities = []
        algorithm = Config.COMMUNITY_DETECTION_ALGORITHM
        logging.info(f"Running community detection with algorithm: {algorithm}")
        if algorithm == 'maximal_cliques':
            communities = graph.maximal_cliques()
            logging.info(f"Found {len(communities)} cliques (overlapping communities).")
        elif algorithm == 'slpa':
            if graph.vcount() > 0:
                original_names = graph.vs["name"] if "name" in graph.vs.attributes() else [str(i) for i in range(graph.vcount())]
                graph.vs["name"] = [str(n) for n in original_names]
                name_to_idx = {name: i for i, name in enumerate(graph.vs["name"])}
                coms_from_cdlib = algorithms.slpa(graph, t=21, r=0.1)
                for community_names in coms_from_cdlib.communities:
                    communities.append([name_to_idx[name] for name in community_names if name in name_to_idx])
                logging.info(f"Found {len(communities)} overlapping communities using SLPA.")
        else:
            logging.error(f"Unknown community detection algorithm: {algorithm}. Skipping community detection.")
            return graph
        community_summaries = {}
        if "communities" not in graph.vs.attributes():
            graph.vs["communities"] = [[] for _ in range(graph.vcount())]
        for i, v in enumerate(graph.vs):
            entity_id = v["name"]
            entity_summary_parts = []
            if v.attributes().get("type"):
                entity_summary_parts.append(f"Type: {v.attributes().get('type')}")
            summary = v.attributes().get("summary")
            name = v.attributes().get("name")
            if summary:
                entity_summary_parts.append(f"Summary: {summary}")
            elif name:
                entity_summary_parts.append(f"Name: {name}")
            entity_text_for_summary = ", ".join(entity_summary_parts) if entity_summary_parts else entity_id
            for j, community_indices in enumerate(communities):
                if i in community_indices:
                    community_id = f"{algorithm}_{j}"
                    v["communities"].append(community_id)
                    if community_id not in community_summaries:
                        community_summaries[community_id] = []
                    community_summaries[community_id].append(entity_text_for_summary)
        community_creation_data = []
        for comm_id, entity_texts in community_summaries.items():
            community_description_parts = [f"An entity with the following properties: {entity_text}" for entity_text in entity_texts]
            entities_description = ". ".join(community_description_parts)
            entities_description += f". These entities form a community, which can be summarized as:"
            summary_response = self.summarization_chain.invoke({"text_chunk": entities_description})
            full_community_summary = summary_response.get('text', '')
            if not full_community_summary:
                logging.warning(f"Skipping Community entity creation for {comm_id} due to empty summary.")
                continue
            community_creation_data.append({"id": comm_id, "summary": full_community_summary, "entity_texts": entity_texts})
        if community_creation_data:
            community_ids = [c['id'] for c in community_creation_data]
            community_summaries_for_embedding = [c['summary'] for c in community_creation_data]
            all_community_embeddings = self.llm_ops.get_embeddings(community_summaries_for_embedding, community_ids)
            if not all_community_embeddings:
                logging.warning("Failed to get embeddings for any communities.")
            else:
                for community_data in community_creation_data:
                    comm_id = community_data['id']
                    community_embeddings = all_community_embeddings.get(comm_id)
                    if not community_embeddings:
                        logging.warning(f"Skipping Community entity creation for {comm_id} due to missing embeddings.")
                        continue
                    community_vertex = graph.add_vertex(name=comm_id)
                    community_vertex["type"] = "Community"
                    community_vertex["Summary"] = community_data['summary']
                    community_vertex["cluster_embedding"] = community_embeddings.get("clustering", [0.0] * Config.EMBEDDING_DIMENSION)
                    community_vertex["retrieval_document_embedding"] = community_embeddings.get("semantic_search", [0.0] * Config.EMBEDDING_DIMENSION)
                    community_vertex["embedding"] = community_embeddings.get("semantic_search", [0.0] * Config.EMBEDDING_DIMENSION)
                    community_vertex["communities"] = []
        logging.info(f"Created {len(community_creation_data)} standard Community entities.")
        return graph

    def remove_entities_with_null_keys_and_relationships(self, graph: ig.Graph):
        logging.info("Starting removal of entities with null/empty IDs and their relationships...")
        if not graph.vcount():
            return graph
        vertices_to_remove = []
        for v in graph.vs:
            entity_id = v["name"]
            if entity_id is None or str(entity_id).strip() == "":
                vertices_to_remove.append(v.index)
                logging.warning(f"Removing entity with null/empty ID: {v.attributes()}")
        if vertices_to_remove:
            graph.delete_vertices(vertices_to_remove)
            logging.info(f"Removed {len(vertices_to_remove)} entities with null/empty IDs.")
        else:
            logging.info("No entities with null/empty IDs found.")
        return graph

    def remove_none_attributes(self, graph: ig.Graph) -> ig.Graph:
        logging.info("Removing None attributes from graph...")
        for v in graph.vs:
            attrs_to_delete = [key for key, value in v.attributes().items() if value is None]
            for key in attrs_to_delete:
                del v[key]
        for e in graph.es:
            attrs_to_delete = [key for key, value in e.attributes().items() if value is None]
            for key in attrs_to_delete:
                del e[key]
        logging.info("Finished removing None attributes.")
        return graph

    def backfill_entity_embeddings(self, graph: ig.Graph) -> ig.Graph:
        """
        Finds entities missing core embeddings and generates them.
        """
        if not graph.vcount():
            logging.info("No entities in the graph to backfill embeddings for.")
            return graph

        vertices_to_embed = []
        for v in graph.vs:
            # We only backfill for entities that are not communities themselves
            if v.attributes().get("type") == "Community":
                continue
            
            # Check if core embeddings are missing.
            if v.attributes().get("embedding") is None or v.attributes().get("cluster_embedding") is None:
                 vertices_to_embed.append(v)

        if not vertices_to_embed:
            logging.info("All entities already have core embeddings.")
            return graph

        logging.info(f"Found {len(vertices_to_embed)} entities missing core embeddings. Backfilling now...")

        texts_to_embed = [v.attributes().get("description") or v.attributes().get("name") or v["name"] for v in vertices_to_embed]
        ids_to_embed = [v["name"] for v in vertices_to_embed]

        embeddings_map = self.llm_ops.get_embeddings(texts_to_embed, ids_to_embed)

        updated_count = 0
        for v in vertices_to_embed:
            entity_id = v["name"]
            entity_embeddings = embeddings_map.get(entity_id)
            if entity_embeddings:
                updated_count += 1
                # Backfill all embedding types from the result
                if v.attributes().get("embedding") is None:
                    v["embedding"] = entity_embeddings.get("semantic_search")
                if v.attributes().get("cluster_embedding") is None:
                    v["cluster_embedding"] = entity_embeddings.get("clustering")
                # Also backfill retrieval if it's missing, using semantic as a fallback
                if v.attributes().get("retrieval_document_embedding") is None:
                     v["retrieval_document_embedding"] = entity_embeddings.get("semantic_search")

        logging.info(f"Successfully generated and backfilled embeddings for {updated_count} entities.")
        return graph

    def backfill_relationship_embeddings(self, graph: ig.Graph) -> ig.Graph:
        """
        Finds relationships missing an embedding and generates it.
        """
        if not graph.ecount():
            logging.info("No relationships in the graph to backfill embeddings for.")
            return graph

        edges_to_embed = []
        for edge in graph.es:
            if "embedding" not in edge.attributes() or edge["embedding"] is None:
                edges_to_embed.append(edge)

        if not edges_to_embed:
            logging.info("All relationships already have embeddings.")
            return graph

        logging.info(f"Found {len(edges_to_embed)} relationships missing an embedding. Backfilling now...")

        rel_texts = []
        rel_ids = []
        for edge in edges_to_embed:
            source_name = graph.vs[edge.source]["name"]
            target_name = graph.vs[edge.target]["name"]
            rel_type = edge.attributes().get("type", "RELATED_TO")
            text = f"Relationship from {source_name} to {target_name} of type {rel_type}"
            rel_texts.append(text)
            # Create a unique ID for the relationship to use in the embedding map
            rel_ids.append(hashlib.sha256(text.encode()).hexdigest())

        rel_embeddings_map = self.llm_ops.get_embeddings(rel_texts, rel_ids)

        for i, edge in enumerate(edges_to_embed):
            rel_id = rel_ids[i]
            rel_embeddings = rel_embeddings_map.get(rel_id)
            if rel_embeddings:
                edge["embedding"] = rel_embeddings.get("semantic_search")

        logging.info(f"Successfully generated and backfilled embeddings for {len(edges_to_embed)} relationships.")
        return graph


def generate_class_eid(name):
    if not name:
        return None
    return hashlib.sha256(name.encode('utf-8')).hexdigest()

def _graph_to_dict(graph: ig.Graph) -> dict:
    entities = []
    for v in graph.vs:
        all_attrs = v.attributes()
        entity_properties = {k: v for k, v in all_attrs.items() if k not in ['name', 'type', 'embedding', 'cluster_embedding', 'retrieval_document_embedding'] and v is not None}
        entity = {
            "id": all_attrs.get("name"),
            "type": all_attrs.get("type"),
            "properties": entity_properties
        }
        if "embedding" in all_attrs:
            entity["embedding"] = all_attrs["embedding"]
        if "cluster_embedding" in all_attrs:
            entity["cluster_embedding"] = all_attrs["cluster_embedding"]
        if "retrieval_document_embedding" in all_attrs:
            entity["retrieval_document_embedding"] = all_attrs["retrieval_document_embedding"]
        entities.append(entity)
    relationships = []
    for e in graph.es:
        all_attrs = e.attributes()
        rel_properties = {k: v for k, v in all_attrs.items() if k not in ['source', 'target', 'type', 'embedding'] and v is not None}
        rel = {
            "source": graph.vs[e.source]["name"],
            "target": graph.vs[e.target]["name"],
            "type": all_attrs.get("type"),
            "properties": rel_properties
        }
        if "embedding" in all_attrs:
            rel["embedding"] = all_attrs["embedding"]
        relationships.append(rel)
    return {"entities": entities, "relationships": relationships}

def _dict_to_graph(data: dict) -> ig.Graph:
    graph = ig.Graph(directed=True)
    for entity in data.get("entities", []):
        vertex_attrs = {k: v for k, v in entity.items() if k != 'id'}
        graph.add_vertex(name=entity["id"], **vertex_attrs)
    for rel in data.get("relationships", []):
        try:
            source_vertex = graph.vs.find(name=rel["source"])
            target_vertex = graph.vs.find(name=rel["target"])
            edge_attrs = {k: v for k, v in rel.items() if k not in ['source', 'target']}
            graph.add_edge(source_vertex, target_vertex, **edge_attrs)
        except ValueError as e:
            logging.warning(f"Could not add edge for relationship {rel}: {e}")
    return graph
