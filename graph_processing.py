# Graph processing for the consolidator function
import logging
import igraph as ig
import numpy as np
import uuid
import json
import re
import hashlib
import base64
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.chains import LLMChain

from llm_operations import LLMOperations, CLASS_PROPERTY_GENERATION_PROMPT, SUMMARY_PROMPT, CLASS_SCHEMA
from config import Config

class GraphProcessor:
    def __init__(self, llm_operations):
        self.llm_ops = llm_operations

    def aggregate_results(self, data):
        all_entities = {}
        all_relationships = []
        for res_str in data["partial_results"]:
            res_json = json.loads(res_str)
            for entity in res_json.get("extracted_graph_data", {}).get("entities", []):
                entity_id = entity.get("id")
                if entity_id:
                    all_entities[entity_id] = entity
                else:
                    logging.warning(f"Skipping entity without id: {entity}")
            all_relationships.extend(res_json.get("extracted_graph_data", {}).get("relationships", []))
        
        logging.info(f"Aggregated {len(all_entities)} entities and {len(all_relationships)} relationships.")

        return {
            "batch_id": data["batch_id"],
            "entities": list(all_entities.values()),
            "relationships": all_relationships
        }

    def cluster_and_merge_entities(self, data, similarity_threshold=0.9):
        """
        Clusters similar entities, creates Class nodes with name-based IDs, and merges
        classes with the same name by re-linking their instances.
        """
        entities = data.get("entities", [])
        relationships = data.get("relationships", [])

        if not entities:
            return data

        id_to_entity = {entity['id']: entity for entity in entities}

        entity_id_to_source_text = {}
        for rel in relationships:
            if rel.get("type") == "ARE_PART_OF_CHUNK":
                entity_id = rel.get("source")
                chunk_id = rel.get("target")
                if entity_id and chunk_id:
                    chunk_entity = id_to_entity.get(chunk_id)
                    if chunk_entity and chunk_entity.get("type") == "Chunk":
                        original_text = chunk_entity.get("properties", {}).get("original_text")
                        if original_text:
                            entity_id_to_source_text[entity_id] = original_text

        clusterable_entities = [e for e in entities if e.get("type") not in ["Chunk", "Community"]]
        entity_ids = [e["id"] for e in clusterable_entities]
        embeddings = np.array([e.get("embedding") for e in clusterable_entities])

        valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None and len(emb) > 0]
        if len(valid_indices) < 2:
            logging.warning("Not enough entities with embeddings to perform clustering.")
            return data
            
        valid_embeddings = embeddings[valid_indices]
        valid_entity_ids = [entity_ids[i] for i in valid_indices]
        
        similarity_matrix = cosine_similarity(valid_embeddings)

        visited = [False] * len(valid_entity_ids)
        clusters = []
        for i in range(len(valid_entity_ids)):
            if visited[i]:
                continue
            cluster = [valid_entity_ids[i]]
            visited[i] = True
            for j in range(i + 1, len(valid_entity_ids)):
                if not visited[j] and similarity_matrix[i][j] > similarity_threshold:
                    cluster.append(valid_entity_ids[j])
                    visited[j] = True
            clusters.append(cluster)

        class_property_chain = LLMChain(llm=self.llm_ops.llm, prompt=CLASS_PROPERTY_GENERATION_PROMPT)
        summarization_chain = LLMChain(llm=self.llm_ops.llm, prompt=SUMMARY_PROMPT)

        name_to_class_entity = {}
        class_id_map = {}

        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            future_to_cluster = {}
            for cluster_member_ids in clusters:
                instances_text_parts = []
                source_text_parts = set()
                cluster_text_parts = []

                for member_id in cluster_member_ids:
                    member_entity = id_to_entity.get(member_id)
                    if member_entity:
                        properties = member_entity.get('properties', {})
                        instances_text_parts.append(f"- {json.dumps(properties)}")
                        source_text = entity_id_to_source_text.get(member_id)
                        if source_text:
                            source_text_parts.add(source_text)
                        
                        entity_type = member_entity.get('type', '')
                        cluster_text_parts.append(f"Type: {entity_type}, Properties: {json.dumps(properties)}")

                instances_text = "\n".join(instances_text_parts)
                source_text_context = "\n\n---\n\n".join(source_text_parts)
                schema_str = json.dumps(CLASS_SCHEMA, indent=2)
                
                future = executor.submit(self.llm_ops.generate_class_properties, class_property_chain, instances_text, schema_str, source_text_context)
                future_to_cluster[future] = {
                    "member_ids": cluster_member_ids,
                    "cluster_text": " ".join(cluster_text_parts)
                }

            processed_clusters = 0
            total_clusters = len(clusters)
            for future in as_completed(future_to_cluster):
                cluster_info = future_to_cluster[future]
                cluster_member_ids = cluster_info["member_ids"]
                processed_clusters += 1
                try:
                    generated_properties_str = future.result()
                    extracted_json_str = self.llm_ops.extract_json_from_llm_response(generated_properties_str)
                    generated_properties = json.loads(extracted_json_str)
                    class_name = generated_properties.get("name")
                    if not class_name or not class_name.strip():
                        logging.warning(f"Skipping class creation for cluster due to empty class name. Cluster info: {cluster_info}")
                        continue
                    class_eid = generate_class_eid(class_name)

                    if not class_eid:
                        logging.warning(f"Could not generate a valid EID for class from name: '{class_name}'. Skipping cluster.")
                        continue

                    if class_name in name_to_class_entity:
                        existing_class_eid = name_to_class_entity[class_name]["id"]
                        for member_id in cluster_member_ids:
                            class_id_map[member_id] = existing_class_eid
                        logging.info(f"Merged cluster into existing class '{class_name}' (ID: {existing_class_eid})")
                    else:
                        summary = summarization_chain.invoke({"text_chunk": cluster_info["cluster_text"]}).get("text")
                        embedding = self.llm_ops.get_embedding(summary, class_eid)
                        class_entity = {
                            "id": class_eid,
                            "type": "Class",
                            "properties": generated_properties,
                            "embedding": embedding
                        }
                        name_to_class_entity[class_name] = class_entity
                        for member_id in cluster_member_ids:
                            class_id_map[member_id] = class_eid
                        logging.info(f"Created new class '{class_name}' (ID: {class_eid})")

                except Exception as e:
                    logging.error(f"Failed to process future for cluster: {e}", exc_info=True)
                
                if total_clusters > 0 and processed_clusters % (total_clusters // 10 or 1) == 0:
                    progress = (processed_clusters / total_clusters) * 100
                    logging.info(f"Class property generation progress: {progress:.0f}% ({processed_clusters}/{total_clusters})")


        new_entities = list(name_to_class_entity.values())
        new_relationships = []

        for entity in entities:
            if entity.get("type") not in ["Chunk", "Community"]:
                entity["type"] = "Instance"
            new_entities.append(entity)
            
            class_id = class_id_map.get(entity["id"])
            if class_id:
                new_relationships.append({
                    "source": entity["id"],
                    "target": class_id,
                    "type": "INSTANCE_OF",
                    "properties": {"description": "Indicates that an entity is an instance of a specific class."}
                })

        for rel in relationships:
            source_class_id = class_id_map.get(rel.get("source"))
            target_class_id = class_id_map.get(rel.get("target"))
            if source_class_id and target_class_id and source_class_id != target_class_id:
                new_relationships.append({
                    "source": source_class_id,
                    "target": target_class_id,
                    "type": rel.get("type"),
                    "properties": rel.get("properties", {})
                })

        data["entities"] = new_entities
        data["relationships"] = new_relationships
        
        logging.info(f"Clustering complete. Result: {len(new_entities)} entities, {len(new_relationships)} relationships.")
        return data

    def deduplicate_entities(self, data):
        """Finds and resolves duplicate Eids before community detection."""
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
                    logging.info(f"Merging class '{loser['id']}' into '{winner['id']}'.")

            elif all(e.get("type") == "Instance" for e in group):
                logging.info(f"Handling duplicate Instance EID: {eid}")
                final_entities[eid] = group[0]
                for i, duplicate in enumerate(group[1:]):
                    original_id = duplicate["id"]
                    while True:
                        new_eid = f"{original_id}_{uuid.uuid4().hex[:6]}"
                        if new_eid not in id_to_entity and new_eid not in final_entities:
                            break
                    
                    eids_to_remap[original_id] = new_eid
                    duplicate["id"] = new_eid
                    final_entities[new_eid] = duplicate
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
                        logging.info(f"Renamed duplicate entity '{original_id}' of type '{duplicate.get('type')}' to '{new_eid}'.")

        for rel in relationships:
            # Ensure rel is a dictionary and has 'source' and 'target' keys before processing
            if isinstance(rel, dict) and "source" in rel and "target" in rel:
                if rel["source"] in eids_to_remap:
                    rel["source"] = eids_to_remap[rel["source"]]
                if rel["target"] in eids_to_remap:
                    rel["target"] = eids_to_remap[rel["target"]]
            else:
                logging.warning(f"Skipping malformed relationship: {rel}. Missing 'source' or 'target' or not a dictionary.")

        data["entities"] = list(final_entities.values())
        logging.info(f"De-duplication complete. Result: {len(data['entities'])} entities.")
        return data

    def run_igraph_community_detection(self, data):
        logging.info("Running igraph community detection...")
        entities = data.get("entities", [])
        relationships = data.get("relationships", [])

        id_to_vertex = {entity["id"]: i for i, entity in enumerate(entities)}
        
        g = ig.Graph(directed=False)
        g.add_vertices(len(entities))
        g.vs["id"] = [entity["id"] for entity in entities]
        g.vs["type"] = [entity["type"] for entity in entities]
        g.vs["properties"] = [entity["properties"] for entity in entities]
        g.vs["embedding"] = [entity.get("embedding") for entity in entities]

        edges = []
        for rel in relationships:
            source_id = rel.get("source")
            target_id = rel.get("target")
            if source_id in id_to_vertex and target_id in id_to_vertex:
                edges.append((id_to_vertex[source_id], id_to_vertex[target_id]))
        g.add_edges(edges)

        cliques = g.maximal_cliques()
        
        community_summaries = {}
        for entity in entities:
            entity_id = entity["id"]
            entity["communities"] = []
            
            entity_summary_parts = []
            if entity.get("type"):
                entity_summary_parts.append(f"Type: {entity.get('type')}")
            if entity.get("properties", {}).get("summary"):
                entity_summary_parts.append(f"Summary: {entity['properties']['summary']}")
            elif entity.get("properties", {}).get("name"):
                entity_summary_parts.append(f"Name: {entity['properties']['name']}")
            
            entity_text_for_summary = ", ".join(entity_summary_parts) if entity_summary_parts else entity_id

            for i, clique in enumerate(cliques):
                if id_to_vertex.get(entity_id) is not None and id_to_vertex[entity_id] in clique:
                    community_id = f"clique_{i}"
                    entity["communities"].append(community_id)
                    
                    if community_id not in community_summaries:
                        community_summaries[community_id] = []
                    community_summaries[community_id].append(entity_text_for_summary)

        for comm_id, entity_texts in community_summaries.items():
            full_community_summary = " ".join(entity_texts) 
            
            if full_community_summary:
                all_embeddings = self.llm_ops._get_single_embedding(full_community_summary, comm_id)
                semantic_search_embedding = all_embeddings.get("semantic_search", [0.0] * Config.EMBEDDING_DIMENSION)
            else:
                semantic_search_embedding = [0.0] * Config.EMBEDDING_DIMENSION
            
            community_entity = {
                "id": comm_id,
                "type": "Community",
                "properties": {
                    "summary": full_community_summary,
                    "community_type": "structural"
                },
                "embedding": semantic_search_embedding, # Use semantic search embedding for the main embedding field
                "communities": []
            }
            entities.append(community_entity)

        logging.info(f"Found {len(cliques)} cliques (overlapping communities) and created {len(community_summaries)} standard Community entities.")
        return data

def generate_class_eid(name):
    """Creates a consistent and unique ID from a string using SHA256 hash."""
    if not name:
        return None
    # Use SHA256 hash to create a consistent and unique ID
    return hashlib.sha256(name.encode('utf-8')).hexdigest()