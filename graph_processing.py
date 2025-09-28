# Graph processing for the consolidator function
import logging
import igraph as ig
import numpy as np
import uuid
import json
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chains import LLMChain

from config import Config
from llm_operations import SUMMARY_PROMPT, CLASS_SCHEMA
import hashlib

class GraphProcessor:
    """
    The GraphProcessor class is responsible for various graph-related operations within the consolidator function.
    This includes aggregating partial results, clustering and merging entities, deduplicating entities,
    and performing community detection.
    """
    def __init__(self, llm_operations):
        self.llm_ops = llm_operations
        self.summarization_chain = LLMChain(llm=self.llm_ops.llm, prompt=SUMMARY_PROMPT)

    def cluster_and_merge_entities(self, graph: ig.Graph, similarity_threshold=0.9):
        """
        Clusters similar entities based on their embeddings, creates 'Class' nodes for these clusters,
        and merges classes with the same name by re-linking their instances.
        This process aims to reduce redundancy and create a more abstract representation of entities.

        Args:
            graph (igraph.Graph): The input graph containing entities and relationships.
            similarity_threshold (float): The cosine similarity threshold for grouping entities into clusters.

        Returns:
            igraph.Graph: The updated graph with new 'Class' entities and 'INSTANCE_OF' relationships.
        """
        if not graph.vcount():
            return graph

        # Extract entities and relationships from the graph
        entities = []
        for v in graph.vs:
            # Start with all attributes, then refine
            all_attributes = v.attributes()
            
            # Extract 'properties' and parse it from JSON string to dict
            properties_json_str = all_attributes.get("properties", "{}")
            if properties_json_str is None:
                properties_json_str = "{}"
            try:
                parsed_properties = json.loads(properties_json_str)
            except json.JSONDecodeError:
                logging.warning(f"Could not decode properties JSON for entity {v['name']}: {properties_json_str}")
                parsed_properties = {} # Default to empty dict on error

            entity = {
                "id": v["name"],
                "type": all_attributes.get("type"), # Use .get for safety
                "properties": parsed_properties
            }
            
            # Add other relevant attributes directly to the entity if they are not 'name', 'type', or 'properties'
            for key, value in all_attributes.items():
                if key not in ["name", "type", "properties", "embedding", "cluster_embedding", "retrieval_document_embedding"]:
                    entity["properties"][key] = value
            
            # Handle embeddings separately as they are not part of 'properties'
            if "embedding" in all_attributes:
                entity["embedding"] = all_attributes["embedding"]
            if "cluster_embedding" in all_attributes:
                entity["cluster_embedding"] = all_attributes["cluster_embedding"]
            if "retrieval_document_embedding" in all_attributes:
                entity["retrieval_document_embedding"] = all_attributes["retrieval_document_embedding"]

            entities.append(entity)
        relationships = []
        for e in graph.es:
            rel = {
                "source": graph.vs[e.source]["name"],
                "target": graph.vs[e.target]["name"],
                "type": e["type"] if "type" in e.attributes() else None,
                "properties": e.attributes()
            }
            rel["properties"].pop("type", None)
            relationships.append(rel)

        id_to_entity = {entity['id']: entity for entity in entities}
        entity_id_to_source_text = {} # Initialize as empty for now, needs to be populated from graph attributes if available

        # Filter out 'Chunk' and 'Community' entities as they are not subject to clustering in this step.
        clusterable_entities = [e for e in entities if e.get("type") not in ["Chunk", "Community"]]
        entity_ids = [e["id"] for e in clusterable_entities]
        embeddings = np.array([e.get("cluster_embedding") for e in clusterable_entities])

        # Identify valid embeddings (non-None and non-empty) for clustering.
        valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None and len(emb) > 0]
        
        total_entities = len(clusterable_entities)
        entities_with_embeddings = len(valid_indices)
        entities_without_embeddings = total_entities - entities_with_embeddings

        logging.info(f"Total clusterable entities: {total_entities}")
        logging.info(f"Entities with embeddings: {entities_with_embeddings}")
        logging.info(f"Entities without embeddings: {entities_without_embeddings}")

        if entities_without_embeddings > 0:
            missing_embedding_examples = []
            for i, emb in enumerate(embeddings):
                if emb is None or len(emb) == 0:
                    missing_embedding_examples.append(clusterable_entities[i].get('id', 'N/A'))
                if len(missing_embedding_examples) >= 5:
                    break
            logging.warning(f"Examples of entities missing embeddings: {missing_embedding_examples}")

        if len(valid_indices) < 2:
            logging.warning("Not enough entities with embeddings to perform clustering.")
            return graph
            
        valid_embeddings = embeddings[valid_indices]
        valid_entity_ids = [entity_ids[i] for i in valid_indices]
        
        # Calculate cosine similarity between entity embeddings to identify similar entities.
        similarity_matrix = cosine_similarity(valid_embeddings)

        # Perform a simple greedy clustering based on the similarity matrix.
        # Entities are grouped into clusters if their similarity exceeds the threshold.
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

        logging.info(f"Found {len(clusters)} clusters.")
        for i, cluster in enumerate(clusters):
            logging.info(f"Cluster {i}: {cluster}")

        # Initialize LLM chains for generating class properties and summaries.
        # class_property_chain = LLMChain(llm=self.llm_ops.llm, prompt=CLASS_PROPERTY_GENERATION_PROMPT)
        # summarization_chain = LLMChain(llm=self.llm_ops.llm, prompt=SUMMARY_PROMPT)

        name_to_class_entity = {}
        class_id_map = {}

        batched_llm_inputs = []
        cluster_info_list = [] # To store info for mapping results back to original clusters

        # Prepare batched inputs for LLM calls to generate class properties.
        # This optimizes LLM usage by sending multiple requests in a single batch.
        for cluster_member_ids in clusters:
            instances_text_parts = []
            source_text_parts = set()
            cluster_text_parts = []

            for member_id in cluster_member_ids:
                member_entity = id_to_entity.get(member_id)
                if member_entity:
                    properties = member_entity.get('properties', {})
                    instances_text_parts.append(f"- {json.dumps(properties)}")
                    source_text = entity_id_to_source_text.get(member_id) # This will be empty for now
                    if source_text:
                        source_text_parts.add(source_text)
                    
                    entity_type = member_entity.get('type', '')
                    cluster_text_parts.append(f"Type: {entity_type}, Properties: {json.dumps(properties)}")

            instances_text = "\n".join(instances_text_parts)
            source_text_context = "\n\n---\n\n".join(source_text_parts)
            schema_str = json.dumps(CLASS_SCHEMA, indent=2)
            
            batched_llm_inputs.append({
                "instances_text": instances_text,
                "schema_str": schema_str,
                "source_text_context": source_text_context
            })
            cluster_info_list.append({
                "member_ids": cluster_member_ids,
                "cluster_text": " ".join(cluster_text_parts)
            })

        processed_clusters = 0
        total_clusters = len(clusters)
        all_generated_properties = []

        # Process clusters in batches to generate class properties using the LLM.
        for i in range(0, total_clusters, Config.LLM_BATCH_SIZE):
            batch_inputs = batched_llm_inputs[i:i + Config.LLM_BATCH_SIZE]
            # batch_cluster_info = cluster_info_list[i:i + Config.LLM_BATCH_SIZE]

            try:
                # Call the LLM with the batched inputs to generate properties for the classes.
                generated_properties_str = self.llm_ops.generate_class_properties(batch_inputs, CLASS_SCHEMA)
                extracted_json_str = self.llm_ops.extract_json_from_llm_response(generated_properties_str)
                
                try:
                    batch_generated_properties = json.loads(extracted_json_str)
                except json.JSONDecodeError as json_e:
                    logging.error(f"JSONDecodeError: Failed to decode extracted_json_str. Error: {json_e}")
                    logging.error(f"Problematic generated_properties_str: {generated_properties_str}")
                    logging.error(f"Problematic extracted_json_str: {extracted_json_str}")
                    raise # Re-raise to be caught by the outer exception handler
                
                if not isinstance(batch_generated_properties, list):
                    logging.error(f"LLM did not return a list of properties for batch starting at index {i}. Response: {extracted_json_str}")
                    batch_generated_properties = [{{}} for _ in batch_inputs] # Fill with empty dicts to avoid index errors
                
                all_generated_properties.extend(batch_generated_properties)

            except Exception as e:
                logging.error(f"Failed to process LLM batch starting at index {i}: {e}", exc_info=True)
                # On error, extend with empty dictionaries to maintain list length and prevent index errors
                all_generated_properties.extend([{}] * len(batch_inputs))

            processed_clusters += len(batch_inputs)
            if total_clusters > 0 and processed_clusters % (total_clusters // 10 or 1) == 0:
                progress = (processed_clusters / total_clusters) * 100
                logging.info(f"Class property generation progress: {progress:.0f}% ({processed_clusters}/{total_clusters})")

        logging.info(f"Generated properties for {len(all_generated_properties)} clusters.")
        for i, props in enumerate(all_generated_properties):
            logging.info(f"Cluster {i} properties: {props}")

        # Iterate through the generated properties and create or merge 'Class' entities.
        summaries_to_embed = []
        class_eids_to_embed = []
        for idx, generated_properties in enumerate(all_generated_properties):
            cluster_info = cluster_info_list[idx]
            cluster_member_ids = cluster_info["member_ids"]
            
            try:
                class_name = generated_properties.get("name")
                if not class_name or not class_name.strip():
                    logging.warning(f"Skipping class creation for cluster due to empty class name. Cluster info: {cluster_info}")
                    continue
                # Generate a consistent and unique EID for the class based on its name.
                class_eid = generate_class_eid(class_name)

                if not class_eid:
                    logging.warning(f"Could not generate a valid EID for class from name: '{class_name}'. Skipping cluster.")
                    continue

                # If a class with the same name already exists, merge the current cluster's members
                # into the existing class by re-linking their instances.
                if class_name in name_to_class_entity:
                    existing_class_eid = name_to_class_entity[class_name]["id"]
                    for member_id in cluster_member_ids:
                        class_id_map[member_id] = existing_class_eid
                    logging.info(f"Merged cluster into existing class '{class_name}' (ID: {existing_class_eid})")
                else:
                    # If it's a new class, create a new 'Class' entity.
                    # Generate a summary for the class using an LLM.
                    summary = self.llm_ops.summarization_chain.invoke({"text_chunk": cluster_info["cluster_text"]}).content
                    summaries_to_embed.append(summary)
                    class_eids_to_embed.append(class_eid)
                    class_entity = {
                        "id": class_eid,
                        "type": "Class",
                        "properties": generated_properties,
                    }
                    name_to_class_entity[class_name] = class_entity
                    for member_id in cluster_member_ids:
                        class_id_map[member_id] = class_eid
                    logging.info(f"Created new class '{class_name}' (ID: {class_eid})")

            except Exception as e:
                logging.error(f"Failed to process generated properties for cluster {cluster_info}: {e}", exc_info=True)

        all_embeddings = self.llm_ops.get_embeddings(summaries_to_embed, class_eids_to_embed)
        for class_name, class_entity in name_to_class_entity.items():
            class_eid = class_entity["id"]
            embeddings = all_embeddings.get(class_eid, {})
            class_entity["clustering_embedding"] = embeddings.get("clustering", [0.0] * Config.EMBEDDING_DIMENSION)
            class_entity["retrieval_document_embedding"] = embeddings.get("semantic_search", [0.0] * Config.EMBEDDING_DIMENSION)

        # Collect all entities for the new graph: all original entities, plus new class entities
        new_entities = entities.copy()
        new_entities.extend(name_to_class_entity.values())

        logging.info(f"class_id_map: {json.dumps(class_id_map, indent=2)}")
        logging.info(f"new_entities IDs: {[e['id'] for e in new_entities]}")

        new_graph = ig.Graph(directed=True) # Create a new graph to build the result
        new_graph.add_vertices(len(new_entities))
        new_graph.vs["name"] = [e["id"] for e in new_entities]
        new_graph.vs["type"] = [e["type"] for e in new_entities]
        for i, entity in enumerate(new_entities):
            for k, v in entity["properties"].items():
                new_graph.vs[i][k] = v
            if "embedding" in entity:
                new_graph.vs[i]["embedding"] = entity["embedding"]
            if "clustering_embedding" in entity:
                new_graph.vs[i]["clustering_embedding"] = entity["clustering_embedding"]
            if "retrieval_document_embedding" in entity:
                new_graph.vs[i]["retrieval_document_embedding"] = entity["retrieval_document_embedding"]

        logging.info(f"new_graph vertex names: {new_graph.vs['name']}")
        name_to_vertex = {v["name"]: v for v in new_graph.vs}

        # Remap relationships to the new class entities
        for rel in relationships:
            if rel["source"] in class_id_map:
                rel["source"] = class_id_map[rel["source"]]
            if rel["target"] in class_id_map:
                rel["target"] = class_id_map[rel["target"]]

        # Add relationships to the new graph
        for rel in relationships:
            try:
                source_vertex = new_graph.vs.find(name=rel["source"])
                target_vertex = new_graph.vs.find(name=rel["target"])
                if source_vertex and target_vertex:
                    edge = new_graph.add_edge(source_vertex, target_vertex)
                    edge["type"] = rel["type"]
                    for k, v in rel["properties"].items():
                        edge[k] = v
            except ValueError:
                logging.warning(f"Skipping relationship due to missing source or target vertex: {rel}")

        # Update existing entities: change their type to 'Instance' if they are not 'Chunk' or 'Community'.
        # This reflects their new role as instances of the newly created 'Class' entities.
        for v in new_graph.vs:
            if v["type"] not in ["Chunk", "Community"]:
                v["type"] = "Instance"
            
            # Create 'INSTANCE_OF' relationships linking instances to their respective classes.
            class_id = class_id_map.get(v["name"])
            if class_id:
                source_vertex = name_to_vertex.get(v["name"])
                target_vertex = name_to_vertex.get(class_id)
                if source_vertex and target_vertex:
                    edge = new_graph.add_edge(source_vertex, target_vertex)
                    edge["type"] = "INSTANCE_OF"
                    edge["description"] = "Indicates that an entity is an instance of a specific class."

        # Propagate relationships from instances to their corresponding classes.
        # This creates higher-level relationships between classes based on the relationships between their instances.
        for rel in relationships:
            source_class_id = class_id_map.get(rel.get("source"))
            target_class_id = class_id_map.get(rel.get("target"))
            if source_class_id and target_class_id and source_class_id != target_class_id:
                source_vertex = name_to_vertex.get(source_class_id)
                target_vertex = name_to_vertex.get(target_class_id)
                if source_vertex and target_vertex:
                    edge = new_graph.add_edge(source_vertex, target_vertex)
                    edge["type"] = rel.get("type")
                    for k, v in rel.get("properties", {}).items():
                        edge[k] = v

        logging.info(f"Clustering complete. Result: {new_graph.vcount()} entities, {new_graph.ecount()} relationships.")
        return new_graph

    def deduplicate_entities(self, data):
        """
        Finds and resolves duplicate Entity IDs (EIDs) before community detection.
        This function handles two main cases for duplicates:
        1. Duplicate 'Class' entities: Merges them by selecting a winner based on instance count
           and remapping relationships to the winning class.
        2. Duplicate 'Instance' or unhandled type entities: Renames duplicates by appending a unique suffix
           to their EIDs and updates related relationships.
        This ensures that each entity has a unique identifier, which is crucial for graph integrity.

        Args:
            data (dict): A dictionary containing 'entities' and 'relationships'.

        Returns:
            dict: The updated data dictionary with duplicate entities resolved and relationships remapped.
        """
        logging.info("Starting entity de-duplication process...")
        entities = data.get("entities", [])
        relationships = data.get("relationships", [])
        id_to_entity = {e["id"]: e for e in entities}

        # Group entities by their EID to identify duplicates.
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

            # Handle duplicate 'Class' entities.
            # The class with the most instances is chosen as the winner, and others are merged into it.
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

            # Handle duplicate 'Instance' entities by renaming them with a unique suffix.
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
            # Handle other unhandled duplicate EID cases by renaming them.
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
        logging.info(f"Relationships before remapping: {relationships}")

        # Remap relationships to reflect the changes in EIDs due to deduplication.
        for rel in relationships:
            # Ensure rel is a dictionary and has 'source' and 'target' keys before processing
            if isinstance(rel, dict) and "source" in rel and "target" in rel:
                try:
                    if rel["source"] in eids_to_remap:
                        rel["source"] = eids_to_remap[rel["source"]]
                    if rel["target"] in eids_to_remap:
                        rel["target"] = eids_to_remap[rel["target"]]
                except KeyError as e:
                    logging.error(f"KeyError processing relationship {rel}: {e}", exc_info=True)
                    continue # Skip this malformed relationship
            else:
                logging.warning(f"Skipping malformed relationship: {rel}. Missing 'source' or 'target' or not a dictionary.")

        logging.info(f"Relationships after remapping: {relationships}")

        data["entities"] = list(final_entities.values())
        data["relationships"] = relationships
        logging.info(f"De-duplication complete. Result: {len(data['entities'])} entities.")
        return data

    def run_igraph_community_detection(self, graph: ig.Graph):
        """
        Performs community detection on the graph using igraph's maximal cliques algorithm.
        It identifies densely connected groups of entities (cliques) and creates new 'Community'
        entities for each detected community. Entities are then associated with their respective
        communities.

        Args:
            graph (igraph.Graph): The input graph containing entities and relationships.

        Returns:
            igraph.Graph: The updated graph with new 'Community' entities added and
                          original entities updated with their community affiliations.
        """
        logging.info("Running igraph community detection...")

        if not graph.vcount():
            return graph

        # Find maximal cliques, which represent the communities.
        cliques = graph.maximal_cliques()
        
        community_summaries = {} # Initialize a dictionary to store summaries for each community. 
        
        # Add a 'communities' attribute to all vertices if it doesn't exist
        if "communities" not in graph.vs.attributes():
            graph.vs["communities"] = [[] for _ in range(graph.vcount())]

        for i, v in enumerate(graph.vs):
            entity_id = v["name"]
            
            # Prepare a summary for the entity to be used in community summaries.
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

            # Assign entities to communities (cliques) they belong to.
            for j, clique in enumerate(cliques):
                if i in clique:
                    community_id = f"clique_{j}" # Generate a unique ID for the community.
                    v["communities"].append(community_id) # Associate the entity with this community. 
                    
                    # Aggregate entity summaries for each community.
                    if community_id not in community_summaries:
                        community_summaries[community_id] = []
                    community_summaries[community_id].append(entity_text_for_summary)

        # new_community_entities = []
        # Create new 'Community' entities based on the detected communities.
        for comm_id, entity_texts in community_summaries.items():
            entities_description = " ".join(entity_texts)

            # Generate summary using LLM
            full_community_summary = self.summarization_chain.invoke({"text_chunk": entities_description})['text']

            if not full_community_summary:
                logging.warning(f"Skipping Community entity creation for {comm_id} due to empty summary.")
                continue

            # Generate embeddings for the new community entity.
            all_embeddings = self.llm_ops.get_embeddings([full_community_summary], [comm_id])
            if not all_embeddings:
                logging.warning(f"Skipping Community entity creation for {comm_id} due to missing embeddings.")
                continue

            # Get the embeddings for the current community
            community_embeddings = all_embeddings.get(comm_id, {})
            semantic_search_embedding = community_embeddings.get("semantic_search", [0.0] * Config.EMBEDDING_DIMENSION)
            clustering_embedding = community_embeddings.get("clustering", [0.0] * Config.EMBEDDING_DIMENSION)
            
            community_entity_properties = {
                "community_type": "structural",
                "Summary": full_community_summary
            }
            
            # Add the new community as a vertex to the graph
            community_vertex = graph.add_vertex(name=comm_id)
            community_vertex["type"] = "Community"
            for k, v in community_entity_properties.items():
                community_vertex[k] = v
            community_vertex["cluster_embedding"] = clustering_embedding
            community_vertex["embedding"] = semantic_search_embedding
            community_vertex["communities"] = [] # Communities of a community entity are not relevant in this context.

        logging.info(f"Found {len(cliques)} cliques (overlapping communities) and created {len(community_summaries)} standard Community entities.")
        for v in graph.vs:
            if v["type"] == "Community":
                logging.info(f"Community entity: {v.attributes()}")
        return graph

    def remove_entities_with_null_keys_and_relationships(self, graph: ig.Graph):
        """
        Removes entities that have null or empty IDs and any relationships connected to them.
        This is a data cleaning step to ensure graph integrity before persistence.

        Args:
            graph (igraph.Graph): The input graph.

        Returns:
            igraph.Graph: The updated graph with invalid entities and their relationships removed.
        """
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

def generate_class_eid(name):
    """
    Generates a consistent and unique Entity ID (EID) for a class based on its name.
    This ensures that classes with the same name always have the same EID, facilitating merging.

    Args:
        name (str): The name of the class.

    Returns:
        str: A SHA256 hash of the class name, serving as its EID, or None if the name is empty.
    """
    if not name:
        return None
    # Use SHA256 hash to create a consistent and unique ID
    return hashlib.sha256(name.encode('utf-8')).hexdigest()

def _graph_to_dict(graph: ig.Graph) -> dict:
    """
    Converts an igraph.Graph object into a dictionary format with 'entities' and 'relationships' keys.
    This format is expected by the deduplicate_entities method.
    """
    entities = []
    for v in graph.vs:
        entity_properties = {k: v[k] for k in v.attributes() if k not in ["name", "type", "embedding", "cluster_embedding", "retrieval_document_embedding"]}
        entity = {
            "id": v["name"],
            "type": v["type"] if "type" in v.attributes() else None,
            "properties": entity_properties
        }
        if "embedding" in v.attributes():
            entity["embedding"] = v["embedding"]
        if "cluster_embedding" in v.attributes():
            entity["cluster_embedding"] = v["cluster_embedding"]
        if "retrieval_document_embedding" in v.attributes():
            entity["retrieval_document_embedding"] = v["retrieval_document_embedding"]
        entities.append(entity)

    relationships = []
    for e in graph.es:
        rel_properties = {k: e[k] for k in e.attributes() if k not in ["type"]}
        rel = {
            "source": graph.vs[e.source]["name"],
            "target": graph.vs[e.target]["name"],
            "type": e["type"] if "type" in e.attributes() else None,
            "properties": rel_properties
        }
        relationships.append(rel)

    return {"entities": entities, "relationships": relationships}

def _dict_to_graph(data: dict) -> ig.Graph:
    """
    Converts a dictionary with 'entities' and 'relationships' keys back into an igraph.Graph object.
    """
    graph = ig.Graph(directed=True)
    
    # Add vertices
    for entity in data.get("entities", []):
        vertex_attrs = {
            "name": entity["id"],
            "type": entity.get("type"),
            "properties": json.dumps(entity.get("properties", {})), # Store properties as JSON string
        }
        if "embedding" in entity:
            vertex_attrs["embedding"] = entity["embedding"]
        if "cluster_embedding" in entity:
            vertex_attrs["cluster_embedding"] = entity["cluster_embedding"]
        if "retrieval_document_embedding" in entity:
            vertex_attrs["retrieval_document_embedding"] = entity["retrieval_document_embedding"]
        graph.add_vertex(**vertex_attrs)

    # Add edges
    for rel in data.get("relationships", []):
        try:
            source_vertex = graph.vs.find(name=rel["source"])
            target_vertex = graph.vs.find(name=rel["target"])
            edge_attrs = {
                "type": rel.get("type"),
                "properties": json.dumps(rel.get("properties", {})), # Store properties as JSON string
            }
            graph.add_edge(source_vertex, target_vertex, **edge_attrs)
        except ValueError as e:
            logging.warning(f"Could not add edge for relationship {rel}: {e}")

    return graph