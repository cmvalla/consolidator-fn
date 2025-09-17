# Graph processing for the consolidator function
import logging
import igraph as ig
import numpy as np
import uuid
import json
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import Config
from llm_operations import LLMOperations
from llm_operations import CLASS_PROPERTY_GENERATION_PROMPT, SUMMARY_PROMPT, CLASS_SCHEMA
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

    def aggregate_results(self, data):
        """
        Aggregates partial graph data (entities and relationships) received from Redis.
        This function combines all extracted entities and relationships from multiple
        partial results into a single, unified set.

        Args:
            data (dict): A dictionary containing 'partial_results', where each result
                         is a JSON string representing extracted graph data.

        Returns:
            dict: A dictionary containing the aggregated 'entities' and 'relationships',
                  along with the original 'batch_id'.
        """
        all_entities = {}
        all_relationships = []
        for res_str in data["partial_results"]:
            logging.debug(f"Processing res_str: {res_str}")
            res_json = json.loads(res_str)
            logging.debug(f"Decoded res_json: {res_json}")
            
            extracted_entities = res_json.get("extracted_graph_data", {}).get("entities", [])
            extracted_relationships = res_json.get("extracted_graph_data", {}).get("relationships", [])
            
            logging.debug(f"Extracted entities from res_json: {extracted_entities}")
            logging.debug(f"Extracted relationships from res_json: {extracted_relationships}")

            for entity in extracted_entities:
                entity_id = entity.get("id")
                # Ensure entities have a valid ID before adding them to the aggregated set.
                # This prevents issues with entities that might be malformed or incomplete.
                if entity_id:
                    all_entities[entity_id] = entity
                else:
                    logging.warning(f"Skipping entity without id: {entity}")
            # Process relationships to ensure 'source' and 'target' keys are present.
            # The LLM might return 'id_1' and 'id_2', which are renamed to 'source' and 'target'
            # for consistency with graph processing conventions.
            for rel in extracted_relationships:
                if "id_1" in rel and "id_2" in rel:
                    rel["source"] = rel.pop("id_1") # Rename id_1 to source
                    rel["target"] = rel.pop("id_2") # Rename id_2 to target
                    all_relationships.append(rel)
                elif "source" in rel and "target" in rel:
                    all_relationships.append(rel)
                else:
                    logging.warning(f"Skipping malformed relationship from LLM: {rel}. Missing 'id_1'/'id_2' or 'source'/'target'.")
        
        logging.info(f"Aggregated {len(all_entities)} entities and {len(all_relationships)} relationships.")

        return {
            "batch_id": data["batch_id"],
            "entities": list(all_entities.values()),
            "relationships": all_relationships
        }

    def cluster_and_merge_entities(self, data, similarity_threshold=0.9):
        """
        Clusters similar entities based on their embeddings, creates 'Class' nodes for these clusters,
        and merges classes with the same name by re-linking their instances.
        This process aims to reduce redundancy and create a more abstract representation of entities.

        Args:
            data (dict): A dictionary containing 'entities' (with 'cluster_embedding') and 'relationships'.
            similarity_threshold (float): The cosine similarity threshold for grouping entities into clusters.

        Returns:
            dict: The updated data dictionary with new 'Class' entities and 'INSTANCE_OF' relationships.
        """
        entities = data.get("entities", [])
        relationships = data.get("relationships", [])

        if not entities:
            return data

        id_to_entity = {entity['id']: entity for entity in entities}

        # Map entity IDs to their source text (from chunks) for LLM context during class property generation.
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
            return data
            
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

        # Initialize LLM chains for generating class properties and summaries.
        class_property_chain = LLMChain(llm=self.llm_ops.llm, prompt=CLASS_PROPERTY_GENERATION_PROMPT)
        summarization_chain = LLMChain(llm=self.llm_ops.llm, prompt=SUMMARY_PROMPT)

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
                    source_text = entity_id_to_source_text.get(member_id)
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
            batch_cluster_info = cluster_info_list[i:i + Config.LLM_BATCH_SIZE]

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

        # Iterate through the generated properties and create or merge 'Class' entities.
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
                    # Generate embeddings for the new class entity.
                    all_embeddings = self.llm_ops._get_single_embedding(summary, class_eid)
                    class_entity = {
                        "id": class_eid,
                        "type": "Class",
                        "properties": generated_properties,
                        "clustering_embedding": all_embeddings.get("clustering", [0.0] * Config.EMBEDDING_DIMENSION),
                        "retrieval_document_embedding": all_embeddings.get("semantic_search", [0.0] * Config.EMBEDDING_DIMENSION)
                    }
                    name_to_class_entity[class_name] = class_entity
                    for member_id in cluster_member_ids:
                        class_id_map[member_id] = class_eid
                    logging.info(f"Created new class '{class_name}' (ID: {class_eid})")

            except Exception as e:
                logging.error(f"Failed to process generated properties for cluster {cluster_info}: {e}", exc_info=True)


        new_entities = list(name_to_class_entity.values())
        new_relationships = []

        # Update existing entities: change their type to 'Instance' if they are not 'Chunk' or 'Community'.
        # This reflects their new role as instances of the newly created 'Class' entities.
        for entity in entities:
            if entity.get("type") not in ["Chunk", "Community"]:
                entity["type"] = "Instance"
            new_entities.append(entity)
            
            # Create 'INSTANCE_OF' relationships linking instances to their respective classes.
            class_id = class_id_map.get(entity["id"])
            if class_id:
                new_relationships.append({
                    "source": entity["id"],
                    "target": class_id,
                    "type": "INSTANCE_OF",
                    "properties": {"description": "Indicates that an entity is an instance of a specific class."}
                })

        # Propagate relationships from instances to their corresponding classes.
        # This creates higher-level relationships between classes based on the relationships between their instances.
        for rel in relationships:
            source_class_id = class_id_map.get(rel.get("source"))
            target_class_id = class_id_map.get(rel.get("target")),
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
                    logging.info(f"Merging class '{loser['id']}' into '{winner['id']}'.")

            # Handle duplicate 'Instance' entities by renaming them with a unique suffix.
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
                        logging.info(f"Renamed duplicate entity '{original_id}' of type '{duplicate.get('type')}' to '{new_eid}'.")

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

        data["entities"] = list(final_entities.values())
        logging.info(f"De-duplication complete. Result: {len(data['entities'])} entities.")
        return data

    def run_igraph_community_detection(self, data):
        """
        Performs community detection on the graph using igraph's maximal cliques algorithm.
        It identifies densely connected groups of entities (cliques) and creates new 'Community'
        entities for each detected community. Entities are then associated with their respective
        communities.

        Args:
            data (dict): A dictionary containing 'entities' and 'relationships'.

        Returns:
            dict: The updated data dictionary with new 'Community' entities added and
                  original entities updated with their community affiliations.
        """
        logging.info("Running igraph community detection...")
        entities = data.get("entities", [])
        relationships = data.get("relationships", [])

        # Create a mapping from entity ID to igraph vertex index, ensuring entities have IDs.
        valid_entities = [e for e in entities if e.get("id") is not None]
        id_to_vertex = {entity["id"]: i for i, entity in enumerate(valid_entities)}
        
        # Initialize an igraph graph and add vertices based on the valid entities.
        g = ig.Graph(directed=False)
        g.add_vertices(len(valid_entities))
        
        # Assign entity properties to igraph vertex attributes.
        g.vs["id"] = [entity["id"] for entity in valid_entities]
        g.vs["type"] = [entity.get("type") for entity in valid_entities]
        g.vs["properties"] = [entity.get("properties", {}) for entity in valid_entities]
        g.vs["embedding"] = [entity.get("clustering_embedding") for entity in valid_entities]

        # Add edges to the igraph graph based on the relationships.
        edges = []
        for rel in relationships:
            source_id = rel.get("source")
            target_id = rel.get("target")
            if source_id in id_to_vertex and target_id in id_to_vertex:
                edges.append((id_to_vertex[source_id], id_to_vertex[target_id]))
        g.add_edges(edges)

        # Find maximal cliques, which represent the communities.
        cliques = g.maximal_cliques()
        
        community_summaries = {} # Initialize a dictionary to store summaries for each community.
        for entity in entities:
            entity_id = entity.get("id")
            if not entity_id:
                logging.warning(f"Skipping entity in community detection due to missing id: {entity}")
                continue
            entity["communities"] = [] # Initialize an empty list to store community IDs for each entity.
            
            # Prepare a summary for the entity to be used in community summaries.
            entity_summary_parts = []
            if entity.get("type"):
                entity_summary_parts.append(f"Type: {entity.get('type')}")
            summary = entity.get("properties", {}).get("summary")
            name = entity.get("properties", {}).get("name")
            if summary:
                entity_summary_parts.append(f"Summary: {summary}")
            elif name:
                entity_summary_parts.append(f"Name: {name}")
            
            entity_text_for_summary = ", ".join(entity_summary_parts) if entity_summary_parts else entity_id

            # Assign entities to communities (cliques) they belong to.
            for i, clique in enumerate(cliques):
                if id_to_vertex.get(entity_id) is not None and id_to_vertex[entity_id] in clique:
                    community_id = f"clique_{i}" # Generate a unique ID for the community.
                    entity["communities"].append(community_id) # Associate the entity with this community.
                    
                    # Aggregate entity summaries for each community.
                    if community_id not in community_summaries:
                        community_summaries[community_id] = []
                    community_summaries[community_id].append(entity_text_for_summary)

        new_community_entities = []
        # Create new 'Community' entities based on the detected communities.
        for comm_id, entity_texts in community_summaries.items():
            entities_description = " ".join(entity_texts)

            # Generate summary using LLM
            full_community_summary = self.summarization_chain.invoke({"text_chunk": entities_description})['text']

            if not full_community_summary:
                logging.warning(f"Skipping Community entity creation for {comm_id} due to empty summary.")
                continue

            # Generate embeddings for the new community entity.
            all_embeddings = self.llm_ops._get_single_embedding(full_community_summary, comm_id)
            if not all_embeddings:
                logging.warning(f"Skipping Community entity creation for {comm_id} due to missing embeddings.")
                continue

            semantic_search_embedding = all_embeddings.get("semantic_search", [0.0] * Config.EMBEDDING_DIMENSION)
            
            community_entity = {
                "id": comm_id,
                "type": "Community",
                "Summary": full_community_summary,
                "properties": {
                    "community_type": "structural"
                },
                "cluster_embedding": all_embeddings.get("clustering", [0.0] * Config.EMBEDDING_DIMENSION),
                "embedding": semantic_search_embedding, # Use semantic search embedding for the main embedding field
                "communities": [] # Communities of a community entity are not relevant in this context.
            }
            new_community_entities.append(community_entity)

        # Extend the main entities list with the newly created 'Community' entities.
        logging.info(f"New Community Entities created: {[e['id'] for e in new_community_entities]}")
        entities.extend(new_community_entities)
        logging.info(f"Entities after extending with new Community Entities: {[e['id'] for e in entities if e.get('type') == 'Community']}")

        logging.info(f"Found {len(cliques)} cliques (overlapping communities) and created {len(new_community_entities)} standard Community entities.")
        return data

    def remove_entities_with_null_keys_and_relationships(self, data):
        """
        Removes entities that have null or empty IDs and any relationships connected to them.
        This is a data cleaning step to ensure graph integrity before persistence.

        Args:
            data (dict): A dictionary containing 'entities' and 'relationships'.

        Returns:
            dict: The updated data dictionary with invalid entities and their relationships removed.
        """
        logging.info("Starting removal of entities with null/empty IDs and their relationships...")
        entities = data.get("entities", [])
        relationships = data.get("relationships", [])

        logging.info(f"Entities before removal: {[e['id'] for e in entities if e.get('type') == 'Community']}")

        entities_to_remove_ids = set()
        cleaned_entities = []

        # Identify entities with null or empty IDs.
        for entity in entities:
            entity_id = entity.get("id")
            if entity_id is None or str(entity_id).strip() == "":
                entities_to_remove_ids.add(entity_id)
                logging.warning(f"Removing entity with null/empty ID: {entity}")
            else:
                cleaned_entities.append(entity)
        
        if not entities_to_remove_ids:
            logging.info("No entities with null/empty IDs found. Skipping relationship filtering.")
            logging.info(f"Entities after removal (no changes): {[e['id'] for e in cleaned_entities if e.get('type') == 'Community']}")
            return data

        # Remove relationships connected to the identified invalid entities.
        cleaned_relationships = []
        for rel in relationships:
            source_id = rel.get("source")
            target_id = rel.get("target")
            if source_id in entities_to_remove_ids or target_id in entities_to_remove_ids:
                logging.warning(f"Removing relationship connected to null/empty ID entity: {rel}")
            else:
                cleaned_relationships.append(rel)

        data["entities"] = cleaned_entities
        data["relationships"] = cleaned_relationships
        logging.info(f"Finished removal. {len(entities) - len(cleaned_entities)} entities and {len(relationships) - len(cleaned_relationships)} relationships removed.")
        logging.info(f"Entities after removal: {[e['id'] for e in cleaned_entities if e.get('type') == 'Community']}")
        return data

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