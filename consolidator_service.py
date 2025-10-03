import logging
import os
import json
import datetime
import pickle
import igraph as ig
from google.cloud import pubsub_v1
from google.cloud import storage

import psutil

from llm_operations import LLMOperations
from graph_processing import GraphProcessor, _graph_to_dict, _dict_to_graph
from spanner_operations import SpannerOperations

class ConsolidatorService:
    def __init__(self, llm_ops: LLMOperations, graph_processor: GraphProcessor, 
                 spanner_ops: SpannerOperations, publisher: pubsub_v1.PublisherClient, 
                 storage_client: storage.Client, invocation_id: str):
        self.llm_ops = llm_ops
        self.graph_processor = graph_processor
        self.spanner_ops = spanner_ops
        self.publisher = publisher
        self.storage_client = storage_client
        self.invocation_id = invocation_id
        logging.info(f"ConsolidatorService initialized with Invocation ID: {self.invocation_id}.")

    def _log_memory_usage(self, stage: str):
        memory_info = psutil.virtual_memory()
        logging.info(f"Memory usage at stage '{stage}': Total={memory_info.total / 1024**3:.2f}GB, Available={memory_info.available / 1024**3:.2f}GB, Used={memory_info.used / 1024**3:.2f}GB, Percentage={memory_info.percent}%")

    def process_message(self, data):
        batch_id = data.get("batch_id")
        gcs_paths = data.get("gcs_paths")
        instance_id = os.environ.get("GAE_INSTANCE")
        persistor_topic_name = os.environ.get("PERSISTOR_TOPIC_NAME")
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        persistor_topic_path = self.publisher.topic_path(project_id, persistor_topic_name)


        if not gcs_paths:
            logging.error(f"No GCS paths found in Pub/Sub message for batch {batch_id}. Stopping execution.")
            if batch_id:
                self.spanner_ops.release_lock(batch_id, "FAILED")
            return None

        # Macro Phase 1: Acquiring Lock
        start_time_phase1 = datetime.datetime.now()
        logging.info(f"Macro Phase 1: Acquiring lock for batch {batch_id}. Start Time: {start_time_phase1}.")
        if not self.spanner_ops.acquire_lock(batch_id, instance_id):
            return None
        end_time_phase1 = datetime.datetime.now()
        logging.info(f"Macro Phase 1: Lock acquired for batch {batch_id}. End Time: {end_time_phase1}. Duration: {end_time_phase1 - start_time_phase1}.")

        try:
            # Macro Phase 2: Downloading and Merging Graphs
            start_time_phase2 = datetime.datetime.now()
            logging.info(f"Macro Phase 2: Downloading and merging graphs for batch {batch_id}. Start Time: {start_time_phase2}.")
            graphs_to_merge = []
            for path in gcs_paths:
                bucket_name, blob_name = path.replace("gs://", "").split("/", 1)
                bucket = self.storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                downloaded_blob = blob.download_as_bytes()
                worker_graph = pickle.loads(downloaded_blob)
                
                num_entities_worker = worker_graph.vcount()
                num_relationships_worker = worker_graph.ecount()
                
                num_clustering_embeddings_worker = 0
                num_semantic_embeddings_worker = 0
                for vertex in worker_graph.vs:
                    if "cluster_embedding" in vertex.attributes() and vertex["cluster_embedding"] is not None and any(e != 0.0 for e in vertex["cluster_embedding"]):
                        num_clustering_embeddings_worker += 1
                    if "embedding" in vertex.attributes() and vertex["embedding"] is not None and any(e != 0.0 for e in vertex["embedding"]):
                        num_semantic_embeddings_worker += 1
                
                logging.info(f"Deserialized graph from {path}: Entities={num_entities_worker}, Relationships={num_relationships_worker}, ClusteringEmbeddings={num_clustering_embeddings_worker}, SemanticEmbeddings={num_semantic_embeddings_worker}.")
                
                graphs_to_merge.append(worker_graph)
            
            if not graphs_to_merge:
                logging.info(f"No graphs to merge for batch {batch_id}. Stopping execution.")
                self.spanner_ops.release_lock(batch_id, "FAILED")
                return None

            self._log_memory_usage("After deserializing graphs")

            merged_graph = ig.union(graphs_to_merge, byname=True)

            # Manually combine vertex attributes after union
            for v in merged_graph.vs:
                for attr_prefix in ['embedding', 'cluster_embedding', 'retrieval_document_embedding']:
                    # If the base attribute already exists, we are good
                    if attr_prefix in v.attributes() and v[attr_prefix] is not None:
                        continue

                    # Otherwise, find the first available embedding from the suffixed attributes
                    for i in range(1, len(graphs_to_merge) + 1):
                        suffixed_attr = f'{attr_prefix}_{i}'
                        if suffixed_attr in v.attributes() and v[suffixed_attr] is not None:
                            v[attr_prefix] = v[suffixed_attr]
                            break  # Found one, no need to check further
                    
                    # Clean up all suffixed attributes
                    for i in range(1, len(graphs_to_merge) + 1):
                        suffixed_attr = f'{attr_prefix}_{i}'
                        if suffixed_attr in v.attributes():
                            del v[suffixed_attr]
            
            end_time_phase2 = datetime.datetime.now()
            logging.info(f"Macro Phase 2: Merged {len(graphs_to_merge)} graphs. Resulting graph has {merged_graph.vcount()} vertices and {merged_graph.ecount()} edges. End Time: {end_time_phase2}. Duration: {end_time_phase2 - start_time_phase2}.")
            self._log_memory_usage("After merging graphs")
            
            if not merged_graph.vcount():
                logging.info(f"No graph data found after merging for batch {batch_id}. Stopping execution.")
                self.spanner_ops.release_lock(batch_id, "FAILED")
                return None

            embedded_graph = merged_graph            
            
            # Macro Phase 3: Clustering and Deduplication
            start_time_phase3 = datetime.datetime.now()
            logging.info(f"Macro Phase 3: Clustering and deduplicating entities for batch {batch_id}. Start Time: {start_time_phase3}.")
            self._log_memory_usage("Before clustering")
            
            logging.info(f"Before clustering, graph has {embedded_graph.vcount()} vertices and {embedded_graph.ecount()} edges.")
            clustered_graph = self.graph_processor.cluster_and_merge_entities(embedded_graph)
            logging.info(f"After clustering, graph has {clustered_graph.vcount()} vertices and {clustered_graph.ecount()} edges.")
            self._log_memory_usage("After clustering")

            # Convert igraph.Graph to dictionary format for deduplication
            logging.info("Before converting to dict, graph has %d vertices and %d edges.", clustered_graph.vcount(), clustered_graph.ecount())
            self._log_memory_usage("Before _graph_to_dict")
            clustered_graph_dict = _graph_to_dict(clustered_graph)
            self._log_memory_usage("After _graph_to_dict")
            logging.info("After converting to dict, dict has %d entities and %d relationships.", len(clustered_graph_dict.get('entities', [])), len(clustered_graph_dict.get('relationships', [])))
            
            self._log_memory_usage("Before deduplication")
            deduplicated_graph_dict = self.graph_processor.deduplicate_entities(clustered_graph_dict)
            self._log_memory_usage("After deduplication")
            end_time_phase3 = datetime.datetime.now()
            logging.info(f"Macro Phase 3: Clustered graph had {clustered_graph.vcount()} entities. Deduplicated graph has {len(deduplicated_graph_dict.get('entities', []))} entities. End Time: {end_time_phase3}. Duration: {end_time_phase3 - start_time_phase3}.")
            
            # Convert back to igraph.Graph for community detection
            logging.info("Before converting back to graph, dict has %d entities and %d relationships.", len(deduplicated_graph_dict.get('entities', [])), len(deduplicated_graph_dict.get('relationships', [])))
            self._log_memory_usage("Before _dict_to_graph")
            deduplicated_graph = _dict_to_graph(deduplicated_graph_dict)
            self._log_memory_usage("After _dict_to_graph")
            logging.info("After converting back to graph, graph has %d vertices and %d edges.", deduplicated_graph.vcount(), deduplicated_graph.ecount())
            
            # Macro Phase 4: Community Detection
            start_time_phase4 = datetime.datetime.now()
            logging.info(f"Macro Phase 4: Running community detection for batch {batch_id}. Start Time: {start_time_phase4}.")
            self._log_memory_usage("Before community detection")
            logging.info(f"Before community detection, graph has {deduplicated_graph.vcount()} vertices and {deduplicated_graph.ecount()} edges.")
            community_graph = self.graph_processor.run_igraph_community_detection(deduplicated_graph)
            logging.info(f"After community detection, graph has {community_graph.vcount()} vertices and {community_graph.ecount()} edges.")
            self._log_memory_usage("After community detection")
            end_time_phase4 = datetime.datetime.now()
            logging.info(f"Macro Phase 4: Community detection found {community_graph.vcount()} communities. End Time: {end_time_phase4}. Duration: {end_time_phase4 - start_time_phase4}.")
            
            # Macro Phase 5: Removing Null Entities/Relationships
            start_time_phase5 = datetime.datetime.now()
            logging.info(f"Macro Phase 5: Removing entities with null keys and relationships for batch {batch_id}. Start Time: {start_time_phase5}.")
            self._log_memory_usage("Before removing nulls")
            logging.info(f"Before removing nulls, graph has {community_graph.vcount()} vertices and {community_graph.ecount()} edges.")
            final_graph = self.graph_processor.remove_entities_with_null_keys_and_relationships(community_graph)
            if final_graph:
                logging.info(f"Summary Phase 5: Final graph has {final_graph.vcount()} entities and {final_graph.ecount()} relationships after removing nulls.")
            self._log_memory_usage("After removing nulls")
            end_time_phase5 = datetime.datetime.now()
            logging.info(f"Macro Phase 5: Null entities/relationships removed for batch {batch_id}. End Time: {end_time_phase5}. Duration: {end_time_phase5 - start_time_phase5}.")
            
            if final_graph:
                # Macro Phase 6: Serializing and Uploading to GCS
                start_time_phase6 = datetime.datetime.now()
                logging.info(f"Macro Phase 6: Serializing and uploading graph to GCS for batch {batch_id}. Start Time: {start_time_phase6}.")
                try:
                    serialized_graph = pickle.dumps(final_graph)
                    
                    gcs_bucket_name = os.environ.get("GRAPH_DATA_BUCKET_NAME")
                    if not gcs_bucket_name:
                        logging.error("GRAPH_DATA_BUCKET_NAME environment variable not set.")
                        self.spanner_ops.release_lock(batch_id, "FAILED")
                        return None

                    bucket = self.storage_client.bucket(gcs_bucket_name)
                    
                    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
                    object_name = f"graph_data/{batch_id.replace('/', '_').replace(':', '_')}_{timestamp}.pkl"
                    blob = bucket.blob(object_name)
                    
                    blob.upload_from_string(serialized_graph)
                    gcs_path = f"gs://{gcs_bucket_name}/{object_name}"
                    logging.info(f"Uploaded serialized graph to GCS: {gcs_path}.")

                    # Macro Phase 7: Publishing to Persistor Topic
                    start_time_phase7 = datetime.datetime.now()
                    logging.info(f"Macro Phase 7: Publishing message to persistor topic for batch {batch_id}. Start Time: {start_time_phase7}.")
                    message_payload = {
                        "batch_id": batch_id,
                        "gcs_path": gcs_path
                    }
                    future = self.publisher.publish(persistor_topic_path, json.dumps(message_payload).encode("utf-8"))
                    future.result()
                    end_time_phase7 = datetime.datetime.now()
                    logging.info(f"Published message with GCS path for batch {batch_id} to topic {persistor_topic_path}. End Time: {end_time_phase7}. Duration: {end_time_phase7 - start_time_phase7}.")

                except Exception as e:
                    logging.error(f"Error serializing or uploading graph to GCS for batch {batch_id}: {e}.", exc_info=True)
                    self.spanner_ops.release_lock(batch_id, "FAILED")
                    return None
            else:
                logging.error(f"No community_data to serialize for batch {batch_id}. This is an error condition.")
                self.spanner_ops.release_lock(batch_id, "FAILED")
                return None

            # Macro Phase 8: Releasing Lock
            start_time_phase8 = datetime.datetime.now()
            logging.info(f"Macro Phase 8: Releasing lock for batch {batch_id} with status PENDING_PERSISTENCE. Start Time: {start_time_phase8}.")
            self.spanner_ops.release_lock(batch_id, "PENDING_PERSISTENCE")
            end_time_phase8 = datetime.datetime.now()
            logging.info(f"Macro Phase 8: Lock released for batch {batch_id}. End Time: {end_time_phase8}. Duration: {end_time_phase8 - start_time_phase8}.")

            # Final Summary Log
            num_entities = final_graph.vcount()
            num_relationships = final_graph.ecount()
            num_clustering_embeddings = 0
            num_semantic_embeddings = 0
            entities_without_semantic_embedding = []
            for vertex in final_graph.vs:
                if "cluster_embedding" in vertex.attributes() and vertex["cluster_embedding"] is not None and any(e != 0.0 for e in vertex["cluster_embedding"]):
                    num_clustering_embeddings += 1
                
                has_embedding = "embedding" in vertex.attributes() and vertex["embedding"] is not None and any(e != 0.0 for e in vertex["embedding"])
                if has_embedding:
                    num_semantic_embeddings += 1
                else:
                    entity_info = {
                        "id": vertex.attributes().get("name", "N/A"),
                        "description": vertex.attributes().get("description", "N/A")
                    }
                    entities_without_semantic_embedding.append(entity_info)

            if entities_without_semantic_embedding:
                logging.warning(f"Found {len(entities_without_semantic_embedding)} entities without a semantic embedding:")
                for entity in entities_without_semantic_embedding:
                    logging.warning(f"  - Entity ID: {entity['id']}, Description: {entity['description']}")
            
            logging.info(f"--- FINAL CONSOLIDATOR SUMMARY FOR BATCH {batch_id} ---")
            logging.info(f"Total Entities: {num_entities}")
            logging.info(f"Total Relationships: {num_relationships}")
            logging.info(f"Entities with Clustering Embedding: {num_clustering_embeddings}/{num_entities}")
            logging.info(f"Entities with Semantic Embedding: {num_semantic_embeddings}/{num_entities}")
            logging.info(f"-----------------------------------------------------")

        except Exception as e:
            logging.error(f'An error occurred while processing batch {batch_id}: {e}.', exc_info=True)
            self.spanner_ops.release_lock(batch_id, "FAILED")
            raise e
