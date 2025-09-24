import logging
import os
import json
import datetime
import pickle
import igraph as ig
from google.cloud import pubsub_v1
from google.cloud import storage

from llm_operations import LLMOperations
from graph_processing import GraphProcessor, _graph_to_dict, _dict_to_graph
from spanner_operations import SpannerOperations

class ConsolidatorService:
    def __init__(self, llm_ops: LLMOperations, graph_processor: GraphProcessor, 
                 spanner_ops: SpannerOperations, publisher: pubsub_v1.PublisherClient, 
                 storage_client: storage.Client):
        self.llm_ops = llm_ops
        self.graph_processor = graph_processor
        self.spanner_ops = spanner_ops
        self.publisher = publisher
        self.storage_client = storage_client
        logging.info("ConsolidatorService initialized.")

    def process_message(self, batch_id: str, gcs_paths: list, persistor_topic_path: str, instance_id: str):
        logging.info(f"Processing message for batch_id: {batch_id}")

        if not gcs_paths:
            logging.error(f"No GCS paths found in Pub/Sub message for batch {batch_id}. Stopping execution.")
            if batch_id:
                self.spanner_ops.release_lock(batch_id, "FAILED")
            return None

        # Macro Phase 1: Acquiring Lock
        logging.info(f"Macro Phase 1: Acquiring lock for batch {batch_id}.")
        if not self.spanner_ops.acquire_lock(batch_id, instance_id):
            return None

        try:
            # Macro Phase 2: Downloading and Merging Graphs
            logging.info(f"Macro Phase 2: Downloading and merging graphs for batch {batch_id}.")
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
                    if "cluster_embedding" in vertex.attributes() and any(e != 0.0 for e in vertex["cluster_embedding"]):
                        num_clustering_embeddings_worker += 1
                    if "embedding" in vertex.attributes() and any(e != 0.0 for e in vertex["embedding"]):
                        num_semantic_embeddings_worker += 1
                
                logging.info(f"Deserialized graph from {path}: Entities={num_entities_worker}, Relationships={num_relationships_worker}, ClusteringEmbeddings={num_clustering_embeddings_worker}, SemanticEmbeddings={num_semantic_embeddings_worker}")
                
                graphs_to_merge.append(worker_graph)
            
            if not graphs_to_merge:
                logging.info(f"No graphs to merge for batch {batch_id}. Stopping execution.")
                self.spanner_ops.release_lock(batch_id, "FAILED")
                return None

            merged_graph = ig.union(graphs_to_merge, byname=True)
            logging.info(f"Summary Phase 2: Merged {len(graphs_to_merge)} graphs. Resulting graph has {merged_graph.vcount()} vertices and {merged_graph.ecount()} edges.")
            
            if not merged_graph.vcount():
                logging.info(f"No graph data found after merging for batch {batch_id}. Stopping execution.")
                self.spanner_ops.release_lock(batch_id, "FAILED")
                return None

            embedded_graph = merged_graph            
            
            # Macro Phase 3: Clustering and Deduplication
            logging.info(f"Macro Phase 3: Clustering and deduplicating entities for batch {batch_id}.")
            clustered_graph = self.graph_processor.cluster_and_merge_entities(embedded_graph)
            
            # Convert igraph.Graph to dictionary format for deduplication
            clustered_graph_dict = _graph_to_dict(clustered_graph)
            
            deduplicated_graph_dict = self.graph_processor.deduplicate_entities(clustered_graph_dict)
            logging.info(f"Summary Phase 3: Clustered graph had {clustered_graph.vcount()} entities. Deduplicated graph has {len(deduplicated_graph_dict.get("entities", []))} entities.")
            
            # Convert back to igraph.Graph for community detection
            deduplicated_graph = _dict_to_graph(deduplicated_graph_dict)
            
            # Macro Phase 4: Community Detection
            logging.info(f"Macro Phase 4: Running community detection for batch {batch_id}.")
            community_graph = self.graph_processor.run_igraph_community_detection(deduplicated_graph)
            logging.info(f"Summary Phase 4: Community detection found {community_graph.vcount()} communities.")
            
            # Macro Phase 5: Removing Null Entities/Relationships
            logging.info(f"Macro Phase 5: Removing entities with null keys and relationships for batch {batch_id}.")
            final_graph = self.graph_processor.remove_entities_with_null_keys_and_relationships(community_graph)
            if final_graph:
                logging.info(f"Summary Phase 5: Final graph has {final_graph.vcount()} entities and {final_graph.ecount()} relationships after removing nulls.")
            
            if final_graph:
                # Macro Phase 6: Serializing and Uploading to GCS
                logging.info(f"Macro Phase 6: Serializing and uploading graph to GCS for batch {batch_id}.")
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
                    logging.info(f"Uploaded serialized graph to GCS: {gcs_path}")

                    # Macro Phase 7: Publishing to Persistor Topic
                    logging.info(f"Macro Phase 7: Publishing message to persistor topic for batch {batch_id}.")
                    message_payload = {
                        "batch_id": batch_id,
                        "gcs_path": gcs_path
                    }
                    future = self.publisher.publish(persistor_topic_path, json.dumps(message_payload).encode("utf-8"))
                    future.result()
                    logging.info(f"Published message with GCS path for batch {batch_id} to topic {persistor_topic_path}")

                except Exception as e:
                    logging.error(f"Error serializing or uploading graph to GCS for batch {batch_id}: {e}", exc_info=True)
                    self.spanner_ops.release_lock(batch_id, "FAILED")
                    return None
            else:
                logging.error(f"No community_data to serialize for batch {batch_id}. This is an error condition.")
                self.spanner_ops.release_lock(batch_id, "FAILED")
                return None

            # Macro Phase 8: Releasing Lock
            logging.info(f"Macro Phase 8: Releasing lock for batch {batch_id} with status PENDING_PERSISTENCE.")
            self.spanner_ops.release_lock(batch_id, "PENDING_PERSISTENCE")

        except Exception as e:
            logging.error(f'An error occurred while processing batch {batch_id}: {e}', exc_info=True)
            self.spanner_ops.release_lock(batch_id, "FAILED")
            raise e