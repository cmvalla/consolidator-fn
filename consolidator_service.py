import logging
import os
import json
import time
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
                 storage_client: storage.Client, invocation_id: str):
        self.llm_ops = llm_ops
        self.graph_processor = graph_processor
        self.spanner_ops = spanner_ops
        self.publisher = publisher
        self.storage_client = storage_client
        self.invocation_id = invocation_id
        logging.info(f"ConsolidatorService initialized with Invocation ID: {self.invocation_id}.")

    def _log_graph_stats(self, graph: ig.Graph, step_name: str):
        """Logs statistics of a graph at a given processing step."""
        if not graph:
            logging.info(f"Graph is None at step: '{step_name}'")
            return
        
        num_entities = graph.vcount()
        num_relationships = graph.ecount()
        
        num_semantic_embeddings = 0
        num_clustering_embeddings = 0
        num_retrieval_embeddings = 0
        num_relationship_embeddings = 0

        if num_entities > 0:
            vertex_attributes = graph.vs.attributes()
            has_semantic_attr = "embedding" in vertex_attributes
            has_clustering_attr = "cluster_embedding" in vertex_attributes
            has_retrieval_attr = "retrieval_document_embedding" in vertex_attributes

            for v in graph.vs:
                if has_semantic_attr and v["embedding"] is not None:
                    num_semantic_embeddings += 1
                if has_clustering_attr and v["cluster_embedding"] is not None:
                    num_clustering_embeddings += 1
                if has_retrieval_attr and v["retrieval_document_embedding"] is not None:
                    num_retrieval_embeddings += 1
        
        if num_relationships > 0:
            edge_attributes = graph.es.attributes()
            if "embedding" in edge_attributes:
                for e in graph.es:
                    if e["embedding"] is not None:
                        num_relationship_embeddings += 1

        logging.info(f"--- Stats after '{step_name}' ---")
        logging.info(f"Entities: {num_entities}")
        logging.info(f"Relationships: {num_relationships}")
        logging.info(f"Entities with Semantic Embedding: {num_semantic_embeddings}/{num_entities}")
        logging.info(f"Entities with Clustering Embedding: {num_clustering_embeddings}/{num_entities}")
        logging.info(f"Entities with Retrieval Embedding: {num_retrieval_embeddings}/{num_entities}")
        logging.info(f"Relationships with Embedding: {num_relationship_embeddings}/{num_relationships}")
        logging.info("-------------------------------------------")

    def process_message(self, data):
        total_start_time = time.time()
        timings = {}

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

        # Phase 1: Acquiring Lock
        start_time = time.time()
        logging.info(f"Acquiring lock for batch {batch_id}.")
        if not self.spanner_ops.acquire_lock(batch_id, instance_id):
            return None
        timings['acquire_lock'] = time.time() - start_time

        try:
            # Phase 2: Downloading and Merging Graphs
            start_time = time.time()
            logging.info(f"Downloading and merging graphs for batch {batch_id}.")
            graphs_to_merge = []
            for path in gcs_paths:
                bucket_name, blob_name = path.replace("gs://", "").split("/", 1)
                bucket = self.storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                downloaded_blob = blob.download_as_bytes()
                worker_graph = pickle.loads(downloaded_blob)
                self._log_graph_stats(worker_graph, f"Downloaded from {path}")
                graphs_to_merge.append(worker_graph)
            
            if not graphs_to_merge:
                logging.info(f"No graphs to merge for batch {batch_id}. Stopping execution.")
                self.spanner_ops.release_lock(batch_id, "FAILED")
                return None

            merged_graph = ig.union(graphs_to_merge, byname=True)

            # Manually combine vertex attributes
            for v in merged_graph.vs:
                for attr_prefix in ['embedding', 'cluster_embedding', 'retrieval_document_embedding']:
                    if attr_prefix in v.attributes() and v[attr_prefix] is not None:
                        continue
                    for i in range(1, len(graphs_to_merge) + 1):
                        suffixed_attr = f'{attr_prefix}_{i}'
                        if suffixed_attr in v.attributes() and v[suffixed_attr] is not None:
                            v[attr_prefix] = v[suffixed_attr]
                            break
                    for i in range(1, len(graphs_to_merge) + 1):
                        suffixed_attr = f'{attr_prefix}_{i}'
                        if suffixed_attr in v.attributes():
                            del v[suffixed_attr]
            
            timings['download_and_merge'] = time.time() - start_time
            self._log_graph_stats(merged_graph, "Merged Graphs")

            if not merged_graph.vcount():
                logging.info(f"No graph data found after merging for batch {batch_id}. Stopping execution.")
                self.spanner_ops.release_lock(batch_id, "FAILED")
                return None

            # Phase 3: Clustering and Deduplication
            start_time = time.time()
            logging.info(f"Clustering and Deduplication for batch {batch_id}.")
            clustered_graph = self.graph_processor.cluster_and_merge_entities(merged_graph)
            self._log_graph_stats(clustered_graph, "Clustered Entities")
            clustered_graph_dict = _graph_to_dict(clustered_graph)
            deduplicated_graph_dict = self.graph_processor.deduplicate_entities(clustered_graph_dict)
            deduplicated_graph = _dict_to_graph(deduplicated_graph_dict)
            timings['cluster_and_deduplicate'] = time.time() - start_time
            self._log_graph_stats(deduplicated_graph, "Deduplicated Entities")

            # Phase 4: Community Detection
            start_time = time.time()
            logging.info(f"Community Detection for batch {batch_id}.")
            community_graph = self.graph_processor.run_igraph_community_detection(deduplicated_graph)
            timings['community_detection'] = time.time() - start_time
            self._log_graph_stats(community_graph, "Community Detection")

            # Phase 5: Removing Null Entities/Relationships
            start_time = time.time()
            logging.info(f"Removing Null Entities/Relationships for batch {batch_id}.")
            final_graph = self.graph_processor.remove_entities_with_null_keys_and_relationships(community_graph)
            timings['remove_nulls'] = time.time() - start_time
            self._log_graph_stats(final_graph, "Removed Nulls")
            
            if final_graph:
                # Phase 5.4: Backfill Entity Embeddings
                start_time = time.time()
                logging.info(f"Backfilling missing entity embeddings for batch {batch_id}.")
                final_graph = self.graph_processor.backfill_entity_embeddings(final_graph)
                timings['backfill_entity_embeddings'] = time.time() - start_time
                self._log_graph_stats(final_graph, "Backfilled Entity Embeddings")

                # Phase 5.5: Backfill Relationship Embeddings
                start_time = time.time()
                logging.info(f"Backfilling missing relationship embeddings for batch {batch_id}.")
                final_graph = self.graph_processor.backfill_relationship_embeddings(final_graph)
                timings['backfill_rel_embeddings'] = time.time() - start_time
                self._log_graph_stats(final_graph, "Backfilled Relationship Embeddings")

                # Clean None attributes before serializing
                final_graph = self.graph_processor.remove_none_attributes(final_graph)

                # Phase 6: Serializing and Uploading to GCS
                start_time = time.time()
                logging.info(f"Serializing and uploading graph to GCS for batch {batch_id}.")
                serialized_graph = pickle.dumps(final_graph)
                gcs_bucket_name = os.environ.get("GRAPH_DATA_BUCKET_NAME")
                if not gcs_bucket_name:
                    logging.error("GRAPH_DATA_BUCKET_NAME environment variable not set.")
                    self.spanner_ops.release_lock(batch_id, "FAILED")
                    return None
                bucket = self.storage_client.bucket(gcs_bucket_name)
                timestamp = time.strftime("%Y%m%d%H%M%S")
                object_name = f"graph_data/{batch_id.replace('/', '_').replace(':', '_')}_{timestamp}.pkl"
                blob = bucket.blob(object_name)
                blob.upload_from_string(serialized_graph, content_type='application/octet-stream')
                gcs_path = f"gs://{gcs_bucket_name}/{object_name}"
                timings['upload_to_gcs'] = time.time() - start_time
                logging.info(f"Uploaded serialized graph to GCS: {gcs_path}.")

                # Phase 7: Publishing to Persistor Topic
                start_time = time.time()
                logging.info(f"Publishing message to persistor topic for batch {batch_id}.")
                message_payload = {"batch_id": batch_id, "gcs_path": gcs_path}
                future = self.publisher.publish(persistor_topic_path, json.dumps(message_payload).encode("utf-8"))
                future.result()
                timings['publish_to_persistor'] = time.time() - start_time
                logging.info(f"Published message for batch {batch_id} to topic {persistor_topic_path}.")
            else:
                logging.error(f"No graph data to serialize for batch {batch_id}.")
                self.spanner_ops.release_lock(batch_id, "FAILED")
                return None

            # Phase 8: Releasing Lock
            start_time = time.time()
            logging.info(f"Releasing lock for batch {batch_id} with status PENDING_PERSISTENCE.")
            self.spanner_ops.release_lock(batch_id, "PENDING_PERSISTENCE")
            timings['release_lock'] = time.time() - start_time
            
            total_duration = time.time() - total_start_time
            timings['total_duration'] = total_duration

            # Final Summary Logs
            logging.info(f"--- FINAL CONSOLIDATOR TIMING SUMMARY FOR BATCH {batch_id} ---")
            for phase, duration in timings.items():
                logging.info(f"{phase:<25}: {duration:.4f}")
            logging.info("----------------------------------------------------")

        except Exception as e:
            logging.error(f'An error occurred while processing batch {batch_id}: {e}.', exc_info=True)
            self.spanner_ops.release_lock(batch_id, "FAILED")
            raise e
