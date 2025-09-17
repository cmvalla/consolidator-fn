# This Cloud Function acts as the 'consolidator' in the data processing pipeline.
# Its primary responsibilities include:
# 1. Fetching partial graph data (entities and relationships) from Redis.
# 2. Aggregating this partial data into a comprehensive graph.
# 3. Generating embeddings for entities using an LLM.
# 4. Performing entity clustering and deduplication.
# 5. Running community detection algorithms on the graph.
# 6. Serializing the processed graph data and uploading it to Google Cloud Storage (GCS).
# 7. Publishing a Pub/Sub message to trigger the 'persistor' function, passing the GCS path of the processed graph.

import functions_framework
import logging
import json
import os # Added for environment variable access
import psutil
import google.cloud.logging
from google.cloud import pubsub_v1 # Added for Pub/Sub publishing
import pickle # Added for igraph serialization
from google.cloud import storage # Added for GCS operations
import datetime # Added for timestamp in GCS object name

# Forced rebuild comment: 2025-09-07-4

from clients import ClientFactory
from pubsub_handler import decode_pubsub_message
from llm_operations import LLMOperations
from graph_processing import GraphProcessor
from spanner_operations import SpannerOperations

# --- Boilerplate and Configuration ---
logging_client = google.cloud.logging.Client()
logging_client.setup_logging()
logging.basicConfig(level=logging.DEBUG)

@functions_framework.cloud_event
def consolidator(cloud_event):
    logging.info("Consolidator function started.")
    # Initialize batch_id to None. This variable will hold the ID of the current processing batch,
    # crucial for tracking and managing locks in Spanner.
    logging.info(f"--- START OF CONSOLIDATOR INVOCATION ---")
    logging.info(f"GRAPH_DATA_BUCKET_NAME from env: {os.environ.get('GRAPH_DATA_BUCKET_NAME')}")
    batch_id = None
    try:
        logging.info("Initializing clients...")
        # Initialize various clients and operation classes. These are instantiated inside the function
        # to ensure a fresh state for each Cloud Function invocation and to properly handle
        # potential connection issues or resource leaks across invocations.
        client_factory = ClientFactory()
        logging.info("ClientFactory initialized.")
        llm = client_factory.get_llm()
        logging.info("LLM client initialized.")
        publisher = pubsub_v1.PublisherClient()
        logging.info("Pub/Sub client initialized.")
        spanner_client = client_factory.get_spanner_client()
        logging.info("Spanner client initialized.")

        llm_ops = LLMOperations(llm)
        logging.info("LLMOperations initialized.")
        graph_processor = GraphProcessor(llm_ops)
        logging.info("GraphProcessor initialized.")
        # SpannerOperations requires instance and database IDs, which are fetched from environment variables.
        # This promotes flexibility and avoids hardcoding sensitive configuration.
        spanner_ops = SpannerOperations(spanner_client, os.environ.get("SPANNER_INSTANCE_ID"), os.environ.get("SPANNER_DATABASE_ID"))
        logging.info("SpannerOperations initialized.")

        # Log system resource usage to monitor the function's performance and resource consumption.
        # This helps in debugging performance bottlenecks and optimizing resource allocation for the Cloud Function.
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        logging.info(f"System CPU Usage: {cpu_usage}%")
        logging.info(f"System Memory: Total={memory_info.total / 1024**3:.2f}GB, Available={memory_info.available / 1024**3:.2f}GB, Used={memory_info.used / 1024**3:.2f}GB, Percentage={memory_info.percent}%")

        # Decode the Pub/Sub message that triggered this Cloud Function.
        # The message is expected to contain a 'batch_id' and 'gcs_paths' which identifies the data batch to be processed.
        data = decode_pubsub_message(cloud_event)
        batch_id = data.get("batch_id")
        gcs_paths = data.get("gcs_paths", [])

        if not gcs_paths:
            logging.error(f"No GCS paths found in Pub/Sub message for batch {batch_id}. Stopping execution.")
            if batch_id:
                spanner_ops.release_lock(batch_id, "FAILED")
            return None

        # Retrieve the Pub/Sub topic name for the 'persistor' function from environment variables.
        # This topic is used to send the processed graph data to the next stage of the pipeline.
        persistor_topic_name = os.environ.get("PERSISTOR_TOPIC_NAME")
        if not persistor_topic_name:
            logging.error("PERSISTOR_TOPIC_NAME environment variable not set.")
            # Release the lock if the batch_id is available, indicating a failure to proceed.
            if batch_id:
                spanner_ops.release_lock(batch_id, "FAILED")
            return None
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        topic_path = publisher.topic_path(project_id, persistor_topic_name)

        # Get the unique instance ID of the Cloud Run revision. This is used for acquiring
        # and releasing locks in Spanner to prevent multiple instances from processing the same batch concurrently.
        instance_id = os.environ.get("GAE_INSTANCE")

        # Attempt to acquire a lock for the current batch in Spanner. This is a critical step
        # to ensure idempotency and prevent race conditions in a distributed environment.
        if not spanner_ops.acquire_lock(batch_id, instance_id):
            # If the lock cannot be acquired, it means another instance is already processing this batch.
            # In this case, the current invocation should gracefully exit.
            return None

        try:
            # Download and deserialize igraph objects from GCS
            storage_client = storage.Client()
            merged_graph = ig.Graph()

            for path in gcs_paths:
                try:
                    bucket_name = path.split("//")[1].split("/")[0]
                    blob_name = "/".join(path.split("//")[1].split("/")[1:])
                    bucket = storage_client.bucket(bucket_name)
                    blob = bucket.blob(blob_name)
                    
                    downloaded_blob = blob.download_as_bytes()
                    worker_graph = pickle.loads(downloaded_blob)
                    
                    # Merge worker_graph into merged_graph
                    # This is a simplified merge. A more robust merge would handle overlapping entities/relationships
                    # based on IDs and update properties accordingly.
                    merged_graph = ig.Graph.union(merged_graph, worker_graph)
                    logging.info(f"Successfully downloaded and merged graph from {path}")
                except Exception as e:
                    logging.error(f"Error downloading or deserializing graph from {path}: {e}", exc_info=True)
                    # Continue processing other graphs, but log the error
            
            if not merged_graph.vcount():
                logging.info(f"No graph data found after merging for batch {batch_id}. Stopping execution.")
                spanner_ops.release_lock(batch_id, "FAILED")
                return None

            # --- Graph Processing Pipeline ---
            # The graph_processor now directly receives the merged igraph
            # 1. Generate embeddings for entities. Embeddings are numerical representations
            # that capture the semantic meaning of entities, crucial for clustering and similarity searches.
            embedded_graph = llm_ops.generate_embeddings(merged_graph)
            # 2. Cluster and merge similar entities. This step identifies and groups
            # entities that represent the same real-world concept, creating 'Class' nodes.
            clustered_graph = graph_processor.cluster_and_merge_entities(embedded_graph)
            # 3. Deduplicate entities to resolve any remaining duplicate IDs or representations
            # before community detection, ensuring data integrity.
            deduplicated_graph = graph_processor.deduplicate_entities(clustered_graph)
            # 4. Run community detection algorithms (e.g., maximal cliques) on the graph.
            # This identifies densely connected groups of entities, forming 'Community' nodes.
            community_graph = graph_processor.run_igraph_community_detection(deduplicated_graph)
            # 5. Remove any entities or relationships that might have null or empty IDs,
            # ensuring the final graph structure is valid before persistence.
            final_graph = graph_processor.remove_entities_with_null_keys_and_relationships(community_graph)
            
            # Serialize the processed igraph object and upload it to Google Cloud Storage (GCS).
            # GCS is used for durable storage of the large graph object, as Pub/Sub messages have size limits.
            if final_graph:
                try:
                    # Use pickle for serialization of the Python object (igraph graph).
                    serialized_graph = pickle.dumps(final_graph)
                    
                    # Get the GCS bucket name from environment variables.
                    gcs_bucket_name = os.environ.get("GRAPH_DATA_BUCKET_NAME")
                    if not gcs_bucket_name:
                        logging.error("GRAPH_DATA_BUCKET_NAME environment variable not set.")
                        spanner_ops.release_lock(batch_id, "FAILED")
                        return None

                    storage_client = storage.Client()
                    bucket = storage_client.bucket(gcs_bucket_name)
                    
                    # Generate a unique object name for the GCS blob using the batch_id and a timestamp.
                    # This ensures that each processed graph is stored uniquely and can be easily retrieved.
                    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
                    object_name = f"graph_data/{batch_id.replace('/', '_').replace(':', '_')}_{timestamp}.pkl"
                    blob = bucket.blob(object_name)
                    
                    # Upload the serialized graph data to GCS.
                    blob.upload_from_string(serialized_graph)
                    gcs_path = f"gs://{gcs_bucket_name}/{object_name}"
                    logging.info(f"Uploaded serialized graph to GCS: {gcs_path}")

                    # Publish a Pub/Sub message to the 'persistor' topic.
                    # The message payload includes the batch_id and the GCS path to the serialized graph.
                    # This triggers the 'persistor' function to retrieve and store the graph data in Spanner.
                    message_payload = {
                        "batch_id": batch_id,
                        "gcs_path": gcs_path
                    }
                    future = publisher.publish(topic_path, json.dumps(message_payload).encode("utf-8"))
                    future.result() # Wait for the publish call to complete to ensure message delivery.
                    logging.info(f"Published message with GCS path for batch {batch_id} to topic {persistor_topic_name}")

                except Exception as e:
                    logging.error(f"Error serializing or uploading graph to GCS for batch {batch_id}: {e}", exc_info=True)
                    spanner_ops.release_lock(batch_id, "FAILED")
                    return None
            else:
                # This condition indicates a critical error: no graph data was generated after processing.
                logging.error(f"No community_data to serialize for batch {batch_id}. This is an error condition.")
                spanner_ops.release_lock(batch_id, "FAILED")
                return None # Stop execution

            # Release the lock for the current batch in Spanner, marking it as 'COMPLETED'.
            # This signals that the batch has been successfully processed by the consolidator.
            spanner_ops.release_lock(batch_id, "PENDING_PERSISTENCE")

        except Exception as e:
            # Catch any exceptions that occur during the main processing logic.
            logging.error(f'An error occurred while processing batch {batch_id}: {e}', exc_info=True)
            # Release the lock with a 'FAILED' status, indicating an issue during processing.
            spanner_ops.release_lock(batch_id, "FAILED")
            # Re-raise the exception to allow Cloud Functions to handle retries if configured.
            raise e
    except Exception as e:
        # Catch any exceptions that occur outside the main processing try-except block,
        # typically during initial setup or lock acquisition.
        # Attempt to get batch_id if it wasn't set earlier.
        batch_id = batch_id or (data and data.get("batch_id"))
        logging.error(f'An error occurred in the consolidator for batch_id {batch_id}: {e}', exc_info=True)
        
        return None