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
import os
import psutil
import google.cloud.logging


from clients import ClientFactory
from pubsub_handler import decode_pubsub_message
from llm_operations import LLMOperations
from graph_processing import GraphProcessor
from spanner_operations import SpannerOperations
from consolidator_service import ConsolidatorService # New import

import uuid

# --- Boilerplate and Configuration ---

@functions_framework.cloud_event
def consolidator(cloud_event):
    invocation_id = str(uuid.uuid4())
    data = None # Initialize data to None
    batch_id = None # Initialize batch_id to None
    
    # Configure logging to use Cloud Logging
    client = google.cloud.logging.Client()
    client.setup_logging()

    logging.info(f"Consolidator function started. Invocation ID: {invocation_id}")
    logging.info(f"GRAPH_DATA_BUCKET_NAME from env: {os.environ.get('GRAPH_DATA_BUCKET_NAME')}")
    
    try:
        logging.info("Initializing clients...")
        client_factory = ClientFactory()
        llm = client_factory.get_llm()
        publisher = client_factory.get_publisher()
        spanner_client = client_factory.get_spanner_client()
        storage_client = client_factory.get_storage_client()

        llm_ops = LLMOperations(llm)
        graph_processor = GraphProcessor(llm_ops)
        spanner_ops = SpannerOperations(spanner_client, os.environ.get("SPANNER_INSTANCE_ID"), os.environ.get("SPANNER_DATABASE_ID"))
        
        # Initialize ConsolidatorService
        consolidator_service = ConsolidatorService(llm_ops, graph_processor, spanner_ops, publisher, storage_client, invocation_id)

        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        logging.info(f"System CPU Usage: {cpu_usage}%")
        logging.info(f"System Memory: Total={memory_info.total / 1024**3:.2f}GB, Available={memory_info.available / 1024**3:.2f}GB, Used={memory_info.used / 1024**3:.2f}GB, Percentage={memory_info.percent}%")

        data = decode_pubsub_message(cloud_event)
        batch_id = data.get("batch_id")
        gcs_paths = data.get("gcs_paths", [])

        persistor_topic_name = os.environ.get("PERSISTOR_TOPIC_NAME")
        if not persistor_topic_name:
            logging.error(f"PERSISTOR_TOPIC_NAME environment variable not set. Invocation ID: {invocation_id}")
            if batch_id:
                spanner_ops.release_lock(batch_id, "FAILED")
            return None
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        topic_path = publisher.topic_path(project_id, persistor_topic_name)

        instance_id = os.environ.get("GAE_INSTANCE")

        # Call the service to process the message
        consolidator_service.process_message(batch_id, gcs_paths, topic_path, instance_id)

    except Exception as e:
        batch_id = batch_id or (data and data.get("batch_id"))
        logging.error(f'An error occurred in the consolidator for batch_id {batch_id}. Invocation ID: {invocation_id}: {e}', exc_info=True)
        return None
    finally:
        logging.info(f"--- END OF CONSOLIDATOR INVOCATION. Invocation ID: {invocation_id} ---")
