import functions_framework
import logging
import json
import os # Added for environment variable access
import psutil
import google.cloud.logging
from google.cloud import pubsub_v1 # Added for Pub/Sub publishing

# Forced rebuild comment: 2025-09-07-4

from clients import ClientFactory
from pubsub_handler import decode_pubsub_message
from redis_operations import RedisOperations
from llm_operations import LLMOperations
from graph_processing import GraphProcessor
from spanner_operations import SpannerOperations

# --- Boilerplate and Configuration ---
logging_client = google.cloud.logging.Client()
logging_client.setup_logging()
logging.basicConfig(level=logging.DEBUG)

@functions_framework.cloud_event
def consolidator(cloud_event):
    batch_id = None
    try:
        # Initialize clients and operations inside the function
        client_factory = ClientFactory()
        redis_client = client_factory.get_redis_client()
        llm = client_factory.get_llm()
        publisher = pubsub_v1.PublisherClient()
        spanner_client = client_factory.get_spanner_client()

        redis_ops = RedisOperations(redis_client)
        llm_ops = LLMOperations(llm)
        graph_processor = GraphProcessor(llm_ops)
        spanner_ops = SpannerOperations(spanner_client, os.environ.get("SPANNER_INSTANCE_ID"), os.environ.get("SPANNER_DATABASE_ID"))

        # Log system resource usage
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        logging.info(f"System CPU Usage: {cpu_usage}%")
        logging.info(f"System Memory: Total={memory_info.total / 1024**3:.2f}GB, Available={memory_info.available / 1024**3:.2f}GB, Used={memory_info.used / 1024**3:.2f}GB, Percentage={memory_info.percent}%")

        data = decode_pubsub_message(cloud_event)
        batch_id = data.get("batch_id")

        instance_id = os.environ.get("GAE_INSTANCE") # Unique ID for the instance

        if not spanner_ops.acquire_lock(batch_id, instance_id):
            return None # Another instance is processing this batch

        try:
            fetched_data = redis_ops.fetch_from_redis(data)
            if not fetched_data.get("partial_results"):
                logging.info(f"No data found in Redis for batch {data.get('batch_id')}. Stopping execution.")
                spanner_ops.release_lock(batch_id, "FAILED")
                return None

            aggregated_data = graph_processor.aggregate_results(fetched_data)
            embedded_data = llm_ops.generate_embeddings(aggregated_data)
            clustered_data = graph_processor.cluster_and_merge_entities(embedded_data)
            deduplicated_data = graph_processor.deduplicate_entities(clustered_data)
            community_data = graph_processor.run_igraph_community_detection(deduplicated_data)
            
            # Save processed data to Redis
            redis_ops.save_processed_data(batch_id, community_data)

            # Publish message to trigger persistor function
            project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") # Assuming project ID is available as an environment variable
            topic_name = os.environ.get("PERSISTOR_TOPIC_NAME") # Get topic name from environment variable
            topic_path = publisher.topic_path(project_id, topic_name)
            
            future = publisher.publish(topic_path, json.dumps({"batch_id": batch_id}).encode("utf-8"))
            future.result() # Wait for the publish call to complete
            logging.info(f"Published message for batch {batch_id} to topic {topic_name}")

            spanner_ops.release_lock(batch_id, "COMPLETED")

        except Exception as e:
            logging.error(f'An error occurred while processing batch {batch_id}: {e}', exc_info=True)
            spanner_ops.release_lock(batch_id, "FAILED")
            raise e # Re-raise the exception to trigger a retry if configured
    except Exception as e:
        batch_id = batch_id or (data and data.get("batch_id"))
        logging.error(f'An error occurred in the consolidator for batch_id {batch_id}: {e}', exc_info=True)
        
        return None