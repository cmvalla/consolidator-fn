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

# --- Boilerplate and Configuration ---
logging_client = google.cloud.logging.Client()
logging_client.setup_logging()
logging.basicConfig(level=logging.DEBUG)

# --- Global Client and Operations Initialization (executed once per instance) ---
client_factory = ClientFactory()
redis_client = client_factory.get_redis_client()
llm = client_factory.get_llm()
publisher = pubsub_v1.PublisherClient() # Added for Pub/Sub publishing

redis_ops = RedisOperations(redis_client)
llm_ops = LLMOperations(llm)
graph_processor = GraphProcessor(llm_ops)

def aggregate_results(data):
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

@functions_framework.cloud_event
def consolidator(cloud_event):
    global redis_ops, spanner_ops, llm_ops, graph_processor
    batch_id = None
    try:
        # Log system resource usage
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        logging.info(f"System CPU Usage: {cpu_usage}%")
        logging.info(f"System Memory: Total={memory_info.total / 1024**3:.2f}GB, Available={memory_info.available / 1024**3:.2f}GB, Used={memory_info.used / 1024**3:.2f}GB, Percentage={memory_info.percent}%")

        data = decode_pubsub_message(cloud_event)
        batch_id = data.get("batch_id")

        

        fetched_data = redis_ops.fetch_from_redis(data)
        if not fetched_data.get("partial_results"):
            logging.info(f"No data found in Redis for batch {data.get('batch_id')}. Stopping execution.")
            return None

        aggregated_data = graph_processor.aggregate_results(fetched_data)
        embedded_data = llm_ops.generate_embeddings(aggregated_data)
        clustered_data = graph_processor.cluster_and_merge_entities(embedded_data)
        deduplicated_data = graph_processor.deduplicate_entities(clustered_data)
        community_data = graph_processor.run_igraph_community_detection(deduplicated_data)
        
        # Save processed data to Redis
        redis_ops.save_processed_data(batch_id, community_data)

        # Publish message to trigger persistor function
        project_id = os.environ.get("GCP_PROJECT") # Assuming project ID is available as an environment variable
        topic_name = os.environ.get("PERSISTOR_TOPIC_NAME") # Get topic name from environment variable
        topic_path = publisher.topic_path(project_id, topic_name)
        
        future = publisher.publish(topic_path, json.dumps({"batch_id": batch_id}).encode("utf-8"))
        future.result() # Wait for the publish call to complete
        logging.info(f"Published message for batch {batch_id} to topic {topic_name}")

        return None
    except Exception as e:
        batch_id = batch_id or (data and data.get("batch_id"))
        logging.error(f'An error occurred in the consolidator for batch_id {batch_id}: {e}', exc_info=True)
        
        return None