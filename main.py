import functions_framework
import logging
import json
import psutil
import google.cloud.logging

from clients import ClientFactory
from pubsub_handler import decode_pubsub_message
from redis_operations import RedisOperations
from spanner_operations import SpannerOperations
from llm_operations import LLMOperations
from graph_processing import GraphProcessor

# --- Boilerplate and Configuration ---
logging_client = google.cloud.logging.Client()
logging_client.setup_logging()
logging.basicConfig(level=logging.INFO)

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
    batch_id = None
    if cloud_event.data.get("path") == "/":
        return "OK", 200
    try:
        # Log system resource usage
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        logging.info(f"System CPU Usage: {cpu_usage}%")
        logging.info(f"System Memory: Total={memory_info.total / 1024**3:.2f}GB, Available={memory_info.available / 1024**3:.2f}GB, Used={memory_info.used / 1024**3:.2f}GB, Percentage={memory_info.percent}%")

        data = decode_pubsub_message(cloud_event)
        batch_id = data.get("batch_id")

        client_factory = ClientFactory()
        redis_client = client_factory.get_redis_client()
        db_session = client_factory.get_db_session()
        llm = client_factory.get_llm()

        redis_ops = RedisOperations(redis_client)
        spanner_ops = SpannerOperations(db_session)
        llm_ops = LLMOperations(llm)
        graph_processor = GraphProcessor(llm_ops)
        
        spanner_ops.ensure_spanner_schema()

        consolidated_key = f"consolidated_batch:{batch_id}"
        if redis_client.exists(consolidated_key):
            logging.info(f"Consolidated data for batch {batch_id} found in Redis. Skipping processing chain and migrating directly to Spanner.")
            redis_data = redis_client.hgetall(consolidated_key)
            entities = json.loads(redis_data["entities"])
            relationships = json.loads(redis_data["relationships"])
            data_from_redis = {
                "batch_id": batch_id,
                "entities": entities,
                "relationships": relationships
            }
            spanner_ops.migrate_to_spanner(data_from_redis)
            spanner_ops.update_workflow_status(batch_id, "SUCCEEDED")
            return "OK", 200

        fetched_data = redis_ops.fetch_from_redis(data)
        if not fetched_data.get("partial_results"):
            logging.info(f"No data found in Redis for batch {data.get('batch_id')}. Stopping execution.")
            return "OK", 200

        aggregated_data = graph_processor.aggregate_results(fetched_data)
        embedded_data = llm_ops.generate_embeddings(aggregated_data)
        clustered_data = graph_processor.cluster_and_merge_entities(embedded_data)
        deduplicated_data = graph_processor.deduplicate_entities(clustered_data)
        community_data = graph_processor.run_igraph_community_detection(deduplicated_data)
        
        redis_ops.store_consolidated_results_in_redis(community_data)
        spanner_ops.migrate_to_spanner(community_data)
        
        spanner_ops.update_workflow_status(batch_id, "SUCCEEDED")

        return "OK", 200
    except Exception as e:
        batch_id = batch_id or (data and data.get("batch_id"))
        logging.error(f'An error occurred in the consolidator for batch_id {batch_id}: {e}', exc_info=True)
        
        if batch_id:
            try:
                client_factory = ClientFactory()
                db_session = client_factory.get_db_session()
                spanner_ops = SpannerOperations(db_session)
                spanner_ops.update_workflow_status(batch_id, "FAILED")
            except Exception as spanner_e:
                logging.error(f"Could not update workflow status for batch ID {batch_id} to FAILED: {spanner_e}", exc_info=True)

        return "Internal Server Error", 500