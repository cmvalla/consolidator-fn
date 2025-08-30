import base64
import json
import re
import os
import functions_framework
import google.cloud.logging
import logging
import redis
import numpy as np
from langchain_community.graphs import MemgraphGraph
from langchain_google_vertexai import VertexAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
import google.cloud.secretmanager as secretmanager
import time
import google.cloud.spanner_v1 as spanner
import psutil
from google.api_core.exceptions import AlreadyExists, FailedPrecondition



# --- Boilerplate and Configuration ---
logging_client = google.cloud.logging.Client()
logging_client.setup_logging()
logging.basicConfig(level=logging.INFO)

# --- Global Clients (initialized within the function) ---
redis_client = None
llm = None
memgraph_graph = None
spanner_client = None
spanner_instance = None
spanner_database = None


def ensure_spanner_graph_exists(database):
    """
    This function is now a placeholder. The schema is managed by Terraform.
    The DDL definition can be found in `terraform/spanner.tf`.
    """
    logging.info("Schema management is now handled by Terraform. Skipping DDL updates from the function.")
    pass

def initialize_clients():
    """Initializes all external clients."""
    global redis_client, llm, memgraph_graph, spanner_client, spanner_instance, spanner_database, embedding_model

    try:
        # Log system resource usage
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        logging.info(f"System CPU Usage: {cpu_usage}%")
        logging.info(f"System Memory: Total={memory_info.total / 1024**3:.2f}GB, Available={memory_info.available / 1024**3:.2f}GB, Used={memory_info.used / 1024**3:.2f}GB, Percentage={memory_info.percent}%")
        
        logging.info("Initializing global clients...")

        # --- Environment Variables ---
        GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
        REDIS_HOST = os.environ.get("REDIS_HOST")
        REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
        MEMGRAPH_HOST = os.environ.get("MEMGRAPH_HOST", "memgraph-service.memgraph.svc.cluster.local")
        MEMGRAPH_PORT = int(os.environ.get("MEMGRAPH_PORT", 7687))
        MEMGRAPH_USER = os.environ.get("MEMGRAPH_USER", "memgraph")
        MEMGRAPH_PASSWORD = os.environ.get("MEMGRAPH_PASSWORD")
        SPANNER_INSTANCE_ID = os.environ.get("SPANNER_INSTANCE_ID")
        SPANNER_DATABASE_ID = os.environ.get("SPANNER_DATABASE_ID")
        LOCATION = os.environ.get("LOCATION")
        
        # --- Secret Manager ---
        sm_client = secretmanager.SecretManagerServiceClient()
        REDIS_PASSWORD = sm_client.access_secret_version(request={"name": f"projects/{GCP_PROJECT}/secrets/redis-password/versions/latest"}).payload.data.decode("UTF-8")

        logging.info("Initializing Redis client...")
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, ssl=False, ssl_cert_reqs=None, decode_responses=True, socket_connect_timeout=10)
        redis_client.ping()
        logging.info("Redis client initialized successfully.")

        logging.info("Initializing Memgraph client...")
        os.environ["NEO4J_USERNAME"] = MEMGRAPH_USER
        os.environ["NEO4J_PASSWORD"] = MEMGRAPH_PASSWORD
        memgraph_graph = MemgraphGraph(url=f"bolt://{MEMGRAPH_HOST}:{MEMGRAPH_PORT}", username=MEMGRAPH_USER, password=MEMGRAPH_PASSWORD)
        logging.info("Memgraph client initialized successfully.")

        logging.info("Initializing Vertex AI...")
        llm = VertexAI(model_name="gemini-2.5-flash", location=LOCATION)
        logging.info("Vertex AI clients initialized successfully.")

        logging.info("Initializing Spanner client...")
        spanner_client = spanner.Client(project=GCP_PROJECT)
        spanner_instance = spanner_client.instance(SPANNER_INSTANCE_ID)
        spanner_database = spanner_instance.database(SPANNER_DATABASE_ID)
        logging.info("Spanner client initialized successfully.")

        

        logging.info("All global clients initialized successfully.")
    except Exception as e:
        logging.critical(f'FATAL: Failed to initialize one or more global clients: {e}', exc_info=True)
        raise  # Re-raise the exception to halt execution if initialization fails

# --- LangChain Runnables ---
def decode_pubsub_message(cloud_event):
    message_data = base64.b64decode(cloud_event.data["message"]["data"]).decode("utf-8")
    message_json = json.loads(message_data)
    return {"batch_id": message_json.get("batch_id")}

def fetch_from_redis(data):
    batch_id = data["batch_id"]
    results_key = f"batch:{batch_id}:results"
    partial_results_str = redis_client.lrange(results_key, 0, -1)
    return {"batch_id": batch_id, "partial_results": partial_results_str}

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

import requests

def generate_embeddings(data):
    """Generates embeddings for all entities by calling the embedding service."""
    embedding_service_url = os.environ.get("EMBEDDING_SERVICE_URL")
    if not embedding_service_url:
        logging.error("EMBEDDING_SERVICE_URL environment variable not set.")
        # Decide how to handle this error: raise exception, return data as is, etc.
        # For now, let's return the data without embeddings.
        return data

    entities = data.get("entities", [])
    for entity in entities:
        text_to_embed = f"Type: {entity.get('type', '')}, Properties: {json.dumps(entity.get('properties', {}))}"
        try:
            response = requests.post(embedding_service_url, json={"text": text_to_embed})
            response.raise_for_status()  # Raise an exception for bad status codes
            embedding = response.json().get("embedding")
            if embedding:
                entity['embedding'] = embedding
            else:
                logging.warning(f"Embedding not found in response for entity: {entity.get('id')}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Error calling embedding service for entity {entity.get('id')}: {e}")
            # Decide how to handle this error, e.g., skip embedding for this entity
    return data

def cluster_and_merge_entities(data):
    """
    Clusters similar entities, creates Class and Instance nodes, promotes relationships, and aggregates their weights.
    """
    entities = data.get("entities", [])
    relationships = data.get("relationships", [])

    if not entities:
        return data

    prompt = PromptTemplate(
        input_variables=["entities"],
        template="""
        From the following list of entities, identify entities that refer to the same real-world object and group them.
        For each group, choose one canonical entity and provide a mapping from the old entity IDs to the new canonical entity ID.
        Entities: {entities}
        Mapping:
        """
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    entity_list_str = json.dumps(entities, indent=2)
    llm_response = chain.run(entities=entity_list_str)

    try:
        json_match = re.search(r"\{.*\}", llm_response, re.DOTALL)
        if not json_match:
            logging.error(f"No JSON object found in LLM response: {llm_response}")
            return data
        entity_id_map = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON from LLM response: {llm_response}")
        return data

    grouped_entities = {}
    for entity in entities:
        canonical_id = entity_id_map.get(entity["id"], entity["id"])
        if canonical_id not in grouped_entities:
            grouped_entities[canonical_id] = []
        grouped_entities[canonical_id].append(entity)

    new_entities = []
    instance_of_relationships = []

    for canonical_id, group in grouped_entities.items():
        if len(group) > 1:
            common_properties = group[0].get("properties", {}).copy()
            for entity in group[1:]:
                entity_properties = entity.get("properties", {})
                common_keys = common_properties.keys() & entity_properties.keys()
                common_properties = {k: common_properties[k] for k in common_keys if common_properties[k] == entity_properties[k]}

            class_entity = {
                "id": f"class_{canonical_id}",
                "type": "Class",
                "properties": common_properties,
                "embedding": np.mean([e['embedding'] for e in group], axis=0).tolist()
            }
            new_entities.append(class_entity)

            for entity in group:
                entity["type"] = "Instance"
                instance_of_relationships.append({"source": entity["id"], "target": class_entity["id"], "type": "INSTANCE_OF"})
                new_entities.append(entity)
        else:
            entity = group[0]
            entity["type"] = "Class"
            new_entities.append(entity)

    # Promote and aggregate relationships
    promoted_rels = {}
    for rel in relationships:
        source_class_id = f"class_{entity_id_map.get(rel['source'], rel['source'])}"
        target_class_id = f"class_{entity_id_map.get(rel['target'], rel['target'])}"
        rel_type = rel["type"]
        weight = rel.get("properties", {}).get("weight", 1.0)

        rel_key = (source_class_id, target_class_id, rel_type)
        if rel_key not in promoted_rels:
            promoted_rels[rel_key] = {"weights": [], "properties": rel.get("properties", {})}
        promoted_rels[rel_key]["weights"].append(weight)

    aggregated_relationships = []
    for (source, target, type), data in promoted_rels.items():
        avg_weight = np.mean(data["weights"])
        final_properties = data["properties"]
        final_properties["weight"] = avg_weight
        final_properties["original_relations_count"] = len(data["weights"])
        aggregated_relationships.append({
            "source": source,
            "target": target,
            "type": type,
            "properties": final_properties
        })

    data["entities"] = new_entities
    data["relationships"] = aggregated_relationships + instance_of_relationships
    
    logging.info(f"Clustering complete. Result: {len(new_entities)} entities, {len(aggregated_relationships)} aggregated relationships, {len(instance_of_relationships)} instance-of relationships.")
    return data

def load_to_memgraph(data):
    entities = data.get("entities", [])
    if entities:
        logging.info(f"Loading {len(entities)} entities to Memgraph.")
        instance_query = "UNWIND $nodes AS node MERGE (n:Entity {id: node.id}) SET n += node, n :Instance"
        class_query = "UNWIND $nodes AS node MERGE (n:Entity {id: node.id}) SET n += node, n :Class"
        
        instances = [e for e in entities if e['type'] == 'Instance']
        classes = [e for e in entities if e['type'] == 'Class']
        
        if instances:
            memgraph_graph.query(instance_query, params={"nodes": instances})
        if classes:
            memgraph_graph.query(class_query, params={"nodes": classes})

    relationships = data.get("relationships", [])
    if relationships:
        logging.info(f"Loading {len(relationships)} relationships to Memgraph.")
        rel_query = "UNWIND $rels AS rel MATCH (a:Entity {id: rel.source}), (b:Entity {id: rel.target}) CREATE (a)-[r:`{rel.type}` {properties: rel.properties}]->(b)"
        memgraph_graph.query(rel_query, params={"rels": relationships})

    return data

def migrate_to_spanner(data):
    logging.info("Migrating data to Spanner...")

    entities = data.get("entities", [])
    relationships = data.get("relationships", [])

    entities_to_insert = [(e["id"], e["type"], json.dumps(e.get("properties", {})), e.get("embedding")) for e in entities]
    relationships_to_insert = [(r["source"], r["target"], r["type"], json.dumps(r.get("properties", {}))) for r in relationships if r['type'] != 'INSTANCE_OF']
    instance_of_to_insert = [(r["source"], r["target"]) for r in relationships if r['type'] == 'INSTANCE_OF']

    def chunk_list(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    BATCH_SIZE = 100

    for batch in chunk_list(entities_to_insert, BATCH_SIZE):
        spanner_database.run_in_transaction(lambda transaction: transaction.insert(
            table="Entities",
            columns=("Eid", "Type", "Properties", "Embedding"),
            values=batch,
        ))
        logging.info(f"Inserted {len(batch)} entities.")

    for batch in chunk_list(relationships_to_insert, BATCH_SIZE):
        spanner_database.run_in_transaction(lambda transaction: transaction.insert(
            table="Relationships",
            columns=("SourceEid", "TargetEid", "Type", "Properties"),
            values=batch,
        ))
        logging.info(f"Inserted {len(batch)} relationships.")

    for batch in chunk_list(instance_of_to_insert, BATCH_SIZE):
        spanner_database.run_in_transaction(lambda transaction: transaction.insert(
            table="InstanceOf",
            columns=("InstanceEid", "ClassEid"),
            values=batch,
        ))
        logging.info(f"Inserted {len(batch)} instance-of relationships.")

    return data

# --- LangChain Sequence ---
consolidation_chain = RunnableSequence(
    decode_pubsub_message,
    fetch_from_redis,
    aggregate_results,
    generate_embeddings,
    cluster_and_merge_entities,
    load_to_memgraph,
    migrate_to_spanner,
    # ... (add other steps like community detection, summarization etc. back here)
    #     # cleanup_redis,
)

# --- Main Function ---
@functions_framework.cloud_event
def consolidator(cloud_event):
    try:
        initialize_clients()
        consolidation_chain.invoke(cloud_event)
        return "OK", 200
    except Exception as e:
        logging.error(f'An error occurred in the consolidator: {e}', exc_info=True)
        return "Internal Server Error", 500
