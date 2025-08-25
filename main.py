import base64
import json
import os
import functions_framework
import google.cloud.logging
import logging
import redis
from langchain_community.graphs import MemgraphGraph
from langchain_google_vertexai import VertexAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
import google.cloud.secretmanager as secretmanager

import time
import time
import google.cloud.spanner_v1 as spanner

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

def initialize_clients():
    """Initializes all external clients."""
    global redis_client, llm, memgraph_graph, spanner_client, spanner_instance, spanner_database

    try:
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
        for entity in res_json.get("entities", []):
            all_entities[entity["id"]] = entity
        all_relationships.extend(res_json.get("relationships", []))
    
    logging.info(f"Aggregated {len(all_entities)} entities and {len(all_relationships)} relationships.")

    return {
        "batch_id": data["batch_id"],
        "entities": list(all_entities.values()),
        "relationships": all_relationships
    }

def load_to_memgraph(data):
    memgraph_graph.query("MATCH (n) DETACH DELETE n")
    entities = data.get("entities", [])
    if entities:
        logging.info(f"Loading {len(entities)} entities to Memgraph.")
        node_query = "UNWIND $nodes AS node CREATE (:Entity {id: node.id, type: node.type, properties: node.properties})"
        memgraph_graph.query(node_query, params={'nodes': entities})
    relationships = data.get("relationships", [])
    if relationships:
        logging.info(f"Loading {len(relationships)} relationships to Memgraph.")
        rel_query = "UNWIND $rels AS rel MATCH (a:Entity {id: rel.source}), (b:Entity {id: rel.target}) CREATE (a)-[:RELATIONSHIP {type: rel.type, properties: rel.properties}]->(b)"
        memgraph_graph.query(rel_query, params={'rels': relationships})
    return data

def run_community_detection(data):
    community_query = "CALL community_detection.get() YIELD node, community_id"
    result = memgraph_graph.query(community_query)
    for record in result:
        logging.info(f"Community detection record: {record}")
        logging.info(f"Node in record: {record['node']}")
        node_id = record["node"].get("id")
        community_id = record["community_id"]
        memgraph_graph.query("MERGE (c:Community {id: $community_id})", params={"community_id": community_id})
        memgraph_graph.query("MATCH (e:Entity {id: $node_id}), (c:Community {id: $community_id}) CREATE (e)-[:BELONGS_TO]->(c)", params={"node_id": node_id, "community_id": community_id})
    return data

def generate_summaries(data):
    get_communities_query = "MATCH (c:Community)<-[:BELONGS_TO]-(e:Entity) RETURN c.id AS communityId, COLLECT({id: e.id, properties: e.properties}) AS nodes"
    community_results = memgraph_graph.query(get_communities_query)
    
    summaries = []
    for record in community_results:
        community_id, nodes = record["communityId"], record["nodes"]
        logging.info(f"Nodes for community {community_id}: {json.dumps(nodes)}")
        community_text = " ".join([str(node["properties"]) for node in nodes])
        summary_prompt = f'Summarize the following collection of related entities in one sentence:\n{community_text}'
        summary = llm.invoke(summary_prompt)
        summaries.append({"community_id": community_id, "summary": summary})
    data["summaries"] = summaries
    return data

def store_summaries(data):
    for summary_data in data["summaries"]:
        memgraph_graph.query("MATCH (c:Community {id: $community_id}) SET c.summary = $summary", params=summary_data)
    return data

def cleanup_redis(data):
    batch_id = data["batch_id"]
    redis_client.delete(f"batch:{batch_id}:results", f"batch:{batch_id}:counter")
    return data

def migrate_to_spanner(data):
    logging.info("Migrating data to Spanner...")

    # Extract data from Memgraph
    entities_memgraph = memgraph_graph.query("MATCH (e:Entity) RETURN e.id AS EntityId, e.type AS Type, e.properties AS Properties")
    relationships_memgraph = memgraph_graph.query("MATCH (s)-[r]->(t) RETURN s.id AS SourceEntityId, t.id AS TargetEntityId, r.type AS Type, r.properties AS Properties")
    communities_memgraph = memgraph_graph.query("MATCH (c:Community) RETURN c.id AS CommunityId, c.summary AS Summary, c.embedding AS Embedding")
    entity_community_memgraph = memgraph_graph.query("MATCH (e:Entity)-[:BELONGS_TO]->(c:Community) RETURN e.id AS EntityId, c.id AS CommunityId")

    # Prepare data for Spanner
    entities_to_insert = []
    for entity in entities_memgraph:
        entities_to_insert.append((entity["EntityId"], entity["Type"], json.dumps(entity["Properties"])))

    relationships_to_insert = []
    for rel in relationships_memgraph:
        relationships_to_insert.append((rel["SourceEntityId"], rel["TargetEntityId"], rel["Type"], json.dumps(rel.get("Properties", {}))))

    communities_to_insert = []
    for community in communities_memgraph:
        communities_to_insert.append((community["CommunityId"], community["Summary"], community.get("Embedding")))

    entity_community_to_insert = []
    for ec in entity_community_memgraph:
        entity_community_to_insert.append((ec["EntityId"], ec["CommunityId"]))

    logging.info(f"Migrating {len(entities_to_insert)} entities, {len(relationships_to_insert)} relationships, {len(communities_to_insert)} communities, and {len(entity_community_to_insert)} entity-community relationships to Spanner.")


    # Only run transaction if there is data to insert
    if not any([entities_to_insert, relationships_to_insert, communities_to_insert, entity_community_to_insert]):
        logging.info("No new data to migrate to Spanner.")
        return data

    # Load data into Spanner
    def insert_data(transaction):
        if entities_to_insert:
            transaction.insert(
                table="Entities",
                columns=("EntityId", "Type", "Properties"),
                values=entities_to_insert,
            )
        if relationships_to_insert:
            transaction.insert(
                table="Relationships",
                columns=("SourceEntityId", "TargetEntityId", "Type", "Properties"),
                values=relationships_to_insert,
            )
        if communities_to_insert:
            transaction.insert(
                table="Communities",
                columns=("CommunityId", "Summary", "Embedding"),
                values=communities_to_insert,
            )
        if entity_community_to_insert:
            transaction.insert(
                table="EntityCommunity",
                columns=("EntityId", "CommunityId"),
                values=entity_community_to_insert,
            )

    spanner_database.run_in_transaction(insert_data)
    logging.info("Data migrated to Spanner successfully.")
    return data

def log_redis_counts(data):
    """Logs the number of partial results fetched from Redis."""
    partial_results_count = len(data.get("partial_results", []))
    logging.info(f"Fetched {partial_results_count} partial results from Redis.")
    return data

def log_memgraph_counts(data):
    """Logs the number of nodes and relationships in Memgraph."""
    try:
        node_count_result = memgraph_graph.query("MATCH (n) RETURN count(n) AS count")
        node_count = node_count_result[0]['count'] if node_count_result else 0
        
        rel_count_result = memgraph_graph.query("MATCH ()-[r]->() RETURN count(r) AS count")
        rel_count = rel_count_result[0]['count'] if rel_count_result else 0
        
        logging.info(f"Memgraph contains {node_count} nodes and {rel_count} relationships.")
    except Exception as e:
        logging.error(f"Error counting objects in Memgraph: {e}", exc_info=True)
    return data

def log_spanner_counts(data):
    """Queries and logs the row counts from Spanner tables."""
    logging.info("Querying Spanner for row counts...")
    tables_to_query = ["Entities", "Communities", "Relationships", "EntityCommunity"]
    
    try:
        for table in tables_to_query:
            with spanner_database.snapshot() as snapshot:
                try:
                    results = snapshot.execute_sql(f"SELECT COUNT(*) FROM {table}")
                    for row in results:
                        logging.info(f"Spanner table '{table}' contains {row[0]} rows.")
                except Exception as e:
                    logging.error(f"Error querying row count for table {table}: {e}")
    except Exception as e:
        logging.error(f"Error creating Spanner snapshot: {e}", exc_info=True)
    return data

def cleanup_memgraph(data):
    """Deletes all nodes and relationships from Memgraph."""
    logging.info("Cleaning up Memgraph...")
    try:
        memgraph_graph.query("MATCH (n) DETACH DELETE n")
        logging.info("Memgraph cleaned up successfully.")
    except Exception as e:
        logging.error(f"Error cleaning up Memgraph: {e}", exc_info=True)
    return data

# --- LangChain Sequence ---
consolidation_chain = RunnableSequence(
    decode_pubsub_message,
    fetch_from_redis,
    log_redis_counts,
    aggregate_results,
    load_to_memgraph,
    log_memgraph_counts,
    run_community_detection,
    generate_summaries,
    store_summaries,
    migrate_to_spanner,
    log_spanner_counts,
    cleanup_redis,
    cleanup_memgraph
)

# --- Main Function ---
@functions_framework.cloud_event
def consolidator(cloud_event):
    try:
        initialize_clients()  # Initialize clients on each invocation
        consolidation_chain.invoke(cloud_event)
        return "OK", 200
    except Exception as e:
        logging.error(f'An error occurred in the consolidator: {e}', exc_info=True)
        return "Internal Server Error", 500
