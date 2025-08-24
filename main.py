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

# --- Boilerplate and Configuration ---
logging_client = google.cloud.logging.Client()
logging_client.setup_logging()
logging.basicConfig(level=logging.INFO)

# --- Environment Variables ---
GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
REDIS_HOST = os.environ.get("REDIS_HOST")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_PASSWORD = secretmanager.SecretManagerServiceClient().access_secret_version(request={"name": f"projects/{GCP_PROJECT}/secrets/redis-password/versions/latest"}).payload.data.decode("UTF-8")
MEMGRAPH_HOST = os.environ.get("MEMGRAPH_HOST", "memgraph-service.memgraph.svc.cluster.local")
MEMGRAPH_PORT = int(os.environ.get("MEMGRAPH_PORT", 7687))

# --- Global Clients ---
redis_client = None
llm = None
memgraph_graph = None

try:
    logging.info("Initializing global clients...")
    
    logging.info("Initializing Redis client...")
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, ssl=False, ssl_cert_reqs=None, decode_responses=True, socket_connect_timeout=10)
    redis_client.ping()
    logging.info("Redis client initialized successfully.")

    logging.info("Initializing Memgraph client...")
    memgraph_graph = MemgraphGraph(url=f"bolt://{MEMGRAPH_HOST}:{MEMGRAPH_PORT}", username="", password="")
    logging.info("Memgraph client initialized successfully.")

    logging.info("Initializing Vertex AI...")
    llm = VertexAI(model_name="text-bison@001")
    logging.info("Vertex AI clients initialized successfully.")

    logging.info("All global clients initialized successfully.")
except Exception as e:
    logging.critical(f'FATAL: Failed to initialize one or more global clients: {e}', exc_info=True)

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
    return {
        "batch_id": data["batch_id"],
        "entities": list(all_entities.values()),
        "relationships": all_relationships
    }

def load_to_memgraph(data):
    memgraph_graph.query("MATCH (n) DETACH DELETE n")
    entities = data.get("entities", [])
    if entities:
        node_query = "UNWIND $nodes AS node CREATE (:Entity {id: node.id, type: node.type, properties: apoc.convert.toJson(node.properties)})"
        memgraph_graph.query(node_query, params={'nodes': entities})
    relationships = data.get("relationships", [])
    if relationships:
        rel_query = "UNWIND $rels AS rel MATCH (a:Entity {id: rel.source}), (b:Entity {id: rel.target}) CREATE (a)-[:RELATIONSHIP {type: rel.type, properties: apoc.convert.toJson(rel.properties)}]->(b)"
        memgraph_graph.query(rel_query, params={'rels': relationships})
    return data

def run_community_detection(data):
    community_query = "CALL community_detection.get() YIELD node, community_id"
    result = memgraph_graph.query(community_query)
    for record in result:
        node_id = record["node"].properties["id"]
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
        community_text = " ".join([str(node) for node in nodes])
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

# --- LangChain Sequence ---
consolidation_chain = RunnableSequence(
    decode_pubsub_message,
    fetch_from_redis,
    aggregate_results,
    load_to_memgraph,
    run_community_detection,
    generate_summaries,
    store_summaries,
    cleanup_redis
)

# --- Main Function ---
@functions_framework.cloud_event
def consolidator(cloud_event):
    try:
        consolidation_chain.invoke(cloud_event)
        return "OK", 200
    except Exception as e:
        logging.error(f'An error occurred in the consolidator: {e}', exc_info=True)
        return "Internal Server Error", 500
