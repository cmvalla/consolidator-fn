import base64
import json
import os
import functions_framework
import google.cloud.logging
import logging
import redis
import pymemgpt
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextGenerationModel
import google.cloud.secretmanager as secretmanager

# --- Boilerplate and Configuration ---

# Setup structured logging
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
embedding_model = None
generation_model = None
memgraph_client = None

try:
    logging.info("Initializing global clients...")
    
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, ssl=False, ssl_cert_reqs=None, decode_responses=True, socket_connect_timeout=10)
    redis_client.ping()

    # Initialize Memgraph client
    memgraph_client = pymemgpt.connect(host=MEMGRAPH_HOST, port=MEMGRAPH_PORT)
    memgraph_client.execute("CREATE CONSTRAINT ON (n:Entity) ASSERT n.id IS UNIQUE;")
    memgraph_client.execute("CREATE CONSTRAINT ON (n:Community) ASSERT n.id IS UNIQUE;")


    vertexai.init(project=GCP_PROJECT, location="us-central1")
    embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
    generation_model = TextGenerationModel.from_pretrained("text-bison@001")
    logging.info("All global clients initialized successfully.")
except Exception as e:
    logging.critical(f'FATAL: Failed to initialize one or more global clients: {e}', exc_info=True)


@functions_framework.cloud_event
def consolidator(cloud_event):
    """
    Triggered by a Pub/Sub message indicating a batch is ready for consolidation.
    Fetches partial results from Redis, consolidates them, runs graph analytics,
    generates summaries, and persists the final graph to Memgraph.
    """
    if not all([memgraph_client, redis_client, embedding_model, generation_model]):
        logging.critical("Global clients not initialized. Aborting function.")
        return "ERROR: Client initialization failed", 500

    batch_id = None
    try:
        # 1. Decode the Pub/Sub trigger message to get the batch_id
        message_data = base64.b64decode(cloud_event.data["message"]["data"]).decode("utf-8")
        message_json = json.loads(message_data)
        batch_id = message_json.get("batch_id")

        if not batch_id:
            logging.error("Pub/Sub message is missing 'batch_id'.")
            return "Bad Request: Missing batch_id", 400

        logging.info(f"Starting consolidation for batch_id: {batch_id}")

        # 2. Fetch all partial results from Redis
        results_key = f"batch:{batch_id}:results"
        
        num_results = redis_client.llen(results_key)
        if num_results == 0:
            logging.warning(f"No results found in Redis for batch '{batch_id}'. Aborting.")
            return "OK", 200
            
        partial_results_str = redis_client.lrange(results_key, 0, -1)
        logging.info(f"Fetched {len(partial_results_str)} partial results from Redis for batch '{batch_id}'.")

        # 3. Aggregate partial results into a single payload
        all_entities = {}
        all_relationships = []
        for res_str in partial_results_str:
            res_json = json.loads(res_str)
            for entity in res_json.get("entities", []):
                all_entities[entity["id"]] = entity
            all_relationships.extend(res_json.get("relationships", []))
        
        payload = {
            "entities": list(all_entities.values()),
            "relationships": all_relationships
        }
        logging.info(f"Aggregated into {len(payload['entities'])} unique entities and {len(payload['relationships'])} relationships.")

        # 4. Load data into Memgraph
        try:
            memgraph_client.execute("MATCH (n) DETACH DELETE n;")
            logging.info("Cleared existing graph data from Memgraph.")
        except Exception as e:
            logging.error(f'Error clearing Memgraph: {e}', exc_info=True)


        entities = payload.get("entities", [])
        if entities:
            node_query = "UNWIND $nodes AS node CREATE (:Entity {id: node.id, type: node.type, properties: apoc.convert.toJson(node.properties)})"
            memgraph_client.execute(node_query, {'nodes': entities})
            logging.info(f"Successfully created {len(entities)} nodes in Memgraph.")

        relationships = payload.get("relationships", [])
        if relationships:
            rel_query = "UNWIND $rels AS rel MATCH (a:Entity {id: rel.source}), (b:Entity {id: rel.target}) CREATE (a)-[:RELATIONSHIP {type: rel.type, properties: apoc.convert.toJson(rel.properties)}]->(b)"
            memgraph_client.execute(rel_query, {'rels': relationships})
            logging.info(f"Successfully created {len(relationships)} relationships.")

        # 5. Execute community detection
        logging.info("Executing Louvain community detection...")
        community_query = "CALL community_detection.get() YIELD node, community_id"
        result = memgraph_client.execute_and_fetch(community_query)
        
        # Create Community nodes and relationships
        for record in result:
            node_id = record["node"].properties["id"]
            community_id = record["community_id"]
            
            memgraph_client.execute("MERGE (c:Community {id: $community_id})", {"community_id": community_id})
            
            memgraph_client.execute("MATCH (e:Entity {id: $node_id}), (c:Community {id: $community_id}) CREATE (e)-[:BELONGS_TO]->(c)", {"node_id": node_id, "community_id": community_id})

        logging.info("Community detection and linking complete.")

        # 6. Generate summaries and embeddings for each community
        logging.info("Generating summaries and embeddings...")
        get_communities_query = "MATCH (c:Community)<-[:BELONGS_TO]-(e:Entity) RETURN c.id AS communityId, COLLECT({id: e.id, properties: e.properties}) AS nodes"
        community_results = memgraph_client.execute_and_fetch(get_communities_query)
        
        for record in community_results:
            community_id, nodes = record["communityId"], record["nodes"]
            community_text = " ".join([str(node) for node in nodes])
            summary_prompt = f'Summarize the following collection of related entities in one sentence:\n{community_text}'
            summary = generation_model.predict(summary_prompt, max_output_tokens=128).text
            embedding = embedding_model.get_embeddings([summary])[0].values
            
            memgraph_client.execute("MATCH (c:Community {id: $community_id}) SET c.summary = $summary, c.embedding = $embedding", {"community_id": community_id, "summary": summary, "embedding": embedding})

        logging.info("Generated and stored summaries for communities.")

        # 7. Clean up Redis keys for the completed batch
        redis_client.delete(f"batch:{batch_id}:results", f"batch:{batch_id}:counter")
        logging.info(f"Cleaned up Redis keys for batch_id: {batch_id}")

        return "OK", 200

    except Exception as e:
        logging.error(f'An error occurred in the consolidator for batch \'{batch_id}\': {e}', exc_info=True)
        return "Internal Server Error", 500