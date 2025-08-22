import base64
import json
import os
import functions_framework
import google.cloud.logging
import logging
import redis
from google.cloud import spanner
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextGenerationModel
import uuid

# --- Boilerplate and Configuration ---

# Setup structured logging
logging_client = google.cloud.logging.Client()
logging_client.setup_logging()
logging.basicConfig(level=logging.INFO)

# --- Environment Variables ---
GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
SPANNER_INSTANCE_ID = os.environ.get("SPANNER_INSTANCE_ID")
SPANNER_DATABASE_ID = os.environ.get("SPANNER_DATABASE_ID")
REDIS_HOST = os.environ.get("REDIS_HOST")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD")

# --- Global Clients ---
spanner_client = None
redis_client = None
embedding_model = None
generation_model = None
redis_graph = None

try:
    logging.info("Initializing global clients...")
    spanner_client = spanner.Client(project=GCP_PROJECT)
    spanner_instance = spanner_client.instance(SPANNER_INSTANCE_ID)
    spanner_database = spanner_instance.database(SPANNER_DATABASE_ID)
    
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, ssl=True, ssl_cert_reqs=None, decode_responses=True)
    redis_client.ping()
    redis_graph = redis_client.graph("graphrag_graph")

    vertexai.init(project=GCP_PROJECT, location="us-central1")
    embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
    generation_model = TextGenerationModel.from_pretrained("text-bison@001")
    logging.info("All global clients initialized successfully.")
except Exception as e:
    logging.critical(f"FATAL: Failed to initialize one or more global clients: {e}", exc_info=True)


@functions_framework.cloud_event
def consolidator(cloud_event):
    """
    Triggered by a Pub/Sub message indicating a batch is ready for consolidation.
    Fetches partial results from Redis, consolidates them, runs graph analytics,
    generates summaries, and persists the final graph to Spanner.
    """
    if not all([spanner_client, redis_client, redis_graph, embedding_model, generation_model]):
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
        
        # Llen gets the length, lrange gets all elements.
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
                all_entities[entity["id"]] = entity # Use a dict to auto-deduplicate entities
            all_relationships.extend(res_json.get("relationships", []))
        
        payload = {
            "entities": list(all_entities.values()),
            "relationships": all_relationships
        }
        logging.info(f"Aggregated into {len(payload['entities'])} unique entities and {len(payload['relationships'])} relationships.")

        # 4. Load data into RedisGraph
        try:
            redis_graph.delete()
            logging.info("Cleared existing graph data from 'graphrag_graph'.")
        except redis.exceptions.ResponseError as e:
            if "Graph not found" not in str(e):
                raise

        entities = payload.get("entities", [])
        if entities:
            node_query = "UNWIND $nodes AS node CREATE (:Entity {id: node.id, type: node.type, properties: apoc.convert.toJson(node.properties)})"
            redis_graph.query(node_query, {'nodes': entities})
            logging.info(f"Successfully created {len(entities)} nodes in RedisGraph.")

        relationships = payload.get("relationships", [])
        if relationships:
            rel_query = "UNWIND $rels AS rel MATCH (a:Entity {id: rel.source}), (b:Entity {id: rel.target}) CREATE (a)-[:RELATIONSHIP {type: rel.type, properties: apoc.convert.toJson(rel.properties)}]->(b)"
            redis_graph.query(rel_query, {'rels': relationships})
            logging.info(f"Successfully created {len(relationships)} relationships.")

        # 5. Execute community detection
        logging.info("Executing Louvain community detection...")
        community_query = "CALL gds.louvain.write({nodeProjection: 'Entity', relationshipProjection: 'RELATIONSHIP', writeProperty: 'community_id'}) YIELD communityCount, modularity"
        result = redis_graph.query(community_query)
        community_count, modularity = result.result_set[0]
        logging.info(f"Community detection found {community_count} communities with modularity {modularity:.4f}.")

        # 6. Generate summaries and embeddings for each community
        logging.info("Generating summaries and embeddings...")
        get_communities_query = "MATCH (n:Entity) WHERE n.community_id IS NOT NULL RETURN n.community_id AS communityId, COLLECT({id: n.id, properties: n.properties}) AS nodes"
        community_results = redis_graph.query(get_communities_query)
        
        communities_to_persist = []
        for record in community_results.result_set:
            community_id, nodes = record[0], record[1]
            community_text = " ".join([str(node) for node in nodes])
            summary_prompt = f"Summarize the following collection of related entities in one sentence:\n{community_text}"
            summary = generation_model.predict(summary_prompt, max_output_tokens=128).text
            embedding = embedding_model.get_embeddings([summary])[0].values
            communities_to_persist.append({
                "community_id": str(community_id), "summary": summary, "summary_embedding": embedding,
                "properties": json.dumps({"node_count": len(nodes)})
            })
        logging.info(f"Generated summaries for {len(communities_to_persist)} communities.")

        # 7. Extract full graph data for Spanner
        get_nodes_query = "MATCH (n:Entity) RETURN n.id, n.type, n.properties, n.community_id"
        entities_to_persist = [(r[0], r[1], r[2], str(r[3])) for r in redis_graph.query(get_nodes_query).result_set]
        
        get_rels_query = "MATCH (a:Entity)-[r:RELATIONSHIP]->(b:Entity) RETURN a.id, b.id, r.type, r.properties"
        relationships_to_persist = [(str(uuid.uuid4()), r[0], r[1], r[2], r[3]) for r in redis_graph.query(get_rels_query).result_set]
        logging.info(f"Extracted {len(entities_to_persist)} entities and {len(relationships_to_persist)} relationships for Spanner.")

        # 8. Write to Spanner
        def insert_data(transaction):
            if communities_to_persist:
                transaction.insert("Communities", columns=("community_id", "summary", "summary_embedding", "properties"),
                                   values=[(c["community_id"], c["summary"], c["summary_embedding"], c["properties"]) for c in communities_to_persist])
            if entities_to_persist:
                transaction.insert("Entities", columns=("entity_id", "type", "properties", "community_id"), values=entities_to_persist)
            if relationships_to_persist:
                transaction.insert("Relationships", columns=("relationship_id", "source_entity_id", "target_entity_id", "type", "properties"), values=relationships_to_persist)
        
        spanner_database.run_in_transaction(insert_data)
        logging.info("Successfully persisted all data to Spanner.")

        # 9. Clean up Redis keys for the completed batch
        redis_client.delete(f"batch:{batch_id}:results", f"batch:{batch_id}:counter")
        logging.info(f"Cleaned up Redis keys for batch_id: {batch_id}")

        return "OK", 200

    except Exception as e:
        logging.error(f"An error occurred in the consolidator for batch '{batch_id}': {e}", exc_info=True)
        return "Internal Server Error", 500
