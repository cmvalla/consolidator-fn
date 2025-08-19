import base64
import json
import os
import functions_framework
import google.cloud.logging
import logging
import redis
from google.cloud import spanner
from vertexai.language_models import TextEmbeddingModel, TextGenerationModel

# --- Boilerplate and Configuration ---

# Setup structured logging
logging_client = google.cloud.logging.Client()
logging_client.setup_logging()

# --- Environment Variables ---
GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
SPANNER_INSTANCE_ID = os.environ.get("SPANNER_INSTANCE_ID")
SPANNER_DATABASE_ID = os.environ.get("SPANNER_DATABASE_ID")
REDIS_HOST = os.environ.get("REDIS_HOST")
REDIS_PORT = os.environ.get("REDIS_PORT", 6379)
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD")

# --- Global Clients ---
# It's best practice to initialize clients outside of the function body
# to reuse connections and reduce cold start times.
try:
    spanner_client = spanner.Client(project=GCP_PROJECT)
    spanner_instance = spanner_client.instance(SPANNER_INSTANCE_ID)
    spanner_database = spanner_instance.database(SPANNER_DATABASE_ID)
    
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True)
    redis_graph = redis_client.graph("graphrag_graph") # Working graph in Redis

    vertexai.init(project=GCP_PROJECT, location="us-central1")
    embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
    generation_model = TextGenerationModel.from_pretrained("text-bison@001")

except Exception as e:
    logging.error(f"Failed to initialize global clients: {e}")
    # Handle initialization failure, maybe by exiting or setting a flag
    # For a Cloud Function, a failure here might prevent it from starting correctly
    spanner_client = None
    redis_client = None
    embedding_model = None
    generation_model = None


@functions_framework.cloud_event
def consolidator(cloud_event):
    """
    This function is triggered by a Pub/Sub message via Eventarc.
    It performs the consolidation logic using RedisGraph and Spanner.
    
    Expected Pub/Sub message payload:
    {
        "entities": [{"id": "node1", "type": "person", "properties": {"name": "Alice"}}],
        "relationships": [{"source": "node1", "target": "node2", "type": "knows"}]
    }
    """
    if not all([spanner_client, redis_client, embedding_model, generation_model]):
        logging.critical("Global clients not initialized. Aborting function.")
        return "ERROR: Client initialization failed", 500

    try:
        # 1. Decode the Pub/Sub message
        message_data = base64.b64decode(cloud_event.data["message"]["data"]).decode("utf-8")
        payload = json.loads(message_data)
        logging.info(f"Received payload with {len(payload.get('entities', []))} entities and {len(payload.get('relationships', []))} relationships.")

        # 2. Load data into RedisGraph using parameterized UNWIND queries
        # This is much more efficient and safer than sending one query per element.
        try:
            redis_graph.delete()
            logging.info("Cleared existing graph data from 'graphrag_graph'.")
        except redis.exceptions.ResponseError as e:
            # Graph might not exist on first run, which is okay.
            if "Graph not found" in str(e):
                pass 
            else:
                raise e

        entities = payload.get("entities", [])
        if entities:
            # Create all nodes in a single, parameterized query
            node_query = """
            UNWIND $nodes AS node
            CREATE (n:Entity {id: node.id, type: node.type, properties: apoc.convert.toJson(node.properties)})
            """
            # Note: The properties are stored as a JSON string.
            # RedisGraph itself doesn't have a native JSON type, so this is a common pattern.
            # We might need to install the 'apoc' library in RedisGraph for this to work.
            # A simpler alternative is to flatten properties into top-level attributes.
            redis_graph.query(node_query, {'nodes': entities})
            logging.info(f"Successfully created {len(entities)} nodes in RedisGraph.")

        relationships = payload.get("relationships", [])
        if relationships:
            # Create all relationships in a single, parameterized query
            rel_query = """
            UNWIND $rels AS rel
            MATCH (a:Entity {id: rel.source}), (b:Entity {id: rel.target})
            CREATE (a)-[r:RELATIONSHIP {type: rel.type, properties: apoc.convert.toJson(rel.properties)}]->(b)
            """
            redis_graph.query(rel_query, {'rels': relationships})
            logging.info(f"Successfully created {len(relationships)} relationships in RedisGraph.")

        # 3. Execute community detection in RedisGraph
        # This uses the GDS library (part of redislabs/redisgraph image)
        # to find communities and write the community ID back to each node.
        logging.info("Executing Louvain community detection...")
        community_query = """
        CALL gds.louvain.write({
            nodeProjection: 'Entity',
            relationshipProjection: 'RELATIONSHIP',
            writeProperty: 'community_id'
        })
        YIELD communityCount, modularity
        """
        try:
            result = redis_graph.query(community_query)
            community_count = result.result_set[0][0]
            modularity = result.result_set[0][1]
            logging.info(f"Community detection completed. Found {community_count} communities with modularity {modularity:.4f}.")
        except redis.exceptions.ResponseError as e:
            logging.error(f"Error executing community detection: {e}. Ensure the GDS plugin is loaded.")
            # Handle the error, maybe the graph is too small or has no relationships
            raise e


        # 4. For each community, generate a summary and an embedding
        logging.info("Generating summaries and embeddings for each community...")
        
        # First, get all nodes and their community IDs from the graph
        get_communities_query = """
        MATCH (n:Entity) 
        WHERE n.community_id IS NOT NULL
        RETURN n.community_id AS communityId, COLLECT({id: n.id, properties: n.properties}) AS nodes
        """
        community_results = redis_graph.query(get_communities_query)
        
        communities_to_persist = []
        for record in community_results.result_set:
            community_id = record[0]
            nodes_in_community = record[1]

            # Create a single text block describing the community
            # In a real-world scenario, you would be more selective about the properties.
            community_text = " ".join([str(node) for node in nodes_in_community])

            # Generate a summary with the text model
            summary_prompt = f"Summarize the following collection of related entities in one sentence:\n{community_text}"
            summary_response = generation_model.predict(summary_prompt, max_output_tokens=128)
            summary = summary_response.text

            # Generate an embedding for the summary
            embedding_response = embedding_model.get_embeddings([summary])
            embedding = embedding_response[0].values

            communities_to_persist.append({
                "community_id": str(community_id),
                "summary": summary,
                "summary_embedding": embedding,
                "properties": json.dumps({"node_count": len(nodes_in_community)})
            })

        logging.info(f"Successfully generated summaries and embeddings for {len(communities_to_persist)} communities.")


        # 5. Read the enriched graph from RedisGraph to prepare for Spanner insertion
        logging.info("Reading enriched graph from RedisGraph...")
        
        # Get all nodes with their new community_id
        get_nodes_query = "MATCH (n:Entity) RETURN n.id, n.type, n.properties, n.community_id"
        node_results = redis_graph.query(get_nodes_query)
        entities_to_persist = [
            (record[0], record[1], record[2], str(record[3])) for record in node_results.result_set
        ]

        # Get all relationships
        get_rels_query = "MATCH (a:Entity)-[r:RELATIONSHIP]->(b:Entity) RETURN r.id, a.id, b.id, r.type, r.properties"
        rel_results = redis_graph.query(get_rels_query)
        relationships_to_persist = [
            (record[0] or str(uuid.uuid4()), record[1], record[2], record[3], record[4]) for record in rel_results.result_set
        ]
        logging.info(f"Extracted {len(entities_to_persist)} entities and {len(relationships_to_persist)} relationships to persist.")


        # 6. Write the final graph to Spanner (our System of Record)
        def insert_data(transaction):
            if communities_to_persist:
                transaction.insert(
                    "Communities",
                    columns=("community_id", "summary", "summary_embedding", "properties"),
                    values=[(c["community_id"], c["summary"], c["summary_embedding"], c["properties"]) for c in communities_to_persist]
                )
            if entities_to_persist:
                transaction.insert(
                    "Entities",
                    columns=("entity_id", "type", "properties", "community_id"),
                    values=entities_to_persist
                )
            if relationships_to_persist:
                transaction.insert(
                    "Relationships",
                    columns=("relationship_id", "source_entity_id", "target_entity_id", "type", "properties"),
                    values=relationships_to_persist
                )
            logging.info(f"Persisted {len(communities_to_persist)} communities, {len(entities_to_persist)} entities, and {len(relationships_to_persist)} relationships to Spanner.")

        spanner_database.run_in_transaction(insert_data)
        logging.info("Successfully persisted all data to Spanner.")


        return "OK", 200

    except Exception as e:
        logging.error(f"An error occurred in the consolidator function: {e}", exc_info=True)
        # Depending on the error, you might want to retry the function
        # For now, we return a generic error
        return "Internal Server Error", 500
