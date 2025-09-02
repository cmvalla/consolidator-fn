import base64
import json
import re
import os
import functions_framework
import google.cloud.logging
import logging
import redis
import igraph as ig
import hashlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from langchain_google_vertexai import VertexAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
import google.cloud.secretmanager as secretmanager
import time
import google.cloud.spanner_v1 as spanner
import psutil
from google.api_core.exceptions import AlreadyExists, FailedPrecondition
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

EMBEDDING_DIMENSION = 384



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

CLUSTER_SUMMARY_PROMPT = PromptTemplate.from_template(
    "Summarize the following collection of entities into a single, coherent paragraph. "
    "The summary should capture the main theme and characteristics of the cluster.\n\n"
    "Entities:\n{cluster_text}\n\nSummary:")

SUMMARY_PROMPT = PromptTemplate.from_template(
    "Summarize the following text in one concise sentence:\n\n"
    "TEXT:\n---\n{text_chunk}\n---\n\nSummary:")

CLASS_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "A concise and descriptive name for the class."},
        "description": {"type": "string", "description": "A paragraph that summarizes the common theme and purpose of the instances."},
        "properties": {
            "type": "object",
            "description": "An object representing the common schema of the instances. Keys should reflect common properties, and values should be a representative value or type (e.g., 'string', 'integer', 'boolean')."
        }
    },
    "required": ["name", "description", "properties"]
}

CLASS_PROPERTY_GENERATION_PROMPT = PromptTemplate(
    template="You are a knowledge graph expert. You are tasked with creating a representative 'Class' entity from a collection of 'Instance' entities. "
    "Based on the following instances, and the source text they were extracted from, generate a JSON object for the 'Class' that adheres to the following JSON schema:\n\n"
    "```json\n{schema}\n```\n\n"
    "Instances (as JSON objects):\n{instances_text}\n\n"
    "Source Text (for context):\n{source_text}\n\n"
    "Respond with a single, valid JSON object for the 'Class' entity.",
    input_variables=["instances_text", "schema", "source_text"]
)

def ensure_spanner_graph_exists(database):
    """
    This function is now a placeholder. The schema is managed by Terraform.
    The DDL definition can be found in `terraform/spanner.tf`.
    """
    logging.info("Schema management is now handled by Terraform. Skipping DDL updates from the function.")
    pass

def initialize_clients():
    """
    Initializes all external clients.
    """
    global redis_client, llm, spanner_client, spanner_instance, spanner_database, embedding_model

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
        
        SPANNER_INSTANCE_ID = os.environ.get("SPANNER_INSTANCE_ID")
        SPANNER_DATABASE_ID = os.environ.get("SPANNER_DATABASE_ID")
        logging.info(f"SPANNER_INSTANCE_ID from env: {SPANNER_INSTANCE_ID}")
        logging.info(f"SPANNER_DATABASE_ID from env: {SPANNER_DATABASE_ID}")
        logging.info(f"GOOGLE_APPLICATION_CREDENTIALS: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")
        LOCATION = os.environ.get("LOCATION")
        
        # --- Secret Manager ---
        sm_client = secretmanager.SecretManagerServiceClient()
        try:
            # Get the service account email from the metadata server
            response = requests.get("http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email", headers={"Metadata-Flavor": "Google"})
            service_account_email = response.text
            logging.info(f"Running as service account: {service_account_email}")
        except requests.exceptions.RequestException as e:
            logging.warning(f"Could not retrieve service account email: {e}")
        REDIS_PASSWORD = sm_client.access_secret_version(request={"name": f"projects/{GCP_PROJECT}/secrets/redis-password/versions/latest"}).payload.data.decode("UTF-8")

        logging.info("Initializing Redis client...")
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, ssl=False, ssl_cert_reqs=None, decode_responses=True, socket_connect_timeout=10)
        redis_client.ping()
        logging.info("Redis client initialized successfully.")

        

        logging.info("Initializing Vertex AI...")
        llm = VertexAI(model_name="gemini-2.5-flash", location=LOCATION, response_mime_type="application/json")
        logging.info("Vertex AI clients initialized successfully.")

        logging.info("Initializing Spanner client...")
        spanner_client = spanner.Client(project=GCP_PROJECT)
        spanner_instance = spanner_client.instance(SPANNER_INSTANCE_ID)
        spanner_database = spanner_instance.database(SPANNER_DATABASE_ID)
        logging.info(f"Spanner client configured for project: {spanner_client.project}, instance: {spanner_instance.instance_id}, database: {spanner_database.database_id}")
        logging.info("Spanner client initialized successfully.")

        

        logging.info("All global clients initialized successfully.")
    except Exception as e:
        logging.critical(f'FATAL: Failed to initialize one or more global clients: {e}', exc_info=True)
        raise  # Re-raise the exception to halt execution if initialization fails

# --- LangChain Runnables ---
def extract_json_from_llm_response(text):
    """
    Extracts a JSON object from the model's text response and performs basic validation.
    Handles markdown code blocks.
    """
    # Try to find a JSON block enclosed in ```json ... ```
    match = re.search(r"```json\s*({.*})```", text, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(1).strip()
    else:
        # If not found, try to find a JSON block enclosed in ``` ... ``` (without 'json')
        match = re.search(r"```\s*({.*})```", text, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1).strip()
        else:
            # Fallback: assume the entire text is JSON, but strip common wrappers
            json_str = text.strip()
            # Remove common prefixes/suffixes that are not valid JSON
            if json_str.startswith("json"):
                json_str = json_str[4:].strip()
            if json_str.startswith("```"):
                json_str = json_str[3:].strip()
            if json_str.endswith("```"):
                json_str = json_str[:-3].strip()

    # Remove "insensitive:true" from the string (if present from previous LLM issues)
    json_str = json_str.replace("insensitive:true", "")

    return json_str

def decode_pubsub_message(cloud_event):
    logging.info(f"Received cloud_event: {cloud_event}")

    # Handle manual trigger for testing
    if isinstance(cloud_event, dict) and "batch_id" in cloud_event and "message" not in cloud_event:
        return {"batch_id": cloud_event["batch_id"]}

    # Handle both dict and CloudEvent objects from Pub/Sub
    event_data = cloud_event.get("data") if isinstance(cloud_event, dict) else cloud_event.data

    message_data = base64.b64decode(event_data["message"]["data"]).decode("utf-8")
    message_json = json.loads(message_data)
    return {"batch_id": message_json.get("batch_id")}

def fetch_from_redis(data):
    batch_id = data["batch_id"]
    results_key = f"batch:{batch_id}:results"
    partial_results_str = redis_client.lrange(results_key, 0, -1)
    logging.info(f"Fetched {len(partial_results_str)} partial results from Redis for batch {batch_id}.")
    if not partial_results_str:
        logging.warning(f"No partial results found in Redis for batch {batch_id}. Key: {results_key}")
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

def get_embedding(text: str, entity_id: str = "Unknown"):
    """Generates an embedding for a given text, with retry logic."""
    embedding_service_url = os.environ.get("EMBEDDING_SERVICE_URL")
    if not embedding_service_url:
        logging.error("EMBEDDING_SERVICE_URL environment variable not set.")
        return [0.0] * EMBEDDING_DIMENSION

    MAX_RETRIES = 10
    INITIAL_BACKOFF_SECONDS = 1
    MAX_BACKOFF_SECONDS = 600  # 10 minutes
    
    retries = 0
    backoff_seconds = INITIAL_BACKOFF_SECONDS

    while retries < MAX_RETRIES:
        try:
            # Fetch ID token for the embedding service
            token_url = f"http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity?audience={embedding_service_url}"
            token_response = requests.get(token_url, headers={"Metadata-Flavor": "Google"})
            token = token_response.text
            headers = {"Authorization": f"Bearer {token}"}
            
            response = requests.post(embedding_service_url, json={"text": text}, headers=headers)
            
            if response.status_code == 200:
                embedding = response.json().get("embedding")
                if embedding:
                    return embedding
                else:
                    logging.warning(f"Embedding not found in response for entity: {entity_id}")
                    return [0.0] * EMBEDDING_DIMENSION
            
            elif response.status_code >= 500:
                logging.warning(f"Embedding service returned a server error ({response.status_code}) for entity {entity_id}. Retrying in {backoff_seconds} seconds...")
                time.sleep(backoff_seconds)
                retries += 1
                backoff_seconds = min(backoff_seconds * 2, MAX_BACKOFF_SECONDS)
            
            else:
                logging.error(f"Embedding service returned a client error ({response.status_code}) for entity {entity_id}: {response.text}")
                response.raise_for_status()
                return [0.0] * EMBEDDING_DIMENSION

        except requests.exceptions.RequestException as e:
            logging.error(f"Error calling embedding service for entity {entity_id}: {e}")
            time.sleep(backoff_seconds)
            retries += 1
            backoff_seconds = min(backoff_seconds * 2, MAX_BACKOFF_SECONDS)

    logging.error(f"Failed to get embedding for entity {entity_id} after {MAX_RETRIES} retries.")
    return [0.0] * EMBEDDING_DIMENSION

def generate_embeddings(data):
    """Generates embeddings for all entities and communities."""
    entities = data.get("entities", [])
    
    # Create a summarization chain
    summarization_chain = LLMChain(llm=llm, prompt=SUMMARY_PROMPT)

    for entity in entities:
        entity_type = entity.get('type', '')
        properties = entity.get('properties', {})
        
        if entity_type == 'Chunk':
            text_to_embed = properties.get('summary', '')
            if not text_to_embed:
                logging.warning(f"Chunk {entity.get('id')} has an empty summary. Generating a new one.")
                original_text = properties.get('original_text', '')
                if original_text:
                    summary = summarization_chain.invoke({"text_chunk": original_text}).get("text")
                    properties['summary'] = summary
                    text_to_embed = summary
                else:
                    logging.warning(f"Chunk {entity.get('id')} also has no original_text to generate a summary from.")
        elif entity_type == 'Community':
            text_to_embed = properties.get('summary', '')
        else:
            text_to_embed = f"Type: {entity_type}, Properties: {json.dumps(properties)}"

        if not text_to_embed:
            logging.warning(f"Skipping embedding for entity {entity.get('id')} because there is no text to embed.")
            entity['embedding'] = [0.0] * EMBEDDING_DIMENSION
            continue

        entity['embedding'] = get_embedding(text_to_embed, entity.get('id'))

    return data

def generate_class_properties(class_property_chain, instances_text, schema, source_text):
    """Helper function to run LLM chain in a thread, with logging."""
    prompt = CLASS_PROPERTY_GENERATION_PROMPT.format(instances_text=instances_text, schema=schema, source_text=source_text)
    logging.info(f"Querying LLM for class properties. Prompt length: {len(prompt)}. Prompt:\n{prompt}")
    response = class_property_chain.invoke({"instances_text": instances_text, "schema": schema, "source_text": source_text}).get("text")
    logging.info(f"LLM response for class properties: {response}")
    return response

def cluster_and_merge_entities(data, similarity_threshold=0.9):
    """
    Clusters similar entities based on embedding similarity, creates Class and Instance nodes,
    promotes relationships, and aggregates their weights.
    """
    entities = data.get("entities", [])
    relationships = data.get("relationships", [])

    if not entities:
        return data

    # --- Start of new logic ---
    # 1. Create a map from entity ID to the entity object for quick lookups.
    id_to_entity = {entity['id']: entity for entity in entities}

    # 2. Create a map from an entity ID to its source chunk's text.
    entity_id_to_source_text = {}
    for rel in relationships:
        if rel.get("type") == "ARE_PART_OF_CHUNK":
            entity_id = rel.get("source")
            chunk_id = rel.get("target")
            if entity_id and chunk_id:
                chunk_entity = id_to_entity.get(chunk_id)
                if chunk_entity and chunk_entity.get("type") == "Chunk":
                    original_text = chunk_entity.get("properties", {}).get("original_text")
                    if original_text:
                        entity_id_to_source_text[entity_id] = original_text
    # --- End of new logic ---

    # Extract embeddings and entity IDs, excluding Chunk and Community entities
    entity_ids = [e["id"] for e in entities if e.get("type") not in ["Chunk", "Community"]]
    embeddings = np.array([e.get("embedding") for e in entities if e.get("type") not in ["Chunk", "Community"]])

    # Filter out entities without embeddings
    valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None and len(emb) > 0]
    if len(valid_indices) < 2:
        logging.warning("Not enough entities with embeddings to perform clustering.")
        return data
        
    valid_embeddings = embeddings[valid_indices]
    valid_entity_ids = [entity_ids[i] for i in valid_indices]
    
    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(valid_embeddings)

    # Group entities based on similarity threshold
    visited = [False] * len(valid_entity_ids)
    clusters = []
    for i in range(len(valid_entity_ids)):
        if visited[i]:
            continue
        cluster = [i]
        visited[i] = True
        for j in range(i + 1, len(valid_entity_ids)):
            if not visited[j] and similarity_matrix[i][j] > similarity_threshold:
                cluster.append(j)
                visited[j] = True
        clusters.append(cluster)

    # Create entity_id_map
    entity_id_map = {}
    cluster_embeddings = {}
    
    # Create a summarization chain
    summarization_chain = LLMChain(llm=llm, prompt=CLUSTER_SUMMARY_PROMPT)

    for cluster in clusters:
        canonical_index = cluster[0]
        canonical_id = valid_entity_ids[canonical_index]
        
        cluster_member_ids = [valid_entity_ids[entity_index] for entity_index in cluster]
        
        # Generate a representative text for the cluster
        cluster_text_parts = []
        for member_id in cluster_member_ids:
            member_entity = id_to_entity.get(member_id)
            if member_entity:
                entity_type = member_entity.get('type', '')
                properties = member_entity.get('properties', {})
                cluster_text_parts.append(f"Type: {entity_type}, Properties: {json.dumps(properties)}")
        
        cluster_text = " ".join(cluster_text_parts)
        
        # Summarize the cluster text and generate embedding
        if cluster_text:
            summary = summarization_chain.invoke({"cluster_text": cluster_text}).get("text")
            embedding = get_embedding(summary, f"class_{canonical_id}")
        else:
            embedding = [0.0] * EMBEDDING_DIMENSION
            
        cluster_embeddings[canonical_id] = embedding

        for entity_id in cluster_member_ids:
            entity_id_map[entity_id] = canonical_id

    # Create a map from old entity ids to new class ids
    class_id_map = {}
    for entity_id, canonical_id in entity_id_map.items():
        class_id_map[entity_id] = f"class_{canonical_id}"

    new_entities = []
    new_relationships = []

    # Create a new LLM chain for class property generation
    class_property_chain = LLMChain(llm=llm, prompt=CLASS_PROPERTY_GENERATION_PROMPT)

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_canonical_id = {}
        for canonical_id, embedding in cluster_embeddings.items():
            cluster_member_ids = [entity_id for entity_id, c_id in entity_id_map.items() if c_id == canonical_id]
            
            instances_text_parts = []
            source_text_parts = set() # Use a set to store unique source texts
            for member_id in cluster_member_ids:
                member_entity = id_to_entity.get(member_id)
                if member_entity:
                    properties = member_entity.get('properties', {})
                    instances_text_parts.append(f"- {json.dumps(properties)}")
                    # Use the pre-built map to find the source text
                    source_text = entity_id_to_source_text.get(member_id)
                    if source_text:
                        source_text_parts.add(source_text)
            
            instances_text = "\n".join(instances_text_parts)
            source_text_context = "\n\n---\n\n".join(source_text_parts)
            
            schema_str = json.dumps(CLASS_SCHEMA, indent=2)
            future = executor.submit(generate_class_properties, class_property_chain, instances_text, schema_str, source_text_context)
            future_to_canonical_id[future] = (canonical_id, embedding)

        processed_clusters = 0
        total_clusters = len(cluster_embeddings)
        for future in as_completed(future_to_canonical_id):
            canonical_id, embedding = future_to_canonical_id[future]
            try:
                generated_properties_str = future.result()
                extracted_json_str = extract_json_from_llm_response(generated_properties_str)
                generated_properties = json.loads(extracted_json_str)
            except Exception as exc:
                logging.warning(f"Failed to generate properties for class {canonical_id}: {exc}")
                generated_properties = {"name": f"Class {canonical_id}", "description": ""}

            class_entity = {
                "id": f"class_{canonical_id}",
                "type": "Class",
                "properties": generated_properties,
                "embedding": embedding
            }
            new_entities.append(class_entity)
            
            processed_clusters += 1
            progress = (processed_clusters / total_clusters) * 100
            if processed_clusters % (total_clusters // 10 or 1) == 0:
                logging.info(f"Class property generation progress: {progress:.0f}% ({processed_clusters}/{total_clusters})")


    # Create instance nodes and INSTANCE_OF relationships
    for entity in entities:
        # Keep original entities, but mark them as instances
        if entity.get("type") not in ["Chunk", "Community"]:
            entity["type"] = "Instance"
        new_entities.append(entity)
        
        class_id = class_id_map.get(entity["id"])
        if class_id:
            instance_of_rel = {
                "source": entity["id"],
                "target": class_id,
                "type": "INSTANCE_OF",
                "properties": {"description": "Indicates that an entity is an instance of a specific class."}
            }
            new_relationships.append(instance_of_rel)

    # Update relationships to point to class nodes
    for rel in relationships:
        source_class_id = class_id_map.get(rel.get("source"))
        target_class_id = class_id_map.get(rel.get("target"))
        
        # Promote relationship to class level if both source and target are in the same cluster
        if source_class_id and target_class_id and source_class_id == target_class_id:
            # This is an internal relationship within a class, we can either drop it or handle it.
            # For now, we drop it to avoid self-loops on the class node.
            continue
        elif source_class_id and target_class_id:
            # This is a relationship between two different classes
            rel["source"] = source_class_id
            rel["target"] = target_class_id
            new_relationships.append(rel)
        # Keep original relationships between instances if they are not promoted
        # else:
        #     new_relationships.append(rel)

    data["entities"] = new_entities
    data["relationships"] = new_relationships
    
    logging.info(f"Clustering complete. Result: {len(new_entities)} entities, {len(new_relationships)} relationships.")
    return data


def run_igraph_community_detection(data):
    logging.info("Running igraph community detection...")
    entities = data.get("entities", [])
    relationships = data.get("relationships", [])

    # Create a mapping from entity ID to igraph vertex ID
    id_to_vertex = {entity["id"]: i for i, entity in enumerate(entities)}
    
    # Create igraph graph
    g = ig.Graph(directed=False)
    g.add_vertices(len(entities))
    g.vs["id"] = [entity["id"] for entity in entities]
    g.vs["type"] = [entity["type"] for entity in entities]
    g.vs["properties"] = [entity["properties"] for entity in entities]
    g.vs["embedding"] = [entity.get("embedding") for entity in entities]

    edges = []
    for rel in relationships:
        source_id = rel.get("source")
        target_id = rel.get("target")
        if source_id in id_to_vertex and target_id in id_to_vertex:
            edges.append((id_to_vertex[source_id], id_to_vertex[target_id]))
    g.add_edges(edges)

    # Find maximal cliques (as a proxy for overlapping communities)
    # A node can be part of multiple cliques, thus multiple communities
    cliques = g.maximal_cliques()
    
    # Assign communities to entities and generate community summaries
    community_summaries = {} # To store text summaries for each community
    for entity in entities:
        entity_id = entity["id"]
        entity["communities"] = [] # Initialize a list for communities
        
        # Collect entity details for community summary
        entity_summary_parts = []
        if entity.get("type"):
            entity_summary_parts.append(f"Type: {entity.get('type')}")
        if entity.get("properties", {}).get("summary"):
            entity_summary_parts.append(f"Summary: {entity['properties']['summary']}")
        elif entity.get("properties", {}).get("name"):
            entity_summary_parts.append(f"Name: {entity['properties']['name']}")
        
        entity_text_for_summary = ", ".join(entity_summary_parts) if entity_summary_parts else entity_id

        for i, clique in enumerate(cliques):
            if id_to_vertex[entity_id] in clique:
                community_id = f"clique_{i}"
                entity["communities"].append(community_id)
                
                # Add entity's text to the community's summary
                if community_id not in community_summaries:
                    community_summaries[community_id] = []
                community_summaries[community_id].append(entity_text_for_summary)

    # Create standard "Community" entities and add them to the main entities list
    for comm_id, entity_texts in community_summaries.items():
        # Concatenate entity texts to form a community summary
        full_community_summary = " ".join(entity_texts)
        
        # Generate embedding for the community summary
        if full_community_summary:
            embedding = get_embedding(full_community_summary, comm_id)
        else:
            embedding = [0.0] * EMBEDDING_DIMENSION
        
        community_entity = {
            "id": comm_id,
            "type": "Community",
            "properties": {
                "summary": full_community_summary,
                "community_type": "structural"
            },
            "embedding": embedding,
            "communities": [] # Structural communities are not part of other communities
        }
        entities.append(community_entity)

    logging.info(f"Found {len(cliques)} cliques (overlapping communities) and created {len(community_summaries)} standard Community entities.")
    return data

def migrate_to_spanner(data):
    logging.info("Migrating data to Spanner...")
    logging.info(f"Entities received by migrate_to_spanner: {data.get('entities', [])[:5]}") # Log first 5 entities
    logging.info(f"Relationships received by migrate_to_spanner: {data.get('relationships', [])[:5]}") # Log first 5 relationships

    entities = data.get("entities", [])
    relationships = data.get("relationships", [])

    logging.info(f"Entities before entities_to_insert: {entities[:5]}") # Log first 5 entities
    entities_to_insert = [(e["id"], e["type"], json.dumps(e.get("properties",ப்பட்டன)), json.dumps(e.get("embedding", [])), json.dumps(e.get("communities", []))) for e in entities]
    relationships_to_insert = [
        (
            hashlib.sha256(f"{r['source']}-{r['target']}-{r.get('type')}".encode()).hexdigest(),
            r["source"],
            r["target"],
            r.get("type"),
            json.dumps(r.get("properties", {}))
        )
        for r in relationships
        if r.get('source') and r.get('target') and r.get('type') and r.get('type') != 'INSTANCE_OF'
    ]
    instance_of_to_insert = [
        (r["source"], r["target"])
        for r in relationships
        if r.get('source') and r.get('target') and r.get('type') == 'INSTANCE_OF'
    ]

    def chunk_list(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    BATCH_SIZE = 100

    for batch in chunk_list(entities_to_insert, BATCH_SIZE):
        sample_batch = batch[:5] if len(batch) > 5 else batch
        logging.info(f"Inserting/updating Entities table. Sample batch ({len(batch)} items): {sample_batch}")
        eids_in_batch = [item[0] for item in batch] # Eid is the first element
        logging.info(f"Entities Eids in batch: {eids_in_batch}")
        spanner_database.run_in_transaction(lambda transaction: transaction.insert_or_update(
            table="Entities",
            columns=("Eid", "Type", "Properties", "Embedding", "Communities"),
            values=batch,
        ))
        logging.info(f"Inserted {len(batch)} entities.")

    # for batch in chunk_list(relationships_to_insert, BATCH_SIZE):
    #     sample_batch = batch[:5] if len(batch) > 5 else batch
    #     logging.info(f"Inserting/updating Relationships table. Sample batch ({len(batch)} items): {sample_batch}")
    #     rids_in_batch = [item[0] for item in batch] # Rid is the first element
    #     logging.info(f"Relationships Rids in batch: {rids_in_batch}")
    #     spanner_database.run_in_transaction(lambda transaction: transaction.insert_or_update(
    #         table="Relationships",
    #         columns=("Rid", "SourceEid", "TargetEid", "Type", "Properties"),
    #         values=batch,
    #     ))
    #     logging.info(f"Inserted {len(batch)} relationships.")

    # for batch in chunk_list(instance_of_to_insert, BATCH_SIZE):
    #     sample_batch = batch[:5] if len(batch) > 5 else batch
    #     logging.info(f"Inserting/updating InstanceOf table. Sample batch ({len(batch)} items): {sample_batch}")
    #     instance_class_eids_in_batch = [(item[0], item[1]) for item in batch] # (InstanceEid, ClassEid) are the first two elements
    #     logging.info(f"Inserting/updating InstanceOf table. Sample batch ({len(batch)} items): {sample_batch}")
    #     instance_class_eids_in_batch = [(item[0], item[1]) for item in batch] # (InstanceEid, ClassEid) are the first two elements
    #     logging.info(f"InstanceOf InstanceEid, ClassEid in batch: {instance_class_eids_in_batch}")
    #     spanner_database.run_in_transaction(lambda transaction: transaction.insert_or_update(
    #         table="InstanceOf",
    #         columns=("InstanceEid", "ClassEid"),
    #         values=batch,
    #     ))
    #     logging.info(f"Inserted {len(batch)} instance-of relationships.")

    return data

# --- LangChain Sequence ---
consolidation_chain = RunnableSequence(
    decode_pubsub_message,
    fetch_from_redis,
    aggregate_results,
    generate_embeddings,
    cluster_and_merge_entities,
    run_igraph_community_detection,
    migrate_to_spanner,
)

# --- Main Function ---
@functions_framework.cloud_event
def consolidator(cloud_event):
    batch_id = None
    try:
        initialize_clients()
        
        data = decode_pubsub_message(cloud_event)
        batch_id = data.get("batch_id")

        consolidation_chain.invoke(data)
        
        # If everything is successful, update the status to SUCCEEDED
        if batch_id:
            with spanner_database.batch() as batch:
                batch.update(
                    table="WorkflowStatus",
                    columns=("BatchId", "Status", "UpdatedAt"),
                    values=[(batch_id, "SUCCEEDED", spanner.COMMIT_TIMESTAMP)],
                )
            logging.info(f"Successfully updated workflow status for batch ID {batch_id} to SUCCEEDED.")

        return "OK", 200
    except Exception as e:
        logging.error(f'An error occurred in the consolidator for batch_id {batch_id}: {e}', exc_info=True)
        
        # If an error occurs, update the status to FAILED
        if batch_id:
            try:
                with spanner_database.batch() as batch:
                    batch.update(
                        table="WorkflowStatus",
                        columns=("BatchId", "Status", "UpdatedAt"),
                        values=[(batch_id, "FAILED", spanner.COMMIT_TIMESTAMP)],
                    )
                logging.info(f"Successfully updated workflow status for batch ID {batch_id} to FAILED.")
            except Exception as spanner_e:
                logging.error(f"Could not update workflow status for batch ID {batch_id} to FAILED: {spanner_e}", exc_info=True)

        return "Internal Server Error", 500