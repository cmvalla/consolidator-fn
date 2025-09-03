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
import uuid
from sklearn.metrics.pairwise import cosine_similarity

from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
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

EMBEDDING_DIMENSION = 768
gemini_embeddings_client = None



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
    template="You are a knowledge graph expert. Your task is to define a 'Class' that represents a collection of similar 'Instance' entities. "
    "The goal is to find the most specific, meaningful classification for the instances based *only* on the provided context. "
    "- **Specificity is key:** Do not generalize to high-level categories like 'Fairy Tale Character' if a more specific class like 'Wolf' or 'Antagonist' is appropriate within the source text. "
    "- **Use the context:** The 'Class' name and description should be grounded in the 'Instances' and the 'Source Text' provided. "
    "- **Create a Schema:** Generate a JSON object for the 'Class' that adheres to the following JSON schema:\n\n"
    "```json\n{schema}\n```\n\n"
    "Instances (as JSON objects):\n{instances_text}\n\n"
    "Source Text (for context):\n{source_text}\n\n"
    "Respond with a single, valid JSON object for the 'Class' entity.",
    input_variables=["instances_text", "schema", "source_text"]
)

def generate_class_eid(name):
    """Creates a unique, URL-safe base64 ID from a string."""
    if not name:
        return None
    # Normalize the name first to create a consistent base for the ID
    normalized_name = re.sub(r'[^\w\s-]', '', name.lower())
    normalized_name = re.sub(r'[-\s]+', '_', normalized_name).strip('_')
    
    # Base64 encode the normalized name to ensure it is a safe string
    return base64.urlsafe_b64encode(normalized_name.encode('utf-8')).decode('utf-8').replace('=', '-')

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
    global redis_client, llm, spanner_client, spanner_instance, spanner_database, gemini_embeddings_client

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

        # --- Embedding Model Configuration ---
        USE_GEMINI_EMBEDDINGS = os.environ.get("USE_GEMINI_EMBEDDINGS", "false").lower() == "true"
        if USE_GEMINI_EMBEDDINGS:
            logging.info("Initializing Gemini Embeddings client...")
            gemini_embeddings_client = VertexAIEmbeddings(model_name="gemini-embeddings-001", project=GCP_PROJECT, location=LOCATION)
            logging.info("Gemini Embeddings client initialized successfully.")
        else:
            logging.info("Using external embedding service.")
        
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
    # This function now just returns the data, the check is in the main consolidator function
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

def store_consolidated_results_in_redis(data):
    """Stores the consolidated entities and relationships in Redis."""
    batch_id = data["batch_id"]
    consolidated_key = f"consolidated_batch:{batch_id}"
    
    try:
        # Store entities and relationships as JSON strings in a Redis hash
        redis_client.hset(consolidated_key, mapping={
            "entities": json.dumps(data["entities"]),
            "relationships": json.dumps(data["relationships"])
        })
        # Set an expiry for the key (e.g., 24 hours)
        redis_client.expire(consolidated_key, 86400)
        logging.info(f"Stored consolidated results for batch {batch_id} in Redis.")
    except Exception as e:
        logging.error(f"Error storing consolidated results for batch {batch_id} in Redis: {e}", exc_info=True)
    return data

def get_embedding(text: str, entity_id: str = "Unknown"):
    """Generates an embedding for a given text, with retry logic, using either Gemini Embeddings or an external service."""
    
    USE_GEMINI_EMBEDDINGS = os.environ.get("USE_GEMINI_EMBEDDINGS", "false").lower() == "true"

    if USE_GEMINI_EMBEDDINGS:
        if gemini_embeddings_client is None:
            logging.error("Gemini Embeddings client not initialized.")
            return [0.0] * EMBEDDING_DIMENSION
        try:
            logging.info(f"Generating Gemini embedding for entity {entity_id}")
            embedding = gemini_embeddings_client.embed_query(text)
            return embedding
        except Exception as e:
            logging.error(f"Error generating Gemini embedding for entity {entity_id}: {e}", exc_info=True)
            return [0.0] * EMBEDDING_DIMENSION
    else:
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
    Clusters similar entities, creates Class nodes with name-based IDs, and merges
    classes with the same name by re-linking their instances.
    """
    entities = data.get("entities", [])
    relationships = data.get("relationships", [])

    if not entities:
        return data

    id_to_entity = {entity['id']: entity for entity in entities}

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

    clusterable_entities = [e for e in entities if e.get("type") not in ["Chunk", "Community"]]
    entity_ids = [e["id"] for e in clusterable_entities]
    embeddings = np.array([e.get("embedding") for e in clusterable_entities])

    valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None and len(emb) > 0]
    if len(valid_indices) < 2:
        logging.warning("Not enough entities with embeddings to perform clustering.")
        return data
        
    valid_embeddings = embeddings[valid_indices]
    valid_entity_ids = [entity_ids[i] for i in valid_indices]
    
    similarity_matrix = cosine_similarity(valid_embeddings)

    visited = [False] * len(valid_entity_ids)
    clusters = []
    for i in range(len(valid_entity_ids)):
        if visited[i]:
            continue
        cluster = [valid_entity_ids[i]]
        visited[i] = True
        for j in range(i + 1, len(valid_entity_ids)):
            if not visited[j] and similarity_matrix[i][j] > similarity_threshold:
                cluster.append(valid_entity_ids[j])
                visited[j] = True
        clusters.append(cluster)

    class_property_chain = LLMChain(llm=llm, prompt=CLASS_PROPERTY_GENERATION_PROMPT)
    summarization_chain = LLMChain(llm=llm, prompt=CLUSTER_SUMMARY_PROMPT)

    name_to_class_entity = {}
    class_id_map = {}

    with ThreadPoolExecutor(max_workers=int(os.environ.get("MAX_WORKERS", 5))) as executor:
        future_to_cluster = {}
        for cluster_member_ids in clusters:
            instances_text_parts = []
            source_text_parts = set()
            cluster_text_parts = []

            for member_id in cluster_member_ids:
                member_entity = id_to_entity.get(member_id)
                if member_entity:
                    properties = member_entity.get('properties', {})
                    instances_text_parts.append(f"- {json.dumps(properties)}")
                    source_text = entity_id_to_source_text.get(member_id)
                    if source_text:
                        source_text_parts.add(source_text)
                    
                    entity_type = member_entity.get('type', '')
                    cluster_text_parts.append(f"Type: {entity_type}, Properties: {json.dumps(properties)}")

            instances_text = "\n".join(instances_text_parts)
            source_text_context = "\n\n---\n\n".join(source_text_parts)
            schema_str = json.dumps(CLASS_SCHEMA, indent=2)
            
            future = executor.submit(generate_class_properties, class_property_chain, instances_text, schema_str, source_text_context)
            future_to_cluster[future] = {
                "member_ids": cluster_member_ids,
                "cluster_text": " ".join(cluster_text_parts)
            }

        processed_clusters = 0
        total_clusters = len(clusters)
        for future in as_completed(future_to_cluster):
            cluster_info = future_to_cluster[future]
            cluster_member_ids = cluster_info["member_ids"]
            processed_clusters += 1
            try:
                generated_properties_str = future.result()
                extracted_json_str = extract_json_from_llm_response(generated_properties_str)
                generated_properties = json.loads(extracted_json_str)
                class_name = generated_properties.get("name")
                if not class_name or not class_name.strip():
                    logging.warning(f"Skipping class creation for cluster due to empty class name. Cluster info: {cluster_info}")
                    continue
                class_eid = generate_class_eid(class_name)

                if not class_eid:
                    logging.warning(f"Could not generate a valid EID for class from name: '{class_name}'. Skipping cluster.")
                    continue

                if class_name in name_to_class_entity:
                    existing_class_eid = name_to_class_entity[class_name]["id"]
                    for member_id in cluster_member_ids:
                        class_id_map[member_id] = existing_class_eid
                    logging.info(f"Merged cluster into existing class '{class_name}' (ID: {existing_class_eid})")
                else:
                    summary = summarization_chain.invoke({"cluster_text": cluster_info["cluster_text"]}).get("text")
                    embedding = get_embedding(summary, class_eid)
                    class_entity = {
                        "id": class_eid,
                        "type": "Class",
                        "properties": generated_properties,
                        "embedding": embedding
                    }
                    name_to_class_entity[class_name] = class_entity
                    for member_id in cluster_member_ids:
                        class_id_map[member_id] = class_eid
                    logging.info(f"Created new class '{class_name}' (ID: {class_eid})")

            except Exception as e:
                logging.error(f"Failed to process future for cluster: {e}", exc_info=True)
            
            if total_clusters > 0 and processed_clusters % (total_clusters // 10 or 1) == 0:
                progress = (processed_clusters / total_clusters) * 100
                logging.info(f"Class property generation progress: {progress:.0f}% ({processed_clusters}/{total_clusters})")


    new_entities = list(name_to_class_entity.values())
    new_relationships = []

    for entity in entities:
        if entity.get("type") not in ["Chunk", "Community"]:
            entity["type"] = "Instance"
        new_entities.append(entity)
        
        class_id = class_id_map.get(entity["id"])
        if class_id:
            new_relationships.append({
                "source": entity["id"],
                "target": class_id,
                "type": "INSTANCE_OF",
                "properties": {"description": "Indicates that an entity is an instance of a specific class."}
            })

    for rel in relationships:
        source_class_id = class_id_map.get(rel.get("source"))
        target_class_id = class_id_map.get(rel.get("target") )
        if source_class_id and target_class_id and source_class_id != target_class_id:
            new_relationships.append({
                "source": source_class_id,
                "target": target_class_id,
                "type": rel.get("type"),
                "properties": rel.get("properties", {})
            })

    data["entities"] = new_entities
    data["relationships"] = new_relationships
    
    logging.info(f"Clustering complete. Result: {len(new_entities)} entities, {len(new_relationships)} relationships.")
    return data

def deduplicate_entities(data):
    """Finds and resolves duplicate Eids before community detection."""
    logging.info("Starting entity de-duplication process...")
    entities = data.get("entities", [])
    relationships = data.get("relationships", [])
    id_to_entity = {e["id"]: e for e in entities}

    eid_groups = {}
    for entity in entities:
        eid = entity["id"]
        if eid not in eid_groups:
            eid_groups[eid] = []
        eid_groups[eid].append(entity)

    final_entities = {}
    eids_to_remap = {}

    for eid, group in eid_groups.items():
        if len(group) == 1:
            final_entities[eid] = group[0]
            continue

        logging.warning(f"Found duplicate EID: '{eid}' for {len(group)} entities.")

        # Case 1: Duplicate Classes
        if all(e.get("type") == "Class" for e in group):
            logging.info(f"Handling duplicate Class EID: {eid}")
            # Count instances for each class
            instance_counts = {e["id"]: 0 for e in group}
            for rel in relationships:
                if rel.get("type") == "INSTANCE_OF" and rel.get("target") in instance_counts:
                    instance_counts[rel.get("target")] += 1
            
            # Sort classes by instance count, descending
            sorted_classes = sorted(group, key=lambda e: instance_counts[e["id"]], reverse=True)
            winner = sorted_classes[0]
            losers = sorted_classes[1:]
            final_entities[winner["id"]] = winner

            for loser in losers:
                eids_to_remap[loser["id"]] = winner["id"]
                logging.info(f"Merging class '{loser['id']}' into '{winner['id']}'.")

        # Case 2: Duplicate Instances
        elif all(e.get("type") == "Instance" for e in group):
            logging.info(f"Handling duplicate Instance EID: {eid}")
            # Keep the first one, rename the others
            final_entities[eid] = group[0]
            for i, duplicate in enumerate(group[1:]):
                original_id = duplicate["id"]
                while True:
                    new_eid = f"{original_id}_{uuid.uuid4().hex[:6]}"
                    if new_eid not in id_to_entity and new_eid not in final_entities:
                        break
                
                eids_to_remap[original_id] = new_eid
                duplicate["id"] = new_eid
                final_entities[new_eid] = duplicate
                logging.info(f"Renamed duplicate instance '{original_id}' to '{new_eid}'.")
        else:
            logging.warning(f"Unhandled duplicate EID case for eid '{eid}'. Keeping all entities.")
            for entity in group:
                final_entities[entity["id"]] = entity

    # Remap relationships
    for rel in relationships:
        if rel["source"] in eids_to_remap:
            rel["source"] = eids_to_remap[rel["source"]]
        if rel["target"] in eids_to_remap:
            rel["target"] = eids_to_remap[rel["target"]]

    data["entities"] = list(final_entities.values())
    logging.info(f"De-duplication complete. Result: {len(data['entities'])} entities.")
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
            if id_to_vertex.get(entity_id) is not None and id_to_vertex[entity_id] in clique:
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
    """Migrates the final graph data to Cloud Spanner, with robust filtering and error logging."""
    logging.info("Migrating data to Spanner...")
    
    entities = data.get("entities", [])
    relationships = data.get("relationships", [])

    # --- Robust Filtering ---
    valid_entities = [e for e in entities if e.get("id") and isinstance(e.get("id"), str) and e["id"].strip()]
    if len(valid_entities) != len(entities):
        logging.warning(f"Filtered out {len(entities) - len(valid_entities)} entities with invalid IDs.")

    valid_eids = {e["id"] for e in valid_entities}

    entities_to_insert = [
        (e["id"], e["type"], json.dumps(e.get("properties", {})), json.dumps(e.get("embedding", [])), json.dumps(e.get("communities", [])))
        for e in valid_entities
    ]

    def is_valid_rel(r):
        source = r.get("source")
        target = r.get("target")
        return source and isinstance(source, str) and source.strip() and \
               target and isinstance(target, str) and target.strip() and \
               source in valid_eids and target in valid_eids

    valid_relationships = [r for r in relationships if is_valid_rel(r)]
    if len(valid_relationships) != len(relationships):
        logging.warning(f"Filtered out {len(relationships) - len(valid_relationships)} relationships with invalid or dangling EIDs.")

    relationships_to_insert = [
        (
            hashlib.sha256(f"{r['source']}-{r['target']}-{r.get('type')}".encode()).hexdigest(),
            r["source"],
            r["target"],
            r.get("type"),
            json.dumps(r.get("properties", {}))
        )
        for r in valid_relationships
        if r.get('type') != 'INSTANCE_OF'
    ]
    instance_of_to_insert = [
        (r["source"], r["target"])
        for r in valid_relationships
        if r.get('type') == 'INSTANCE_OF'
    ]
    # --- End of Filtering ---

    def chunk_list(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    BATCH_SIZE = 100

    if entities_to_insert:
        for batch in chunk_list(entities_to_insert, BATCH_SIZE):
            try:
                spanner_database.run_in_transaction(lambda transaction: transaction.insert_or_update(
                    table="Entities",
                    columns=("Eid", "Type", "Properties", "Embedding", "Communities"),
                    values=batch,
                ))
                logging.info(f"Inserted/updated {len(batch)} entities.")
            except Exception as e:
                logging.error(f"Error inserting batch into Entities table: {e}")
                logging.error("Failing entity IDs for Entities table:")
                for row in batch:
                    logging.error(f"  - entity_id: {row[0]}")
                raise e

    if relationships_to_insert:
        for batch in chunk_list(relationships_to_insert, BATCH_SIZE):
            try:
                spanner_database.run_in_transaction(lambda transaction: transaction.insert_or_update(
                    table="Relationships",
                    columns=("Rid", "SourceEid", "TargetEid", "Type", "Properties"),
                    values=batch,
                ))
                logging.info(f"Inserted/updated {len(batch)} relationships.")
            except Exception as e:
                logging.error(f"Error inserting batch into Relationships table: {e}")
                logging.error("Failing Rids for Relationships table:")
                for row in batch:
                    logging.error(f"  - Rid: {row[0]}")
                raise e

    if instance_of_to_insert:
        for batch in chunk_list(instance_of_to_insert, BATCH_SIZE):
            try:
                spanner_database.run_in_transaction(lambda transaction: transaction.insert_or_update(
                    table="InstanceOf",
                    columns=("InstanceEid", "ClassEid"),
                    values=batch,
                ))
                logging.info(f"Inserted/updated {len(batch)} instance-of relationships.")
            except Exception as e:
                logging.error(f"Error inserting batch into InstanceOf table: {e}")
                logging.error("Failing Eids for InstanceOf table:")
                for row in batch:
                    logging.error(f"  - InstanceEid: {row[0]}, ClassEid: {row[1]}")
                raise e

    return data

# --- LangChain Sequence ---

# Define the processing chain, starting from aggregation
processing_chain = RunnableSequence(
    aggregate_results,
    store_consolidated_results_in_redis,
    generate_embeddings,
    cluster_and_merge_entities,
    deduplicate_entities,
    run_igraph_community_detection,
    migrate_to_spanner,
)

# --- Main Function ---
@functions_framework.cloud_event
def consolidator(cloud_event):
    batch_id = None
    try:
        initialize_clients() # This function initializes global clients
        
        data = decode_pubsub_message(cloud_event)
        
        # Fetch data and check if it's empty
        fetched_data = fetch_from_redis(data)
        if not fetched_data.get("partial_results"):
            logging.info(f"No data found in Redis for batch {data.get('batch_id')}. Stopping execution.")
            return "OK", 200

        # Invoke the processing chain with the fetched data
        processing_chain.invoke(fetched_data)
        
        batch_id = fetched_data.get("batch_id")
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
        batch_id = batch_id or (data and data.get("batch_id"))
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
00
