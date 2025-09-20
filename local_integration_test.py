import logging
import requests
import os

# Assuming these are in the same directory or accessible via PYTHONPATH
from graph_processing import GraphProcessor
from llm_operations import LLMOperations
from config import Config

from langchain_core.messages import AIMessage
from unittest.mock import Mock

# Configure logging for better visibility
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Simulate Configuration ---
# For local testing, we can set a dummy embedding dimension
Config.EMBEDDING_DIMENSION = 768

# --- Simulate Input Data ---
# Minimal entities and relationships to form a community
sample_entities = [
    {"id": "entity_1", "type": "Person", "properties": {"name": "Alice"}, "embedding": [0.1]*768},
    {"id": "entity_2", "type": "Person", "properties": {"name": "Bob"}, "embedding": [0.2]*768},
    {"id": "entity_3", "type": "Location", "properties": {"name": "Park"}, "embedding": [0.3]*768},
]

sample_relationships = [
    {"source": "entity_1", "target": "entity_2", "type": "KNOWS"},
    {"source": "entity_1", "target": "entity_3", "type": "VISITS"},
    {"source": "entity_2", "target": "entity_3", "type": "VISITS"},
]

# Data structure as expected by consolidator
simulated_data = {
    "batch_id": "test_batch_123",
    "entities": sample_entities,
    "relationships": sample_relationships
}

# --- Mock LLM for LangChain chains ---
# Instantiate the LLM for LangChain chains
llm_for_chains = Mock()
llm_for_chains.invoke.return_value = AIMessage(content="This is a simulated summary.")

# --- Real Embedding Service LLM ---
class RealEmbeddingServiceLLM:
    def __init__(self, service_url):
        self.service_url = service_url

    def get_embedding(self, text, entity_id):
        logging.debug(f"RealEmbeddingServiceLLM: Requesting embedding for '{text}' (entity: {entity_id})")
        try:
            # Obtain ID token for authentication
            # When running locally, use gcloud auth print-identity-token
            # When running in Cloud Run, use the metadata server
            if os.environ.get("K_SERVICE"):
                # Running in Cloud Run
                token_url = f"http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity?audience={self.service_url}"
                token_response = requests.get(token_url, headers={"Metadata-Flavor": "Google"})
                token = token_response.text
            else:
                # Running locally
                # Ensure gcloud is authenticated and has permissions to get identity tokens
                token = os.popen(f"gcloud auth print-identity-token --audiences={self.service_url}").read().strip()
                if not token:
                    logging.error("Failed to get identity token locally. Ensure gcloud is authenticated.")
                    return {"clustering": [0.0] * Config.EMBEDDING_DIMENSION, "semantic_search": [0.0] * Config.EMBEDDING_DIMENSION}

            headers = {"Authorization": f"Bearer {token}"}
            
            logging.debug(f"RealEmbeddingServiceLLM: Sending embedding request for entity {entity_id}: text='{text}'")
            response = requests.post(self.service_url, json={"text": text, "embedding_source": "gemini"}, headers=headers)
            logging.debug(f"RealEmbeddingServiceLLM: Received embedding response for entity {entity_id} (Status: {response.status_code}): {response.text}")

            if response.status_code == 200:
                all_embeddings = response.json().get("embeddings")
                if all_embeddings and isinstance(all_embeddings, dict):
                    clustering_embedding = all_embeddings.get("clustering", [[0.0] * Config.EMBEDDING_DIMENSION])[0]
                    semantic_search_embedding = all_embeddings.get("semantic_search", [[0.0] * Config.EMBEDDING_DIMENSION])[0]
                    return {"clustering": clustering_embedding, "semantic_search": semantic_search_embedding}
                else:
                    logging.warning(f"Embeddings not found or invalid in response for entity: {entity_id}. Full response: {response.text}")
                    return {"clustering": [0.0] * Config.EMBEDDING_DIMENSION, "semantic_search": [0.0] * Config.EMBEDDING_DIMENSION}
            else:
                logging.error(f"Embedding service returned an unexpected status code ({response.status_code}): {response.text}")
                response.raise_for_status()
                return {"clustering": [0.0] * Config.EMBEDDING_DIMENSION, "semantic_search": [0.0] * Config.EMBEDDING_DIMENSION}

        except Exception as e:
            logging.error(f"RealEmbeddingServiceLLM: Error requesting embedding for '{text}' (entity: {entity_id}): {e}", exc_info=True)
            return {"clustering": [0.0] * Config.EMBEDDING_DIMENSION, "semantic_search": [0.0] * Config.EMBEDDING_DIMENSION}

# --- Instantiate Operations Classes ---
# Point to the local embedding service URL
EMBEDDING_SERVICE_LOCAL_URL = "https://graphrag-embedding-kg7odfkvta-ew.a.run.app"

# Instantiate the embedding service client
embedding_service_client = RealEmbeddingServiceLLM(EMBEDDING_SERVICE_LOCAL_URL)

llm_ops = LLMOperations(llm_for_chains)

def mock_get_embeddings(texts):
    return [
        {"clustering": [0.1]*Config.EMBEDDING_DIMENSION, "semantic_search": [0.1]*Config.EMBEDDING_DIMENSION} for _ in texts
    ]

llm_ops.get_embeddings = mock_get_embeddings

graph_processor = GraphProcessor(llm_ops)


# --- Perform Integration Test Steps ---
logging.info("Starting local integration test...")

# 1. Run igraph community detection
logging.info("Running igraph community detection...")
community_detection_result = graph_processor.run_igraph_community_detection(simulated_data)
logging.debug(f"Community Detection Result (entities): {community_detection_result['entities']}")
logging.debug(f"Community Detection Result (relationships): {community_detection_result['relationships']}")

# Inspect community summaries
logging.info("Inspecting Community entities and their summaries:")
for entity in community_detection_result['entities']:
    if entity.get('type') == 'Community':
        summary = entity.get('properties', {}).get('summary')
        logging.info(f"  Community ID: {entity.get('id')}, Summary: '{summary}'")
        if not summary:
            logging.warning(f"  Community {entity.get('id')} has an empty summary!")

# 2. Generate embeddings for all entities (including new communities)
logging.info("Generating embeddings for all entities...")
final_data_with_embeddings = llm_ops.generate_embeddings(community_detection_result)
logging.debug(f"Final Data with Embeddings (entities): {final_data_with_embeddings['entities']}")

logging.info("Local integration test complete.")