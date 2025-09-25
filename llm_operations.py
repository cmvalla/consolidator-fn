# LLM operations for the consolidator function
import logging
import requests
import time
import json
import re
from typing import List, Dict, Any, Optional
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import Config
import google.auth
import google.auth.transport.requests
import google.oauth2.id_token
import uuid

CLASS_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "A concise and descriptive name for the class."},
        "description": {"type": "string", "description": "A paragraph that summarizes the common theme and purpose of the instances."},
        "properties": {
            "type": "object",
            "description": "An object representing the common schema of the instances. Keys should reflect intrinsic and specific properties relevant to this class, and values should be a representative value or type (e.g., 'string', 'integer', 'boolean'). For example, for a 'Wolf' class, properties might include 'species', 'habitat', 'diet', 'temperament'. For a 'Character' class, properties might include 'gender', 'age', 'occupation', 'personality_traits'."
        }
    },
    "required": ["name", "description", "properties"]
}

CLASS_PROPERTY_GENERATION_PROMPT = PromptTemplate(
    template="You are a knowledge graph expert. Your task is to define 'Class' entities, each representing a collection of similar 'Instance' entities. "
    "The goal is to find the most specific, meaningful classification for each set of instances based *only* on their provided context. "
    "- **Specificity is key:** Do not generalize to high-level categories like 'Fairy Tale Character' if a more specific class like 'Wolf' or 'Antagonist' is appropriate within the source text. "
    "- **Use the context:** Each 'Class' name and description should be grounded in its respective 'Instances' and 'Source Text'. "
    "- **Create a Schema:** Generate a a JSON object for each 'Class' that adheres to the following JSON schema:\n\n"
    "```json\n{schema}\n```\n\n"
    "You will be provided with a JSON array of objects, where each object contains 'instances_text' and 'source_text' for a single cluster. "
    "Your response MUST be a single, valid JSON array of 'Class' entities, corresponding to the input clusters in order.\n\n"
    "Input Clusters (as JSON array of objects):\n{batched_clusters_json}\n\n"
    "Respond with a single, valid JSON array of 'Class' entities.",
    input_variables=["batched_clusters_json", "schema"]
)

SUMMARY_PROMPT = PromptTemplate.from_template(
    "Summarize the following text in one concise sentence:\n\n"
    "TEXT:\n---\n{text_chunk}\n---\n\nSummary:")

class LLMOperations:
    def __init__(self, llm):
        self.llm = llm
        self.summarization_chain = SUMMARY_PROMPT | self.llm
        self.class_property_chain = LLMChain(llm=self.llm, prompt=CLASS_PROPERTY_GENERATION_PROMPT)

    def _get_single_embedding(self, text: str, entity_id: str = "Unknown") -> Dict[str, List[float]]:
        """Generates embeddings for a given text by calling the graphrag-embedding service."""
        embedding_service_url = Config.EMBEDDING_SERVICE_URL
        if not embedding_service_url:
            logging.error("EMBEDDING_SERVICE_URL environment variable not set.")
            return {"clustering": [0.0] * Config.EMBEDDING_DIMENSION, "semantic_search": [0.0] * Config.EMBEDDING_DIMENSION}

        MAX_RETRIES = 10
        INITIAL_BACKOFF_SECONDS = 1
        MAX_BACKOFF_SECONDS = 600  # 10 minutes
        
        retries = 0
        backoff_seconds = INITIAL_BACKOFF_SECONDS

        while retries < MAX_RETRIES:
            try:
                # Explicitly get an ID token for the Cloud Run service
                auth_req = google.auth.transport.requests.Request()
                id_token_raw = google.oauth2.id_token.fetch_id_token(auth_req, embedding_service_url)
                id_token = id_token_raw if id_token_raw is not None else ""

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {id_token}"
                }
                payload = {"text": text, "embedding_source": "gemini", "embedding_types": ["clustering", "semantic_search"], "invocation_id": entity_id}
                logging.info(f"Sending embedding request for entity {entity_id}: url={embedding_service_url}, payload={payload}")
                response = requests.post(embedding_service_url, headers=headers, data=json.dumps(payload))
                logging.info(f"Received raw embedding response for entity {entity_id} (Status: {response.status_code}): {response.text}")
                
                if response.status_code == 200:
                    all_embeddings = response.json().get("embeddings")
                    logging.info(f"Raw embeddings value for entity {entity_id}: {all_embeddings}")
                    if all_embeddings and isinstance(all_embeddings, dict):
                        # Ensure both types are present, return zero embeddings if not
                        clustering_embedding = all_embeddings.get("clustering", [[0.0] * Config.EMBEDDING_DIMENSION])[0]
                        semantic_search_embedding = all_embeddings.get("semantic_search", [[0.0] * Config.EMBEDDING_DIMENSION])[0]
                        return {"clustering": clustering_embedding, "semantic_search": semantic_search_embedding}
                    else:
                        logging.warning(f"Embeddings not found or invalid in response for entity: {entity_id}. Full response: {response.text}")
                        return {"clustering": [0.0] * Config.EMBEDDING_DIMENSION, "semantic_search": [0.0] * Config.EMBEDDING_DIMENSION}
                
                elif response.status_code >= 500:
                    logging.warning(f"Embedding service returned a server error ({response.status_code}) for entity {entity_id}. Retrying in {backoff_seconds} seconds... Full response: {response.text}")
                    time.sleep(backoff_seconds)
                    retries += 1
                    backoff_seconds = min(backoff_seconds * 2, MAX_BACKOFF_SECONDS)
                
                else:
                    logging.error(f"Embedding service returned a client error ({response.status_code}) for entity {entity_id}: {response.text}")
                    response.raise_for_status()
                    return {"clustering": [0.0] * Config.EMBEDDING_DIMENSION, "semantic_search": [0.0] * Config.EMBEDDING_DIMENSION}

            except requests.exceptions.RequestException as e:
                logging.error(f"Error calling embedding service for entity {entity_id}: {e}")
                time.sleep(backoff_seconds)
                retries += 1
                backoff_seconds = min(backoff_seconds * 2, MAX_BACKOFF_SECONDS)

        logging.error(f"Failed to get embeddings for entity {entity_id} after {MAX_RETRIES} retries.")
        return {"clustering": [0.0] * Config.EMBEDDING_DIMENSION, "semantic_search": [0.0] * Config.EMBEDDING_DIMENSION}

    def get_embeddings(self, texts: List[str], ids: List[str]) -> Dict[str, Dict[str, List[float]]]:
        """Generates embeddings for a list of texts in parallel using a thread pool."""
        if not texts:
            return {}

        # Use a ThreadPoolExecutor to parallelize embedding generation
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            # Map each future to its entity ID
            future_to_id = {executor.submit(self._get_single_embedding, text, eid): eid for text, eid in zip(texts, ids)}
            
            results = {}
            for future in as_completed(future_to_id):
                eid = future_to_id[future]
                try:
                    embedding_result = future.result()
                    results[eid] = embedding_result
                except Exception as exc:
                    logging.error(f'Error generating embedding for entity {eid}: {exc}')
                    # Assign default zero embeddings on error
                    results[eid] = {"clustering": [0.0] * Config.EMBEDDING_DIMENSION, "semantic_search": [0.0] * Config.EMBEDDING_DIMENSION}
        return results


    def generate_class_properties(self, batched_clusters_data, schema):
        """Generates class properties for a batch of clusters using the LLM."""
        batched_clusters_json = json.dumps(batched_clusters_data, indent=2)
        prompt_inputs = {"batched_clusters_json": batched_clusters_json, "schema": schema}
        
        class_property_chain = LLMChain(llm=self.llm, prompt=CLASS_PROPERTY_GENERATION_PROMPT)

        logging.info(f"Querying LLM for class properties. Prompt length: {len(prompt_inputs['batched_clusters_json'])}. Prompt:\n{CLASS_PROPERTY_GENERATION_PROMPT.format(**prompt_inputs)}")
        response = class_property_chain.invoke(prompt_inputs).get("text")
        logging.info(f"LLM response for class properties: {response}")
        return response

    def extract_json_from_llm_response(self, text):
        """
        Extracts a JSON object from the model's text response and performs basic validation.
        Handles markdown code blocks.
        """
        # Try to find JSON within markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            json_str = json_match.group(1)
            try:
                # Attempt to parse to validate and then re-serialize to ensure it's clean
                return json.dumps(json.loads(json_str))
            except json.JSONDecodeError:
                logging.warning("Found markdown JSON block, but it was invalid. Attempting fallback extraction.")

        # Fallback to finding the first and last curly braces
        json_start = text.find('[') # Changed to '[' as the expected output is a JSON array
        json_end = text.rfind(']') # Changed to ']' as the expected output is a JSON array
        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_str = text[json_start:json_end+1]
            return json_str
        
        logging.warning(f"No valid JSON found in LLM response: {text}")
        return "[]" # Return an empty JSON array as a safe default
