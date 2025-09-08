# LLM operations for the consolidator function
import logging
import requests
import time
import os
import json
import re
from concurrent.futures import ThreadPoolExecutor
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from config import Config

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

SUMMARY_PROMPT = PromptTemplate.from_template(
    "Summarize the following text in one concise sentence:\n\n"
    "TEXT:\n---\n{text_chunk}\n---\n\nSummary:")

class LLMOperations:
    def __init__(self, llm):
        self.llm = llm

    def get_embedding(self, text: str, entity_id: str = "Unknown"):
        """Generates an embedding for a given text by calling the graphrag-embedding service."""
        embedding_service_url = Config.EMBEDDING_SERVICE_URL
        if not embedding_service_url:
            logging.error("EMBEDDING_SERVICE_URL environment variable not set.")
            return [0.0] * Config.EMBEDDING_DIMENSION

        MAX_RETRIES = 10
        INITIAL_BACKOFF_SECONDS = 1
        MAX_BACKOFF_SECONDS = 600  # 10 minutes
        
        retries = 0
        backoff_seconds = INITIAL_BACKOFF_SECONDS

        while retries < MAX_RETRIES:
            try:
                token_url = f"http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity?audience={embedding_service_url}"
                token_response = requests.get(token_url, headers={"Metadata-Flavor": "Google"})
                token = token_response.text
                headers = {"Authorization": f"Bearer {token}"}
                
                logging.debug(f"Sending embedding request for entity {entity_id}: url={embedding_service_url}, payload={{"text": text, "embedding_source": "gemini"}}, headers={{'Authorization': 'Bearer ...'}}")
                response = requests.post(embedding_service_url, json={"text": text, "embedding_source": "gemini"}, headers=headers)
                logging.debug(f"Received raw embedding response for entity {entity_id} (Status: {response.status_code}): {response.text}")
                
                if response.status_code == 200:
                    embedding = response.json().get("embedding")
                    logging.debug(f"Raw embedding value for entity {entity_id}: {embedding}")
                    if embedding:
                        # If the embedding is a list of lists (e.g., from batch embedding), take the first one
                        if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
                            embedding = embedding[0]
                            logging.debug(f"Processed embedding (first element) for entity {entity_id}: {embedding}")
                        return embedding
                    else:
                        logging.warning(f"Embedding not found in response for entity: {entity_id}")
                        return [0.0] * Config.EMBEDDING_DIMENSION
                
                elif response.status_code >= 500:
                    logging.warning(f"Embedding service returned a server error ({response.status_code}) for entity {entity_id}. Retrying in {backoff_seconds} seconds...")
                    time.sleep(backoff_seconds)
                    retries += 1
                    backoff_seconds = min(backoff_seconds * 2, MAX_BACKOFF_SECONDS)
                
                else:
                    logging.error(f"Embedding service returned a client error ({response.status_code}) for entity {entity_id}: {response.text}")
                    response.raise_for_status()
                    return [0.0] * Config.EMBEDDING_DIMENSION

            except requests.exceptions.RequestException as e:
                logging.error(f"Error calling embedding service for entity {entity_id}: {e}")
                time.sleep(backoff_seconds)
                retries += 1
                backoff_seconds = min(backoff_seconds * 2, MAX_BACKOFF_SECONDS)

        logging.error(f"Failed to get embedding for entity {entity_id} after {MAX_RETRIES} retries.")
        return [0.0] * Config.EMBEDDING_DIMENSION

    def generate_embeddings(self, data):
        """Generates embeddings for all entities and communities."""
        entities = data.get("entities", [])
        
        summarization_chain = LLMChain(llm=self.llm, prompt=SUMMARY_PROMPT)

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
                if not text_to_embed:
                    logging.warning(f"Community {entity.get('id')} has an empty summary. No embedding will be generated.")
            else:
                text_to_embed = f"Type: {entity_type}, Properties: {json.dumps(properties)}"

            if not text_to_embed:
                logging.warning(f"Skipping embedding for entity {entity.get('id')} because there is no text to embed.")
                entity['embedding'] = [0.0] * Config.EMBEDDING_DIMENSION
                continue

            entity['embedding'] = self.get_embedding(text_to_embed, entity.get('id'))

        return data

    def generate_class_properties(self, class_property_chain, instances_text, schema, source_text):
        """Helper function to run LLM chain in a thread, with logging."""
        prompt = CLASS_PROPERTY_GENERATION_PROMPT.format(instances_text=instances_text, schema=schema, source_text=source_text)
        logging.info(f"Querying LLM for class properties. Prompt length: {len(prompt)}. Prompt:\n{prompt}")
        response = class_property_chain.invoke({"instances_text": instances_text, "schema": schema, "source_text": source_text}).get("text")
        logging.info(f"LLM response for class properties: {response}")
        return response

    def extract_json_from_llm_response(self, text):
        """
        Extracts a JSON object from the model's text response and performs basic validation.
        Handles markdown code blocks.
        """
        # Find the start of the JSON object
        json_start = text.find('{')
        # Find the end of the JSON object
        json_end = text.rfind('}')
        if json_start != -1 and json_end != -1:
            json_str = text[json_start:json_end+1]
            return json_str
        return text