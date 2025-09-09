# Tests for llm_operations.py
import pytest
from unittest.mock import Mock, patch
import json
import os
from langchain_core.messages import AIMessage

from llm_operations import LLMOperations

@pytest.fixture
def mock_llm():
    return Mock()

@patch('llm_operations.Config')
def test_get_embedding(mock_config, mock_llm):
    mock_config.EMBEDDING_SERVICE_URL = "mock_url"
    llm_ops = LLMOperations(mock_llm)
    
    with patch("requests.post") as mock_post:
        with patch("requests.get") as mock_get:
            mock_get.return_value.text = "token"
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"embeddings": {"clustering": [[0.1, 0.2, 0.3]], "semantic_search": [[0.4, 0.5, 0.6]]}}
            embedding = llm_ops._get_single_embedding("test text")
            assert embedding == {"clustering": [0.1, 0.2, 0.3], "semantic_search": [0.4, 0.5, 0.6]}

@patch("llm_operations.SUMMARY_PROMPT")
def test_generate_embeddings(mock_summary_prompt, mock_llm):
    llm_ops = LLMOperations(mock_llm)
    data = {
        "entities": [
            {"id": "e1", "type": "Chunk", "properties": {"summary": "", "original_text": "some original text"}},
            {"id": "e2", "type": "Person", "properties": {"name": "John Doe"}}
        ]
    }

    # Mock the invoke method of the llm object
    mock_llm.invoke.return_value = AIMessage(content="simulated summary")

    with patch.object(llm_ops, 'get_embeddings') as mock_get_embeddings:
        mock_get_embeddings.return_value = [
            {"clustering": [0.1, 0.2, 0.3], "semantic_search": [0.4, 0.5, 0.6]},
            {"clustering": [0.1, 0.2, 0.3], "semantic_search": [0.4, 0.5, 0.6]}
        ]
        result = llm_ops.generate_embeddings(data)
        assert len(result["entities"]) == 2
        assert "clustering_embedding" in result["entities"][0]
        assert "semantic_search_embedding" in result["entities"][0]
        assert "clustering_embedding" in result["entities"][1]
        assert "semantic_search_embedding" in result["entities"][1]

    # Assert that llm.invoke was called for the Chunk entity
    mock_llm.invoke.assert_called_once()
