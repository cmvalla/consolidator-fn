# Tests for llm_operations.py
import pytest
from unittest.mock import Mock, patch
import json
import os

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
            mock_post.return_value.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
            embedding = llm_ops.get_embedding("test text")
            assert embedding == [0.1, 0.2, 0.3]

@patch("llm_operations.LLMChain")
def test_generate_embeddings(mock_llm_chain, mock_llm):
    llm_ops = LLMOperations(mock_llm)
    data = {
        "entities": [
            {"id": "e1", "type": "Chunk", "properties": {"summary": "summary1"}},
            {"id": "e2", "type": "Person", "properties": {"name": "John Doe"}}
        ]
    }

    with patch.object(llm_ops, 'get_embedding') as mock_get_embedding:
        mock_get_embedding.return_value = [0.1, 0.2, 0.3]
        result = llm_ops.generate_embeddings(data)
        assert len(result["entities"]) == 2
        assert "embedding" in result["entities"][0]
        assert "embedding" in result["entities"][1]

def test_extract_json_from_llm_response(mock_llm):
    llm_ops = LLMOperations(mock_llm)
    text = '''```json
{"key": "value"}
```'''
    result = llm_ops.extract_json_from_llm_response(text)
    assert result == '{"key": "value"}'
