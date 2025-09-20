# Tests for graph_processing.py
import pytest
from unittest.mock import Mock

from graph_processing import GraphProcessor
from tests.test_data import redis_data

@pytest.fixture
def mock_llm_operations():
    return Mock()

def test_aggregate_results(mock_llm_operations):
    graph_processor = GraphProcessor(mock_llm_operations)
    data = {
        "batch_id": "test_batch_123",
        "partial_results": redis_data
    }
    result = graph_processor.aggregate_results(data)
    assert len(result["entities"]) > 0
    assert len(result["relationships"]) > 0

# I will add more tests here later to test the other functions in graph_processing.py