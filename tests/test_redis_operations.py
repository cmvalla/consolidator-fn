# Tests for redis_operations.py
import pytest
from unittest.mock import Mock

from redis_operations import RedisOperations
from tests.test_data import redis_data

@pytest.fixture
def mock_redis_client():
    return Mock()

def test_fetch_from_redis(mock_redis_client):
    redis_ops = RedisOperations(mock_redis_client)
    batch_id = "fe595147a4851c99fd4917e6a8837b8e65ae962fd25a2f1148ba6b511f28969c"
    data = {"batch_id": batch_id}
    
    mock_redis_client.lrange.return_value = redis_data

    result = redis_ops.fetch_from_redis(data)

    mock_redis_client.lrange.assert_called_once_with(f"batch:{batch_id}:results", 0, -1)
    assert result["batch_id"] == batch_id
    assert len(result["partial_results"]) == 3

def test_store_consolidated_results_in_redis(mock_redis_client):
    redis_ops = RedisOperations(mock_redis_client)
    batch_id = "test_batch_123"
    data = {
        "batch_id": batch_id,
        "entities": [{"id": "e1"}],
        "relationships": []
    }

    redis_ops.store_consolidated_results_in_redis(data)

    mock_redis_client.hset.assert_called_once()
    mock_redis_client.expire.assert_called_once_with(f"consolidated_batch:{batch_id}", 86400)
