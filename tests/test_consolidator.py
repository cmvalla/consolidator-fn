import pytest
from unittest.mock import Mock, patch
from main import consolidator

@patch("main.ClientFactory")
@patch("main.decode_pubsub_message")
@patch("main.psutil")
def test_consolidator_orchestration(mock_psutil, mock_decode_pubsub, mock_client_factory):
    # Mock psutil
    mock_memory_info = Mock()
    mock_memory_info.total = 8 * 1024**3
    mock_memory_info.available = 4 * 1024**3
    mock_memory_info.used = 4 * 1024**3
    mock_memory_info.percent = 50.0
    mock_psutil.virtual_memory.return_value = mock_memory_info
    mock_psutil.cpu_percent.return_value = 10.0

    # Mock the input cloud event
    mock_cloud_event = Mock()
    mock_decode_pubsub.return_value = {"batch_id": "test_batch_123"}

    # Mock the clients
    mock_redis_client = Mock()
    mock_db_session = Mock()
    mock_llm = Mock()
    
    # Configure the mock factory to return mock clients
    mock_factory_instance = mock_client_factory.return_value
    mock_factory_instance.get_redis_client.return_value = mock_redis_client
    mock_factory_instance.get_db_session.return_value = mock_db_session
    mock_factory_instance.get_llm.return_value = mock_llm

    # Mock the operations classes
    with patch("main.RedisOperations") as mock_redis_ops, \
         patch("main.SpannerOperations") as mock_spanner_ops, \
         patch("main.LLMOperations") as mock_llm_ops, \
         patch("main.GraphProcessor") as mock_graph_processor:

        # Mock the return values of the operations
        mock_redis_client.exists.return_value = False
        mock_redis_ops.return_value.fetch_from_redis.return_value = {"partial_results": ["some_data"]}
        mock_graph_processor.return_value.aggregate_results.return_value = {"entities": [], "relationships": []}
        mock_llm_ops.return_value.generate_embeddings.return_value = {"entities": [], "relationships": []}
        mock_graph_processor.return_value.cluster_and_merge_entities.return_value = {"entities": [], "relationships": []}
        mock_graph_processor.return_value.deduplicate_entities.return_value = {"entities": [], "relationships": []}
        mock_graph_processor.return_value.run_igraph_community_detection.return_value = {"entities": [], "relationships": []}

        # Call the consolidator function
        response, status_code = consolidator(mock_cloud_event)

        # Assertions
        assert status_code == 200
        assert response == "OK"
        
        mock_redis_ops.return_value.fetch_from_redis.assert_called_once()
        mock_graph_processor.return_value.aggregate_results.assert_called_once()
        mock_llm_ops.return_value.generate_embeddings.assert_called_once()
        mock_graph_processor.return_value.cluster_and_merge_entities.assert_called_once()
        mock_graph_processor.return_value.deduplicate_entities.assert_called_once()
        mock_graph_processor.return_value.run_igraph_community_detection.assert_called_once()
        mock_redis_ops.return_value.store_consolidated_results_in_redis.assert_called_once()
        mock_spanner_ops.return_value.migrate_to_spanner.assert_called_once()
        mock_spanner_ops.return_value.update_workflow_status.assert_called_once_with("test_batch_123", "SUCCEEDED")