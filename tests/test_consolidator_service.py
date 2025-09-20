import pytest
from unittest.mock import MagicMock, patch
import os
import pickle
import igraph as ig # Import igraph

from consolidator_service import ConsolidatorService
from llm_operations import LLMOperations
from graph_processing import GraphProcessor
from spanner_operations import SpannerOperations
from google.cloud import pubsub_v1
from google.cloud import storage

@pytest.fixture
def mock_llm_ops():
    return MagicMock(spec=LLMOperations)

@pytest.fixture
def mock_graph_processor():
    mock = MagicMock(spec=GraphProcessor)
    # Create a dummy non-empty graph
    dummy_graph = ig.Graph()
    dummy_graph.add_vertex("dummy_vertex")
    # Mock return values for graph processing methods to return a non-empty graph
    mock.cluster_and_merge_entities.return_value = dummy_graph
    mock.deduplicate_entities.return_value = dummy_graph
    mock.run_igraph_community_detection.return_value = dummy_graph
    mock.remove_entities_with_null_keys_and_relationships.return_value = dummy_graph
    return mock

@pytest.fixture
def mock_spanner_ops():
    mock = MagicMock(spec=SpannerOperations)
    mock.acquire_lock.return_value = True  # Assume lock is always acquired for tests
    return mock

@pytest.fixture
def mock_publisher():
    return MagicMock(spec=pubsub_v1.PublisherClient)

@pytest.fixture
def mock_storage_client():
    mock = MagicMock(spec=storage.Client)
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    
    # Create a dummy igraph object and pickle it
    dummy_graph = ig.Graph()
    dummy_graph.add_vertex("test_vertex")
    pickled_graph = pickle.dumps(dummy_graph)

    mock_blob.download_as_bytes.return_value = pickled_graph
    mock_bucket.blob.return_value = mock_blob
    mock.bucket.return_value = mock_bucket
    return mock

@pytest.fixture
def consolidator_service(mock_llm_ops, mock_graph_processor, mock_spanner_ops, mock_publisher, mock_storage_client):
    return ConsolidatorService(
        llm_ops=mock_llm_ops,
        graph_processor=mock_graph_processor,
        spanner_ops=mock_spanner_ops,
        publisher=mock_publisher,
        storage_client=mock_storage_client
    )

def test_process_message_success(consolidator_service, mock_spanner_ops, mock_graph_processor, mock_publisher, mock_storage_client):
    batch_id = "test_batch_id"
    gcs_paths = ["gs://test_bucket/test_graph.pkl"]
    persistor_topic_path = "projects/test_project/topics/test_topic"
    instance_id = "test_instance"

    # Mock os.environ.get for GRAPH_DATA_BUCKET_NAME
    with patch.dict(os.environ, {"GRAPH_DATA_BUCKET_NAME": "test_bucket"}):
        consolidator_service.process_message(batch_id, gcs_paths, persistor_topic_path, instance_id)

        # Assertions
        mock_spanner_ops.acquire_lock.assert_called_once_with(batch_id, instance_id)
        mock_storage_client.bucket.assert_any_call("test_bucket") # For downloading
        mock_storage_client.bucket.assert_any_call("test_bucket")  # For uploading
        mock_graph_processor.cluster_and_merge_entities.assert_called_once()
        mock_graph_processor.deduplicate_entities.assert_called_once()
        mock_graph_processor.run_igraph_community_detection.assert_called_once()
        mock_graph_processor.remove_entities_with_null_keys_and_relationships.assert_called_once()
        mock_publisher.publish.assert_called_once()
        mock_spanner_ops.release_lock.assert_called_once_with(batch_id, "PENDING_PERSISTENCE")

def test_process_message_no_gcs_paths(consolidator_service, mock_spanner_ops):
    batch_id = "test_batch_id"
    gcs_paths = []
    persistor_topic_path = "projects/test_project/topics/test_topic"
    instance_id = "test_instance"

    consolidator_service.process_message(batch_id, gcs_paths, persistor_topic_path, instance_id)

    mock_spanner_ops.acquire_lock.assert_not_called()
    mock_spanner_ops.release_lock.assert_called_once_with(batch_id, "FAILED")

def test_process_message_lock_not_acquired(consolidator_service, mock_spanner_ops):
    mock_spanner_ops.acquire_lock.return_value = False

    batch_id = "test_batch_id"
    gcs_paths = ["gs://test_bucket/test_graph.pkl"]
    persistor_topic_path = "projects/test_project/topics/test_topic"
    instance_id = "test_instance"

    result = consolidator_service.process_message(batch_id, gcs_paths, persistor_topic_path, instance_id)

    assert result is None
    mock_spanner_ops.acquire_lock.assert_called_once_with(batch_id, instance_id)
    mock_spanner_ops.release_lock.assert_not_called()

# Add more tests for error conditions, graph processing failures, GCS upload failures, etc.
