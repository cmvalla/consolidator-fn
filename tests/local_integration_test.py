import os
import time
import json
import pytest
from google.cloud import spanner
import uuid

from clients import ClientFactory
from llm_operations import LLMOperations
from graph_processing import GraphProcessor
from spanner_operations import SpannerOperations
from consolidator_service import ConsolidatorService

# --- Configuration ---
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
SPANNER_INSTANCE_ID = os.environ.get("SPANNER_INSTANCE_ID")
SPANNER_DATABASE_ID = os.environ.get("SPANNER_DATABASE_ID")
CONSOLIDATOR_TOPIC_NAME = os.environ.get("CONSOLIDATOR_TOPIC_NAME", "consolidation-topic-kid")
GRAPH_DATA_BUCKET_NAME = os.environ.get("GRAPH_DATA_BUCKET_NAME")


# --- Constants for Test Data ---
TEST_BATCH_ID = str(uuid.uuid4())
TEST_ENTITY_ID = "test_entity_123"
TEST_ENTITY_TYPE = "Class"
TEST_ENTITY_PROPERTIES = {"name": "Test Entity", "description": "A test entity for local integration tests"}
TEST_ENTITY_EMBEDDING = [0.1, 0.2, 0.3]

TEST_RELATIONSHIP_ID = "test_relationship_456"
TEST_RELATIONSHIP_SOURCE = "test_entity_123"
TEST_RELATIONSHIP_TARGET = "test_entity_457"
TEST_RELATIONSHIP_TYPE = "HAS_PROPERTY"
TEST_RELATIONSHIP_PROPERTIES = {"weight": 1.0}
TEST_RELATIONSHIP_EMBEDDING = [0.4, 0.5, 0.6]

TEST_ENTITY_2_ID = "test_entity_457"
TEST_ENTITY_2_TYPE = "Instance"
TEST_ENTITY_2_PROPERTIES = {"name": "Another Test Entity", "description": "Another test entity for local integration tests"}
TEST_ENTITY_2_EMBEDDING = [0.7, 0.8, 0.9]

# --- Spanner Schema ---
SPANNER_ENTITIES_TABLE = "Entities"
SPANNER_RELATIONSHIPS_TABLE = "Relationships"

# --- Fixtures for Real Clients ---
@pytest.fixture(scope="module")
def client_factory():
    return ClientFactory()

@pytest.fixture(scope="module")
def real_llm_ops(client_factory):
    llm = client_factory.get_llm()
    return LLMOperations(llm)

@pytest.fixture(scope="module")
def real_graph_processor(real_llm_ops):
    return GraphProcessor(real_llm_ops)

@pytest.fixture(scope="module")
def real_spanner_ops(client_factory):
    spanner_client = client_factory.get_spanner_client()
    return SpannerOperations(spanner_client, SPANNER_INSTANCE_ID, SPANNER_DATABASE_ID)



@pytest.fixture(scope="module")
def real_storage_client(client_factory):
    return client_factory.get_storage_client()

@pytest.fixture(scope="module")
def local_consolidator_service(real_llm_ops, real_graph_processor, real_spanner_ops, real_storage_client):
    # Create a mock publisher that does nothing
    class MockPublisher:
        def topic_path(self, project, topic_name):
            return f"projects/{project}/topics/{topic_name}"
        def publish(self, topic_path, data):
            print(f"MockPublisher: Publishing to {topic_path} with data {data}")
            pass

    mock_publisher = MockPublisher()

    return ConsolidatorService(
        llm_ops=real_llm_ops,
        graph_processor=real_graph_processor,
        spanner_ops=real_spanner_ops,
        publisher=mock_publisher,
        storage_client=real_storage_client
    )

@pytest.fixture(scope="module")
def spanner_database_cleanup():
    """Provides a Spanner database client and cleans up test data."""
    spanner_client = spanner.Client(project=PROJECT_ID)
    instance = spanner_client.instance(SPANNER_INSTANCE_ID)
    database = instance.database(SPANNER_DATABASE_ID)
    
    # Clean up Spanner before tests
    with database.batch() as batch:
        batch.delete(SPANNER_ENTITIES_TABLE, spanner.KeySet(all_=True))
        batch.delete(SPANNER_RELATIONSHIPS_TABLE, spanner.KeySet(all_=True))
    
    yield database
    
    # Clean up Spanner after tests
    with database.batch() as batch:
        batch.delete(SPANNER_ENTITIES_TABLE, spanner.KeySet(all_=True))
        batch.delete(SPANNER_RELATIONSHIPS_TABLE, spanner.KeySet(all_=True))

@pytest.fixture(scope="module")
def gcs_setup_for_local_test(real_storage_client):
    """Uses a real igraph object from GCS."""
    gcs_path = "gs://spanner-demo-graph-data-kid/graph_data/ingestion_documents_prj_kid/golang-1758173972.pdf:2025-09-18T05:39:35.758575Z/0_1758198804.pkl"
    # Extract batch_id from the GCS path
    path_parts = gcs_path.split('/')
    # The batch_id is the part before the last segment (the pkl file name)
    # and after the 'graph_data/' prefix.
    batch_id = '/'.join(path_parts[4:-1]) 

    print(f"Using real graph from GCS: {gcs_path}")

    yield batch_id, [gcs_path]

# --- Helper Functions (copied from integration_test.py, adjusted for local context) ---

def verify_spanner_data(spanner_database):
    """Verifies that the expected data exists in Spanner."""
    with spanner_database.snapshot() as snapshot:
        results = snapshot.read(
            table=SPANNER_ENTITIES_TABLE,
            columns=["Eid", "Type", "Properties", "Embedding"],
            keyset=spanner.KeySet(keys=[[TEST_ENTITY_ID], [TEST_ENTITY_2_ID]])
        )
        rows = list(results)
        assert len(rows) == 2, f"Expected 2 entities in Spanner, found {len(rows)}"
        
        found_entity_1 = False
        found_entity_2 = False
        for row in rows:
            entity_id, entity_type, properties_json, embedding = row
            properties = json.loads(properties_json)
            if entity_id == TEST_ENTITY_ID:
                assert entity_type == TEST_ENTITY_TYPE
                assert properties["name"] == TEST_ENTITY_PROPERTIES["name"]
                assert properties["description"] == TEST_ENTITY_PROPERTIES["description"]
                assert embedding == TEST_ENTITY_EMBEDDING
                found_entity_1 = True
            elif entity_id == TEST_ENTITY_2_ID:
                assert entity_type == TEST_ENTITY_2_TYPE
                assert properties["name"] == TEST_ENTITY_2_PROPERTIES["name"]
                assert properties["description"] == TEST_ENTITY_2_PROPERTIES["description"]
                assert embedding == TEST_ENTITY_2_EMBEDDING
                found_entity_2 = True
        assert found_entity_1 and found_entity_2, "Did not find all expected entities in Spanner"

    with spanner_database.snapshot() as snapshot:
        results = snapshot.read(
            table=SPANNER_RELATIONSHIPS_TABLE,
            columns=["Rid", "SourceEid", "TargetEid", "Type", "Properties"],
            keyset=spanner.KeySet(keys=[[TEST_RELATIONSHIP_ID]])
        )
        rows = list(results)
        assert len(rows) == 1, f"Expected 1 relationship in Spanner, found {len(rows)}"
        
        rel_id, source_id, target_id, rel_type, properties_json = rows[0]
        properties = json.loads(properties_json)
        assert rel_id == TEST_RELATIONSHIP_ID
        assert source_id == TEST_RELATIONSHIP_SOURCE
        assert target_id == TEST_RELATIONSHIP_TARGET
        assert rel_type == TEST_RELATIONSHIP_TYPE
        assert properties["weight"] == TEST_RELATIONSHIP_PROPERTIES["weight"]
    print("Spanner data verified successfully.")

# --- Test Case ---

def test_local_consolidator_service_workflow(local_consolidator_service, spanner_database_cleanup, gcs_setup_for_local_test):
    """
    Local integration test for the ConsolidatorService.
    1. Creates and uploads a dummy igraph object to GCS (handled by gcs_setup_for_local_test fixture).
    2. Directly calls ConsolidatorService.process_message with GCS path.
    3. Waits for consolidation and persistence to complete (by polling Spanner).
    4. Verifies the data in Spanner.
    """
    batch_id, gcs_paths = gcs_setup_for_local_test

    # Directly call the service's process_message method
    local_consolidator_service.process_message(batch_id, gcs_paths, "dummy-topic-path", instance_id)

    # Wait for consolidation and persistence to complete (poll Spanner)
    max_retries = 20 
    retry_delay_seconds = 30 
    
    print(f"Waiting for ConsolidatorService to process data (max {max_retries} retries, {retry_delay_seconds}s delay)...")
    for i in range(max_retries):
        try:
            verify_spanner_data(spanner_database_cleanup)
            print("ConsolidatorService workflow completed and verified.")
            return # Test passed
        except AssertionError as e:
            print(f"Attempt {i+1}/{max_retries}: Spanner verification failed: {e}")
            time.sleep(retry_delay_seconds)
    
    pytest.fail(f"ConsolidatorService did not process data correctly within {max_retries * retry_delay_seconds} seconds.")
