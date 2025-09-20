import os
import time
import json
import pytest
import igraph as ig
import pickle
from google.cloud import pubsub_v1
from google.cloud import spanner
from google.cloud import storage
import uuid

# --- Configuration ---
# These should be set as environment variables for the test runner
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
SPANNER_INSTANCE_ID = os.environ.get("SPANNER_INSTANCE_ID")
SPANNER_DATABASE_ID = os.environ.get("SPANNER_DATABASE_ID")
CONSOLIDATOR_TOPIC_NAME = os.environ.get("CONSOLIDATOR_TOPIC_NAME", "consolidation-topic-kid")
GRAPH_DATA_BUCKET_NAME = os.environ.get("GRAPH_DATA_BUCKET_NAME")

# --- Constants for Test Data ---
TEST_BATCH_ID = str(uuid.uuid4())
TEST_ENTITY_ID = "test_entity_123"
TEST_ENTITY_TYPE = "Class"
TEST_ENTITY_PROPERTIES = {"name": "Test Entity", "description": "A test entity for integration tests"}
TEST_ENTITY_EMBEDDING = [0.1, 0.2, 0.3]

TEST_RELATIONSHIP_ID = "test_relationship_456"
TEST_RELATIONSHIP_SOURCE = "test_entity_123"
TEST_RELATIONSHIP_TARGET = "test_entity_457"
TEST_RELATIONSHIP_TYPE = "HAS_PROPERTY"
TEST_RELATIONSHIP_PROPERTIES = {"weight": 1.0}
TEST_RELATIONSHIP_EMBEDDING = [0.4, 0.5, 0.6]

TEST_ENTITY_2_ID = "test_entity_457"
TEST_ENTITY_2_TYPE = "Instance"
TEST_ENTITY_2_PROPERTIES = {"name": "Another Test Entity", "description": "Another test entity for integration tests"}
TEST_ENTITY_2_EMBEDDING = [0.7, 0.8, 0.9]


# --- Spanner Schema (Assumed based on LEARNINGS.gemini.md and common patterns) ---
# You might need to adjust this based on your actual Spanner schema
SPANNER_ENTITIES_TABLE = "Entities"
SPANNER_RELATIONSHIPS_TABLE = "Relationships"

# --- Fixtures ---

@pytest.fixture(scope="module")
def spanner_database():
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
def pubsub_publisher():
    """Provides a Pub/Sub publisher client."""
    publisher = pubsub_v1.PublisherClient()
    yield publisher
    publisher.api.transport.close()

@pytest.fixture(scope="module")
def gcs_setup():
    """Creates a dummy igraph object, uploads it to GCS, and cleans up."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(GRAPH_DATA_BUCKET_NAME)
    
    # Create a dummy igraph object
    g = ig.Graph()
    g.add_vertex(TEST_ENTITY_ID, type=TEST_ENTITY_TYPE, properties=json.dumps(TEST_ENTITY_PROPERTIES), embedding=TEST_ENTITY_EMBEDDING)
    g.add_vertex(TEST_ENTITY_2_ID, type=TEST_ENTITY_2_TYPE, properties=json.dumps(TEST_ENTITY_2_PROPERTIES), embedding=TEST_ENTITY_2_EMBEDDING)
    g.add_edge(TEST_ENTITY_ID, TEST_ENTITY_2_ID, id=TEST_RELATIONSHIP_ID, type=TEST_RELATIONSHIP_TYPE, properties=json.dumps(TEST_RELATIONSHIP_PROPERTIES), embedding=TEST_RELATIONSHIP_EMBEDDING)

    serialized_graph = pickle.dumps(g)
    
    # Generate a unique object name
    object_name = f"test_graphs/{TEST_BATCH_ID}/{uuid.uuid4()}.pkl"
    blob = bucket.blob(object_name)
    
    blob.upload_from_string(serialized_graph)
    gcs_path = f"gs://{GRAPH_DATA_BUCKET_NAME}/{object_name}"
    print(f"Uploaded dummy graph to GCS: {gcs_path}")

    yield TEST_BATCH_ID, [gcs_path]
    
    # Clean up GCS object after tests
    blob.delete()
    print(f"Deleted dummy graph from GCS: {gcs_path}")

# --- Helper Functions ---

def publish_consolidator_start_message(publisher, batch_id, gcs_paths):
    """Publishes a message to trigger the consolidator."""
    topic_path = publisher.topic_path(PROJECT_ID, CONSOLIDATOR_TOPIC_NAME)
    message_payload = {
        "batch_id": batch_id,
        "gcs_paths": gcs_paths
    }
    message_data = json.dumps(message_payload).encode("utf-8")
    future = publisher.publish(topic_path, message_data)
    message_id = future.result()
    print(f"Published message to {CONSOLIDATOR_TOPIC_NAME} with ID: {message_id}, payload: {message_payload}")
    return message_id

def verify_spanner_data(spanner_database):
    """Verifies that the expected data exists in Spanner."""
    # Verify Entities
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

    # Verify Relationships
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

def test_end_to_end_consolidator_workflow(spanner_database, pubsub_publisher, gcs_setup):
    """
    End-to-end integration test for the consolidator function.
    1. Creates and uploads a dummy igraph object to GCS (handled by gcs_setup fixture).
    2. Triggers the consolidator via Pub/Sub with GCS path.
    3. Waits for consolidation and persistence to complete (by polling Spanner).
    4. Verifies the data in Spanner.
    """
    batch_id, gcs_paths = gcs_setup

    # Step 2: Trigger Consolidator via Pub/Sub
    publish_consolidator_start_message(pubsub_publisher, batch_id, gcs_paths)

    # Step 3: Wait for consolidation and persistence to complete (poll Spanner)
    max_retries = 20 
    retry_delay_seconds = 30 
    
    print(f"Waiting for consolidator and persistor to process data (max {max_retries} retries, {retry_delay_seconds}s delay)...")
    for i in range(max_retries):
        try:
            verify_spanner_data(spanner_database)
            print("Consolidator and Persistor workflow completed and verified.")
            return # Test passed
        except AssertionError as e:
            print(f"Attempt {i+1}/{max_retries}: Spanner verification failed: {e}")
            time.sleep(retry_delay_seconds)
    
    pytest.fail(f"Consolidator and Persistor did not process data correctly within {max_retries * retry_delay_seconds} seconds.")


def test_gcs_setup_fixture(gcs_setup):
    """
    Verifies that the gcs_setup fixture correctly uploads a dummy graph to GCS.
    This test primarily checks the setup part of the integration.
    """
    batch_id, gcs_paths = gcs_setup
    assert batch_id is not None
    assert len(gcs_paths) == 1
    assert gcs_paths[0].startswith("gs://")

    # Optionally, try to download the file to confirm it's there and valid
    storage_client = storage.Client()
    bucket_name = gcs_paths[0].split("//")[1].split("/")[0]
    blob_name = "/".join(gcs_paths[0].split("//")[1].split("/")[1:])
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    downloaded_blob = blob.download_as_bytes()
    assert downloaded_blob is not None
    # Try to unpickle to ensure it's a valid graph
    unpickled_graph = pickle.loads(downloaded_blob)
    assert isinstance(unpickled_graph, ig.Graph)
    assert unpickled_graph.vcount() > 0 # Ensure it's not an empty graph