import os
import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from main import consolidator
import main

# --- Test Configuration ---
CREDENTIALS_FILE = os.path.join(os.path.dirname(__file__), "credentials.json")

@pytest.fixture(scope="module", autouse=True)
def set_google_credentials():
    """Fixture to set the GOOGLE_APPLICATION_CREDENTIALS environment variable."""
    if not os.path.exists(CREDENTIALS_FILE):
        pytest.fail(
            f"Service account key file not found at '{CREDENTIALS_FILE}'. "
            "Please create the file as per the instructions."
        )
    
    original_value = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_FILE
    
    yield
    
    # Teardown: Restore original environment variable
    if original_value is None:
        del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    else:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = original_value

@patch("main.initialize_clients")
@patch("main.redis.Redis")
@patch("main.VertexAI")
@patch("main.MemgraphGraph")
@patch("main.LLMChain")
def test_consolidator_local_integration(MockLLMChain, MockMemgraph, MockVertexAI, MockRedis, MockInitializeClients):
    """
    Tests the consolidator function locally.
    """
    # --- Mock the external clients ---
    mock_redis_client = Mock()
    mock_llm = Mock()
    mock_memgraph_graph = Mock()
    main.redis_client = mock_redis_client
    main.llm = mock_llm
    main.memgraph_graph = mock_memgraph_graph


    # --- Mock the input data ---
    mock_cloud_event = Mock()
    mock_cloud_event.data = {
        "message": {
            "data": b'eyJiYXRjaF9pZCI6ICJ0ZXN0X2JhdGNoXzEyMyJ9'
        }
    }

    # --- Mock the data returned by Redis ---
    mock_redis_client.lrange.return_value = [
        json.dumps({
            "entities": [
                {"id": "1", "type": "Person", "properties": {"name": "Bill Gates"}},
                {"id": "2", "type": "Person", "properties": {"name": "William Henry Gates III"}}
            ],
            "relationships": []
        }),
        json.dumps({
            "entities": [
                {"id": "3", "type": "Organization", "properties": {"name": "Microsoft"}}
            ],
            "relationships": [
                {"source": "1", "target": "3", "type": "FOUNDED"}
            ]
        })
    ]

    # --- Mock the LLMChain object ---
    mock_chain = Mock()
    mock_chain.run.return_value = json.dumps({
        "1": "1",
        "2": "1",
        "3": "3"
    })
    MockLLMChain.return_value = mock_chain

    # --- Mock the MemgraphGraph object ---
    mock_memgraph_graph.query.return_value = []

    # --- Invoke the consolidator function ---
    response, status_code = consolidator(mock_cloud_event)

    # --- Assertions ---
    assert status_code == 200
    assert response == "OK"

    # Assert that Redis was called correctly
    mock_redis_client.lrange.assert_called_once_with("batch:test_batch_123:results", 0, -1)

    # Assert that the LLM was called for clustering
    mock_chain.run.assert_called_once()

    # Assert that Memgraph was called with the clustered data
    mock_memgraph_graph.query.assert_any_call(
        'UNWIND $nodes AS node CREATE (:Entity {id: node.id, type: node.type, properties: node.properties})',
        params={'nodes': [
            {
                'id': '1',
                'type': 'MergedEntity',
                'properties': {'merged_entities': [
                    {'id': '1', 'type': 'Person', 'properties': {'name': 'Bill Gates'}},
                    {'id': '2', 'type': 'Person', 'properties': {'name': 'William Henry Gates III'}}
                ]}
            },
            {
                'id': '3',
                'type': 'MergedEntity',
                'properties': {'merged_entities': [
                    {'id': '3', 'type': 'Organization', 'properties': {'name': 'Microsoft'}}
                ]}
            }
        ]}
    )
    mock_memgraph_graph.query.assert_any_call(
        'UNWIND $rels AS rel MATCH (a:Entity {id: rel.source}), (b:Entity {id: rel.target}) CREATE (a)-[:RELATIONSHIP {type: rel.type, properties: rel.properties}]->(b)',
        params={'rels': [
            {'source': '1', 'target': '3', 'type': 'FOUNDED', 'properties': None}
        ]}
    )

if __name__ == "__main__":
    pytest.main([__file__])
