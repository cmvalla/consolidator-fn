import pytest
from unittest.mock import Mock, patch, mock_open, MagicMock

from spanner_operations import SpannerOperations

@pytest.fixture
def mock_db_session():
    session = MagicMock()
    transaction = MagicMock()
    session.begin.return_value = transaction
    transaction.__enter__.return_value = transaction
    transaction.__exit__.return_value = None
    return session

def test_ensure_spanner_schema(mock_db_session):
    spanner_ops = SpannerOperations(mock_db_session)
    
    with patch("builtins.open", mock_open(read_data="CREATE TABLE test;")):
        spanner_ops.ensure_spanner_schema()

    mock_db_session.begin.assert_called()

def test_migrate_to_spanner(mock_db_session):
    spanner_ops = SpannerOperations(mock_db_session)
    data = {
        "entities": [{"id": "e1", "type": "test", "properties": {}, "embedding": [], "communities": []}],
        "relationships": []
    }

    spanner_ops.migrate_to_spanner(data)

    mock_db_session.begin.assert_called()
    mock_db_session.merge.assert_called()

def test_update_workflow_status(mock_db_session):
    spanner_ops = SpannerOperations(mock_db_session)
    batch_id = "test_batch_123"
    status = "SUCCEEDED"

    spanner_ops.update_workflow_status(batch_id, status)

    mock_db_session.begin.assert_called()
    mock_db_session.query.assert_called()
