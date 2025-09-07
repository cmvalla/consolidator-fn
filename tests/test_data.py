# Test data for the consolidator function

redis_data = [
    '{"batch_id": "test_batch_123", "chunk_number": 0, "extracted_graph_data": {"entities": [{"id": "lupo_1", "type": "Character", "name": "lupo"}], "relationships": [{"source": "lupo_1", "target": "cappuccetto_rosso_1", "type": "divorò"}]}}',
    '{"batch_id": "test_batch_123", "chunk_number": 1, "extracted_graph_data": {"entities": [{"id": "cappuccetto_rosso_1", "type": "Character", "name": "cappuccetto rosso"}], "relationships": []}}',
    '{"batch_id": "test_batch_123", "chunk_number": 2, "extracted_graph_data": {"entities": [{"id": "nonna_1", "type": "Character", "name": "nonna"}], "relationships": [{"source": "lupo_1", "target": "nonna_1", "type": "mangiò"}]}}'
]