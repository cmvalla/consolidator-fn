# Redis operations for the consolidator function
import json
import logging

class RedisOperations:
    def __init__(self, redis_client):
        self.redis_client = redis_client

    def fetch_from_redis(self, data):
        batch_id = data["batch_id"]
        results_key = f"batch:{batch_id}:results"
        partial_results_str = self.redis_client.lrange(results_key, 0, -1)
        logging.info(f"Fetched {len(partial_results_str)} partial results from Redis for batch {batch_id}.")
        return {"batch_id": batch_id, "partial_results": partial_results_str}

    def store_consolidated_results_in_redis(self, data):
        """Stores the consolidated entities and relationships in Redis."""
        batch_id = data["batch_id"]
        consolidated_key = f"consolidated_batch:{batch_id}"
        
        try:
            self.redis_client.hset(consolidated_key, mapping={
                "entities": json.dumps(data["entities"]),
                "relationships": json.dumps(data["relationships"])
            })
            self.redis_client.expire(consolidated_key, 86400)
            logging.info(f"Stored consolidated results for batch {batch_id} in Redis.")
        except Exception as e:
            logging.error(f"Error storing consolidated results for batch {batch_id} in Redis: {e}", exc_info=True)
        return data