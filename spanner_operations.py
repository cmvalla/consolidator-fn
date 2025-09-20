
import logging
from datetime import datetime, timezone
from google.cloud.spanner_v1 import Client, KeySet, COMMIT_TIMESTAMP
from google.cloud.spanner_v1.transaction import Transaction
from google.api_core.exceptions import Aborted, NotFound

class SpannerOperations:
    def __init__(self, spanner_client: Client, instance_id: str, database_id: str):
        self.spanner_client = spanner_client
        self.instance_id = instance_id
        self.database_id = database_id
        self.instance = self.spanner_client.instance(self.instance_id)
        self.database = self.instance.database(self.database_id)

    def acquire_lock(self, batch_id: str, instance_id: str) -> bool:
        """
        Acquires a lock for a given batch_id in the WorkflowStatus table.
        Returns True if the lock was acquired, False otherwise.
        """
        def _acquire_lock_in_transaction(transaction: Transaction) -> bool:
            logging.info(f"Acquiring lock for batch_id: {batch_id}, instance_id: {instance_id}")
            try:
                result = transaction.read(
                    table="WorkflowStatus",
                    keyset=KeySet(keys=[[batch_id]]),
                    columns=["status", "lock_owner", "lock_timestamp"],
                )
                row = next(iter(result), None)
                logging.info(f"Read result for batch_id {batch_id}: {row}")

                if row is None:
                    # No lock exists, acquire it
                    logging.info(f"No existing lock for batch {batch_id}. Acquiring new lock.")
                    transaction.insert(
                        table="WorkflowStatus",
                        columns=["batch_id", "status", "lock_owner", "lock_timestamp"],
                        values=[[batch_id, "PROCESSING", instance_id, COMMIT_TIMESTAMP]],
                    )
                    return True
                
                status, lock_owner, lock_timestamp = row
                
                logging.info(f"Current status for batch {batch_id}: status={status}, lock_owner={lock_owner}, lock_timestamp={lock_timestamp}")
                if status == "COMPLETED":
                    logging.info(f"Batch {batch_id} has already been processed.")
                    return False

                if status == "PENDING_CONSOLIDATION":
                    logging.info(f"Batch {batch_id} is pending consolidation. Acquiring lock.")
                    transaction.update(
                        table="WorkflowStatus",
                        columns=["batch_id", "status", "lock_owner", "lock_timestamp"],
                        values=[[batch_id, "PROCESSING", instance_id, COMMIT_TIMESTAMP]],
                    )
                    return True

                if status == "PROCESSING":
                    # Check for lock expiration
                    # This is a simplified example, you might want to use a more robust time comparison
                    if lock_timestamp and (datetime.now(timezone.utc) - lock_timestamp).total_seconds() > 300:
                        logging.warning(f"Lock for batch {batch_id} has expired. Stealing lock.")
                        transaction.update(
                            table="WorkflowStatus",
                            columns=["batch_id", "status", "lock_owner", "lock_timestamp"],
                            values=[[batch_id, "PROCESSING", instance_id, COMMIT_TIMESTAMP]],
                        )
                        return True
                    else:
                        logging.info(f"Batch {batch_id} is already being processed by another instance.")
                        return False

                if status == "FAILED":
                    logging.warning(f"Batch {batch_id} previously failed. Retrying.")
                    transaction.update(
                        table="WorkflowStatus",
                        columns=["batch_id", "status", "lock_owner", "lock_timestamp"],
                        values=[[batch_id, "PROCESSING", instance_id, COMMIT_TIMESTAMP]],
                    )
                    return True

                return False

            except NotFound:
                # No lock exists, acquire it
                transaction.insert(
                    table="WorkflowStatus",
                    columns=["batch_id", "status", "lock_owner", "lock_timestamp"],
                    values=[[batch_id, "PROCESSING", instance_id, COMMIT_TIMESTAMP]],
                )
                return True

        try:
            return self.database.run_in_transaction(_acquire_lock_in_transaction)
        except Aborted:
            logging.warning(f"Transaction aborted while trying to acquire lock for batch {batch_id}. Retrying might be necessary.")
            return False

    def release_lock(self, batch_id: str, status: str):
        """
        Releases the lock for a given batch_id in the WorkflowStatus table.
        """
        def _release_lock_in_transaction(transaction: Transaction):
            transaction.update(
                table="WorkflowStatus",
                columns=["batch_id", "status", "lock_owner", "lock_timestamp"],
                values=[[batch_id, status, None, None]],
            )

        try:
            self.database.run_in_transaction(_release_lock_in_transaction)
        except Aborted:
            logging.error(f"Transaction aborted while trying to release lock for batch {batch_id}.")
            # Handle the aborted transaction, e.g., by retrying.
