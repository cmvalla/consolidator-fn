# Spanner operations for the consolidator function
import logging
import hashlib
import re
from google.api_core.exceptions import AlreadyExists, FailedPrecondition
from models import Entity, Relationship, InstanceOf, WorkflowStatus, Base
from sqlalchemy import func, text
from sqlalchemy.orm import sessionmaker

class SpannerOperations:
    def __init__(self, db_session, engine):
        self.db_session = db_session
        self.engine = engine

    def _table_exists(self, table_name):
        ReadOnlySession = sessionmaker(bind=self.engine)
        read_only_session = ReadOnlySession()
        try:
            query = text(f"SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{table_name}'")
            result = read_only_session.execute(query).scalar()
            return result == 1
        finally:
            read_only_session.close()

    def _index_exists(self, index_name):
        ReadOnlySession = sessionmaker(bind=self.engine)
        read_only_session = ReadOnlySession()
        try:
            query = text(f"SELECT 1 FROM INFORMATION_SCHEMA.INDEXES WHERE INDEX_NAME = '{index_name}'")
            result = read_only_session.execute(query).scalar()
            return result == 1
        finally:
            read_only_session.close()

    def _graph_exists(self, graph_name):
        ReadOnlySession = sessionmaker(bind=self.engine)
        read_only_session = ReadOnlySession()
        try:
            query = text(f"SELECT 1 FROM INFORMATION_SCHEMA.PROPERTY_GRAPH_METADATA WHERE PROPERTY_GRAPH_NAME = '{graph_name}'")
            result = read_only_session.execute(query).scalar()
            return result == 1
        finally:
            read_only_session.close()

    def ensure_spanner_schema(self):
        """
        Ensures that the necessary tables, graphs, and indexes exist in Spanner.
        Tables are created using SQLAlchemy ORM. Spanner-specific DDL (graphs, vector indexes)
        are executed conditionally.
        """
        logging.info("Ensuring Spanner schema...")

        # 1. Create tables using SQLAlchemy ORM
        try:
            Base.metadata.create_all(self.engine)
            logging.info("SQLAlchemy ORM tables ensured.")
        except Exception as e:
            logging.error(f"Error creating tables with SQLAlchemy ORM: {e}", exc_info=True)
            # Continue, as some tables might exist and we need to handle other DDL

        # 2. Execute Spanner-specific DDL from schema.sql conditionally
        try:
            with open("schema.sql", "r") as f:
                ddl_statements = [statement.strip() for statement in f.read().split(';') if statement.strip()]
        except FileNotFoundError:
            logging.error("schema.sql not found. Cannot ensure Spanner schema.")
            return

        for ddl in ddl_statements:
            ddl_type = None
            ddl_name = None

            # Try to parse CREATE TABLE (already handled by ORM, but might be in schema.sql)
            match = re.match(r"CREATE TABLE (\w+)", ddl, re.IGNORECASE)
            if match:
                ddl_type = "TABLE"
                ddl_name = match.group(1)
            
            # Try to parse CREATE VECTOR INDEX
            if not ddl_type:
                match = re.match(r"CREATE VECTOR INDEX (\w+)", ddl, re.IGNORECASE)
                if match:
                    ddl_type = "INDEX"
                    ddl_name = match.group(1)

            # Try to parse CREATE PROPERTY GRAPH
            if not ddl_type:
                match = re.match(r"CREATE PROPERTY GRAPH (\w+)", ddl, re.IGNORECASE)
                if match:
                    ddl_type = "GRAPH"
                    ddl_name = match.group(1)

            should_execute = True
            if ddl_type and ddl_name:
                if ddl_type == "TABLE" and self._table_exists(ddl_name):
                    logging.info(f"Table '{ddl_name}' already exists, skipping DDL.")
                    should_execute = False
                elif ddl_type == "INDEX" and self._index_exists(ddl_name):
                    logging.info(f"Index '{ddl_name}' already exists, skipping DDL.")
                    should_execute = False
                elif ddl_type == "GRAPH" and self._graph_exists(ddl_name):
                    logging.info(f"Property Graph '{ddl_name}' already exists, skipping DDL.")
                    should_execute = False
            
            if should_execute:
                try:
                    with self.db_session.begin():
                        self.db_session.execute(text(ddl))
                    logging.info(f"Successfully executed DDL: {ddl}")
                except (AlreadyExists, FailedPrecondition) as e:
                    logging.info(f"DDL statement already exists or failed precondition, skipping: {ddl} - Error: {e}")
                except Exception as e:
                    logging.error(f"Error executing DDL statement: {ddl}", exc_info=True)
            else:
                logging.info(f"DDL statement for '{ddl_name}' skipped as it already exists.")

    def migrate_to_spanner(self, data):
        """Migrates the final graph data to Cloud Spanner using SQLAlchemy."""
        logging.info("Migrating data to Spanner with SQLAlchemy...")
        
        entities = data.get("entities", [])
        relationships = data.get("relationships", [])

        valid_entities = [e for e in entities if e.get("id") and isinstance(e.get("id"), str) and e["id"].strip()]
        if len(valid_entities) != len(entities):
            logging.warning(f"Filtered out {len(entities) - len(valid_entities)} entities with invalid IDs.")

        valid_eids = {e["id"] for e in valid_entities}

        def is_valid_rel(r):
            source = r.get("source")
            target = r.get("target")
            return source and isinstance(source, str) and source.strip() and \
                   target and isinstance(target, str) and target.strip() and \
                   source in valid_eids and target in valid_eids

        valid_relationships = [r for r in relationships if is_valid_rel(r)]
        if len(valid_relationships) != len(relationships):
            logging.warning(f"Filtered out {len(relationships) - len(valid_relationships)} relationships with invalid or dangling EIDs.")

        try:
            with self.db_session.begin():
                for e in valid_entities:
                    entity = Entity(
                        Eid=e["id"],
                        Type=e["type"],
                        Properties=e.get("properties", {}),
                        Embedding=e.get("embedding", []),
                        Communities=e.get("communities", [])
                    )
                    self.db_session.merge(entity)
                logging.info(f"Merged {len(valid_entities)} entities.")

                rels_to_upsert = [
                    Relationship(
                        Rid=hashlib.sha256(f"{r['source']}-{r['target']}-{r.get('type')}".encode()).hexdigest(),
                        SourceEid=r["source"],
                        TargetEid=r["target"],
                        Type=r.get("type"),
                        Properties=r.get("properties", {})
                    )
                    for r in valid_relationships if r.get('type') != 'INSTANCE_OF'
                ]
                for rel in rels_to_upsert:
                    self.db_session.merge(rel)
                logging.info(f"Merged {len(rels_to_upsert)} relationships.")

                instance_of_to_upsert = [
                    InstanceOf(
                        InstanceEid=r["source"],
                        ClassEid=r["target"]
                    )
                    for r in valid_relationships if r.get('type') == 'INSTANCE_OF'
                ]
                for inst in instance_of_to_upsert:
                    self.db_session.merge(inst)
                logging.info(f"Merged {len(instance_of_to_upsert)} instance-of relationships.")

            logging.info("Successfully committed all changes to Spanner.")

        except Exception as e:
            logging.error(f"Error during Spanner session commit: {e}", exc_info=True)
            self.db_session.rollback()
            raise e

        return data

    def update_workflow_status(self, batch_id, status):
        try:
            with self.db_session.begin():
                workflow_status = self.db_session.query(WorkflowStatus).filter_by(BatchId=batch_id).first()
                if workflow_status:
                    workflow_status.Status = status
                    workflow_status.UpdatedAt = func.now()
                else:
                    workflow_status = WorkflowStatus(BatchId=batch_id, Status=status)
                    self.db_session.add(workflow_status)
            logging.info(f"Successfully updated workflow status for batch ID {batch_id} to {status}.")
        except Exception as e:
            logging.error(f"Could not update workflow status for batch ID {batch_id} to {status}: {e}", exc_info=True)