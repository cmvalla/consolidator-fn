import base64
import json
import re
import os
import functions_framework
import google.cloud.logging
import logging
import redis
from langchain_community.graphs import MemgraphGraph
from langchain_google_vertexai import VertexAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
import google.cloud.secretmanager as secretmanager
import time
import google.cloud.spanner_v1 as spanner
import psutil
from google.api_core.exceptions import AlreadyExists, FailedPrecondition

# --- Boilerplate and Configuration ---
logging_client = google.cloud.logging.Client()
logging_client.setup_logging()
logging.basicConfig(level=logging.INFO)

# --- Global Clients (initialized within the function) ---
redis_client = None
llm = None
memgraph_graph = None
spanner_client = None
spanner_instance = None
spanner_database = None

def ensure_spanner_graph_exists(database):
    """
    Unconditionally attempts to create the 'my-graph' Spanner Property Graph.
    If the graph already exists, it will fail gracefully.
    This approach bypasses any potential issues with querying information_schema.
    """
    try:
        logging.info("Attempting to create Spanner property graph 'my_graph'...")
        
        ddl_statement = """
        CREATE PROPERTY GRAPH my_graph
            NODE TABLES (
                Entities,
                Communities
            )
            EDGE TABLES (
                Relationships
                    SOURCE KEY (EntityId) REFERENCES Entities (EntityId)
                    DESTINATION KEY (TargetEntityId) REFERENCES Entities (EntityId),
                EntityCommunity
                    SOURCE KEY (EntityId) REFERENCES Entities (EntityId)
                    DESTINATION KEY (CommunityId) REFERENCES Communities (CommunityId)
            )
        """
        
        operation = database.update_ddl([ddl_statement])
        
        logging.info("Waiting for 'CREATE PROPERTY GRAPH' operation to complete... (this may take a few minutes)")
        operation.result()  # Blocks until the operation is done
        logging.info("Spanner property graph 'my_graph' created successfully.")

    except (AlreadyExists, FailedPrecondition) as e:
        if "Duplicate name in schema" in str(e) or "already exists" in str(e).lower():
            logging.warning("Spanner property graph 'my_graph' already exists. Continuing.")
        else:
            logging.error(f"Failed to create Spanner property graph 'my_graph': {e}", exc_info=True)
            raise

def initialize_clients():
    """Initializes all external clients."""
    global redis_client, llm, memgraph_graph, spanner_client, spanner_instance, spanner_database

    try:
        # Log system resource usage
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        logging.info(f"System CPU Usage: {cpu_usage}%")
        logging.info(f"System Memory: Total={memory_info.total / 1024**3:.2f}GB, Available={memory_info.available / 1024**3:.2f}GB, Used={memory_info.used / 1024**3:.2f}GB, Percentage={memory_info.percent}%")
        
        logging.info("Initializing global clients...")

        # --- Environment Variables ---
        GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
        REDIS_HOST = os.environ.get("REDIS_HOST")
        REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
        MEMGRAPH_HOST = os.environ.get("MEMGRAPH_HOST", "memgraph-service.memgraph.svc.cluster.local")
        MEMGRAPH_PORT = int(os.environ.get("MEMGRAPH_PORT", 7687))
        MEMGRAPH_USER = os.environ.get("MEMGRAPH_USER", "memgraph")
        MEMGRAPH_PASSWORD = os.environ.get("MEMGRAPH_PASSWORD")
        SPANNER_INSTANCE_ID = os.environ.get("SPANNER_INSTANCE_ID")
        SPANNER_DATABASE_ID = os.environ.get("SPANNER_DATABASE_ID")
        LOCATION = os.environ.get("LOCATION")
        
        # --- Secret Manager ---
        sm_client = secretmanager.SecretManagerServiceClient()
        REDIS_PASSWORD = sm_client.access_secret_version(request={"name": f"projects/{GCP_PROJECT}/secrets/redis-password/versions/latest"}).payload.data.decode("UTF-8")

        logging.info("Initializing Redis client...")
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, ssl=False, ssl_cert_reqs=None, decode_responses=True, socket_connect_timeout=10)
        redis_client.ping()
        logging.info("Redis client initialized successfully.")

        logging.info("Initializing Memgraph client...")
        os.environ["NEO4J_USERNAME"] = MEMGRAPH_USER
        os.environ["NEO4J_PASSWORD"] = MEMGRAPH_PASSWORD
        memgraph_graph = MemgraphGraph(url=f"bolt://{MEMGRAPH_HOST}:{MEMGRAPH_PORT}", username=MEMGRAPH_USER, password=MEMGRAPH_PASSWORD)
        logging.info("Memgraph client initialized successfully.")

        logging.info("Initializing Vertex AI...")
        llm = VertexAI(model_name="gemini-2.5-flash", location=LOCATION)
        logging.info("Vertex AI clients initialized successfully.")

        logging.info("Initializing Spanner client...")
        spanner_client = spanner.Client(project=GCP_PROJECT)
        spanner_instance = spanner_client.instance(SPANNER_INSTANCE_ID)
        spanner_database = spanner_instance.database(SPANNER_DATABASE_ID)
        logging.info("Spanner client initialized successfully.")

        # Ensure the Spanner Graph exists
        ensure_spanner_graph_exists(spanner_database)

        logging.info("All global clients initialized successfully.")
    except Exception as e:
        logging.critical(f'FATAL: Failed to initialize one or more global clients: {e}', exc_info=True)
        raise  # Re-raise the exception to halt execution if initialization fails

# --- LangChain Runnables ---
def decode_pubsub_message(cloud_event):
    message_data = base64.b64decode(cloud_event.data["message"]["data"]).decode("utf-8")
    message_json = json.loads(message_data)
    return {"batch_id": message_json.get("batch_id")}

def fetch_from_redis(data):
    batch_id = data["batch_id"]
    results_key = f"batch:{batch_id}:results"
    partial_results_str = redis_client.lrange(results_key, 0, -1)
    return {"batch_id": batch_id, "partial_results": partial_results_str}

def aggregate_results(data):
    all_entities = {}
    all_relationships = []
    for res_str in data["partial_results"]:
        res_json = json.loads(res_str)
        logging.info(f"DEBUG: Processing res_json: {res_json}") # Added debug log
        for entity in res_json.get("extracted_graph_data", {}).get("entities", []): # Corrected path to entities
            entity_id = None
            # Check for 'id' in a case-insensitive manner
            for key in entity.keys():
                if key.lower() == "id":
                    entity_id = entity[key]
                    break
            
            if entity_id: # Check if entity_id was found and is not None/empty
                all_entities[entity_id] = entity
            else:
                logging.warning(f"Skipping entity without id: {entity}")
        all_relationships.extend(res_json.get("extracted_graph_data", {}).get("relationships", [])) # Corrected path to relationships
    
    logging.info(f"Aggregated {len(all_entities)} entities and {len(all_relationships)} relationships.")

    return {
        "batch_id": data["batch_id"],
        "entities": list(all_entities.values()),
        "relationships": all_relationships
    }

def cluster_and_merge_entities(data):
    """
    Clusters similar entities based on name similarity using an LLM.
    This approach is inspired by the clustering feature in the kg-gen repository:
    https://github.com/stair-lab/kg-gen
    """
    entities = data.get("entities", [])
    relationships = data.get("relationships", [])

    if not entities:
        return data

    # Create a prompt for the LLM to cluster entities.
    prompt = PromptTemplate(
        input_variables=["entities"],
        template="""
        From the following list of entities, identify entities that refer to the same real-world object and group them.
        For each group, choose one canonical entity and provide a mapping from the old entity IDs to the new canonical entity ID.

        Example:
        Input:
        [
            {{"id": "1", "type": "Person", "properties": {{"name": "Bill Gates"}}}},
            {{"id": "2", "type": "Person", "properties": {{"name": "William Henry Gates III"}}}},
            {{"id": "3", "type": "Organization", "properties": {{"name": "Microsoft"}}}}
        ]

        Output:
        {{
            "1": "1",
            "2": "1",
            "3": "3"
        }}

        Entities:
        {entities}

        Mapping:
        """
    )

    # Create a chain to run the LLM.
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the chain.
    entity_list_str = json.dumps(entities, indent=2)
    llm_response = chain.run(entities=entity_list_str)

    try:
        # Attempt to extract JSON from markdown code block if present
        if llm_response.startswith("```json") and llm_response.endswith("```"):
            llm_response = llm_response.lstrip("```json").rstrip("```").strip()
        
        # Use regex to find the JSON object, robust to leading/trailing text
        json_match = re.search(r"\{.*\}", llm_response, re.DOTALL)
        if json_match:
            llm_response = json_match.group(0)
        else:
            logging.error(f"No JSON object found in LLM response: {llm_response}")
            return data # or handle the error in a more sophisticated way

        entity_id_map = json.loads(llm_response)
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON from LLM response: {llm_response}")
        return data # or handle the error in a more sophisticated way

    # Create new entities and relationships.
    new_entities = {}
    for entity in entities:
        canonical_id = entity_id_map.get(entity["id"])
        if canonical_id:
            if canonical_id not in new_entities:
                new_entities[canonical_id] = {
                    "id": canonical_id,
                    "type": "MergedEntity",
                    "properties": {"merged_entities": []}
                }
            new_entities[canonical_id]["properties"]["merged_entities"].append(entity)

    updated_relationships = []
    for rel in relationships:
        source_id = entity_id_map.get(rel.get("source"))
        target_id = entity_id_map.get(rel.get("target"))
        if source_id and target_id:
            updated_relationships.append({
                "source": source_id,
                "target": target_id,
                "type": rel.get("type"),
                "properties": rel.get("properties")
            })

    data["entities"] = list(new_entities.values())
    data["relationships"] = updated_relationships
    
    logging.info(f"Clustered {len(entities)} entities into {len(new_entities)} merged entities.")

    return data

def load_to_memgraph(data):
    entities = data.get("entities", [])
    if entities:
        logging.info(f"Loading {len(entities)} entities to Memgraph.")
        node_query = "UNWIND $nodes AS node CREATE (:Entity {id: node.id, type: node.type, properties: node.properties})"
        memgraph_graph.query(node_query, params={'nodes': entities})
    relationships = data.get("relationships", [])
    if relationships:
        logging.info(f"Loading {len(relationships)} relationships to Memgraph.")
        rel_query = "UNWIND $rels AS rel MATCH (a:Entity {id: rel.source}), (b:Entity {id: rel.target}) CREATE (a)-[:RELATIONSHIP {type: rel.type, properties: rel.properties, weight: rel.properties.weight}]->(b)" # Use actual weight
        memgraph_graph.query(rel_query, params={'rels': relationships})
        # Add debug logs for relationships and weights
        logging.info(f"DEBUG: Sample of relationships loaded to Memgraph: {relationships[:5]}") # Log first 5 relationships
    return data

def run_community_detection(data):
    """
    Performs hierarchical community detection using the Leiden algorithm in Memgraph.
    """
    try:
        # Log Memgraph counts immediately before running community detection
        node_count_before_leiden = memgraph_graph.query("MATCH (n) RETURN count(n) AS count")[0]['count']
        rel_count_before_leiden = memgraph_graph.query("MATCH ()-[r]->() RETURN count(r) AS count")[0]['count']
        logging.info(f"DEBUG: Memgraph contains {node_count_before_leiden} nodes and {rel_count_before_leiden} relationships BEFORE Leiden query.")

        community_query = "CALL leiden_community_detection.get() YIELD node, community_id, communities"
        logging.info(f"DEBUG: Community detection query: {community_query}") # Log query
        result = memgraph_graph.query(community_query)
        
        logging.info(f"DEBUG: Raw result from Memgraph query: {result}") # Log raw result
        if not result:
            logging.warning("Community detection returned no results.")
            logging.info(f"DEBUG: Community detection result (empty): {result}") # Log empty result
            result = []
        else:
            logging.info(f"DEBUG: Community detection result (sample): {result[:5]}") # Log sample of result

    except Exception as e:
        logging.error(f"Error running community detection: {e}", exc_info=True)
        result = []

    for record in result:
        if record and 'node' in record and record['node']:
            node_id = record["node"].get("id")
            communities = record.get("communities")

            if node_id is not None and communities is not None:
                try:
                    # Store the full hierarchy on the entity node
                    memgraph_graph.query("MATCH (e:Entity {id: $node_id}) SET e.communities = $communities", 
                                         params={"node_id": node_id, "communities": communities})

                    # Create relationships to community nodes at each level
                    for i, community_id in enumerate(communities):
                        level_community_id = f"level_{i}_community_{community_id}"
                        memgraph_graph.query("MERGE (c:Community {id: $community_id})", 
                                             params={"community_id": level_community_id})
                        memgraph_graph.query("MATCH (e:Entity {id: $node_id}), (c:Community {id: $community_id}) CREATE (e)-[:BELONGS_TO]->(c)", 
                                             params={"node_id": node_id, "community_id": level_community_id})

                except Exception as e:
                    logging.error(f"Error creating community relationship for node {node_id}: {e}", exc_info=True)
            else:
                logging.warning(f"Skipping record due to missing 'node_id' or 'communities': {record}")
        else:
            logging.warning(f"Skipping invalid record in community detection result: {record}")
            
    return data

def get_community_hierarchy(memgraph_graph):
    """
    Queries Memgraph to reconstruct the community hierarchy.
    Returns a dictionary representing the hierarchy.
    """
    hierarchy = {}
    result = memgraph_graph.query("MATCH (e:Entity) RETURN e.id AS entity_id, e.communities AS communities")
    for record in result:
        entity_id = record["entity_id"]
        communities = record["communities"]
        if communities:
            for i, community_id in enumerate(communities):
                level = f"level_{i}"
                if level not in hierarchy:
                    hierarchy[level] = {}
                if community_id not in hierarchy[level]:
                    hierarchy[level][community_id] = {"entities": [], "children": set()}
                hierarchy[level][community_id]["entities"].append(entity_id)
                if i > 0:
                    parent_community_id = communities[i-1]
                    hierarchy[f"level_{i-1}"][parent_community_id]["children"].add(community_id)
    return hierarchy

COMMUNITY_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["community_id", "entities", "relationships", "children_summaries"],
    template='''
    Here is a summary of the community {community_id}:

    Entities:
    {entities}

    Relationships:
    {relationships}

    Children Summaries:
    {children_summaries}

    Please provide a summary of this community.
    '''
)

def generate_hierarchical_summaries(data):
    """
    Generates summaries for each community in the hierarchy, from the bottom up.
    """
    hierarchy = get_community_hierarchy(memgraph_graph)
    summaries = {} # To store summaries of all communities

    if not hierarchy:
        data["summaries"] = []
        return data

    # Generate summaries for leaf communities (deepest level)
    deepest_level_num = max([int(level.split('_')[1]) for level in hierarchy.keys()])
    deepest_level = f"level_{deepest_level_num}"

    for community_id, community_data in hierarchy[deepest_level].items():
        entity_ids = community_data["entities"]
        
        # Get entities and relationships from Memgraph
        # Summarize entities and relationships to reduce token count for LLM
        entities = memgraph_graph.query(f"MATCH (e:Entity) WHERE e.id IN {entity_ids} RETURN e.id AS id, e.type AS type, e.properties.name AS name")
        relationships = memgraph_graph.query(f"MATCH (e1:Entity)-[r:RELATIONSHIP]->(e2:Entity) WHERE e1.id IN {entity_ids} AND e2.id IN {entity_ids} RETURN e1.id AS source_id, e2.id AS target_id, r.type AS type")

        # Create concise string representations of entities and relationships
        concise_entities = [f"({e['id']}:{e['type']} - {e.get('name', 'N/A')})" for e in entities]
        concise_relationships = [f"({r['source_id']})-[:{r['type']}]->({r['target_id']})" for r in relationships]

        summary_prompt = COMMUNITY_SUMMARY_PROMPT.format(
            community_id=community_id,
            entities="\n".join(concise_entities),
            relationships="\n".join(concise_relationships),
            children_summaries=""
        )
        summary = llm.invoke(summary_prompt)
        summaries[f"level_{deepest_level_num}_community_{community_id}"] = summary

    # Generate summaries for upper levels
    for i in range(deepest_level_num - 1, -1, -1):
        level = f"level_{i}"
        for community_id, community_data in hierarchy[level].items():
            child_summaries = [summaries[f"level_{i+1}_community_{child_id}"] for child_id in community_data["children"]]
            
            # Get entities and relationships from Memgraph
            entity_ids = community_data["entities"]
            entities = memgraph_graph.query(f"MATCH (e:Entity) WHERE e.id IN {entity_ids} RETURN e.properties AS props")
            relationships = memgraph_graph.query(f"MATCH (e1:Entity)-[r:RELATIONSHIP]->(e2:Entity) WHERE e1.id IN {entity_ids} AND e2.id IN {entity_ids} RETURN r.type AS type, r.properties AS props")

            # Create concise string representations of entities and relationships for upper levels
            concise_entities_upper = [f"({e.get('id', 'N/A')}:{e.get('type', 'N/A')} - {e.get('properties', {}).get('name', 'N/A')})" for e in entities]
            concise_relationships_upper = [f"({r.get('source_id', 'N/A')})-[:{r.get('type', 'N/A')}]->({r.get('target_id', 'N/A')})" for r in relationships]

            # Limit children summaries to prevent token limit issues
            limited_child_summaries = child_summaries[:5] # Take only the first 5 summaries

            summary_prompt = COMMUNITY_SUMMARY_PROMPT.format(
                community_id=community_id,
                entities="\n".join(concise_entities_upper),
                relationships="\n".join(concise_relationships_upper),
                children_summaries="\n".join(limited_child_summaries)
            )
            summary = llm.invoke(summary_prompt)
            summaries[f"level_{i}_community_{community_id}"] = summary
    
    data["summaries"] = [{"community_id": key, "summary": value} for key, value in summaries.items()]
    return data

def store_summaries(data):
    for summary_data in data["summaries"]:
        memgraph_graph.query("MATCH (c:Community {id: $community_id}) SET c.summary = $summary", params=summary_data)
    return data

def cleanup_redis(data):
    batch_id = data["batch_id"]
    redis_client.delete(f"batch:{batch_id}:results", f"batch:{batch_id}:counter")
    return data

def migrate_to_spanner(data):
    logging.info("Migrating data to Spanner...")

    # Extract data from Memgraph
    entities_memgraph = memgraph_graph.query("MATCH (e:Entity) RETURN e.id AS EntityId, e.type AS Type, e.properties AS Properties")
    relationships_memgraph = memgraph_graph.query("MATCH (s:Entity)-[r]->(t:Entity) RETURN s.id AS EntityId, id(r) as RelationshipId, t.id AS TargetEntityId, r.type AS Type, r.properties AS Properties")
    communities_memgraph = memgraph_graph.query("MATCH (c:Community) RETURN c.id AS CommunityId, c.summary AS Summary, c.embedding AS Embedding")
    entity_community_memgraph = memgraph_graph.query("MATCH (e:Entity)-[:BELONGS_TO]->(c:Community) RETURN e.id AS EntityId, c.id AS CommunityId")

    # Prepare data for Spanner
    entities_to_insert = []
    for entity in entities_memgraph:
        entities_to_insert.append((entity["EntityId"], entity["Type"], json.dumps(entity["Properties"])))

    relationships_to_insert = []
    for rel in relationships_memgraph:
        relationships_to_insert.append((rel["EntityId"], str(rel["RelationshipId"]), rel["TargetEntityId"], rel["Type"], json.dumps(rel.get("Properties", {}))))

    communities_to_insert = []
    for community in communities_memgraph:
        communities_to_insert.append((community["CommunityId"], community["Summary"], community.get("Embedding")))

    entity_community_to_insert = []
    for ec in entity_community_memgraph:
        entity_community_to_insert.append((ec["EntityId"], ec["CommunityId"]))

    logging.info(f"Migrating {len(entities_to_insert)} entities, {len(relationships_to_insert)} relationships, {len(communities_to_insert)} communities, and {len(entity_community_to_insert)} entity-community relationships to Spanner.")

    # Only run transaction if there is data to insert
    if not any([entities_to_insert, relationships_to_insert, communities_to_insert, entity_community_to_insert]):
        logging.info("No new data to migrate to Spanner.")
        return data

    # Define batch size
    BATCH_SIZE = 5000 # Spanner limit is 80000 mutations per transaction

    # Helper to chunk a list
    def chunk_list(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    # Load data into Spanner in batches
    for batch in chunk_list(entities_to_insert, BATCH_SIZE):
        def insert_entities_batch(transaction):
            transaction.insert_or_update(
                table="Entities",
                columns=("EntityId", "Type", "Properties"),
                values=batch,
            )
        spanner_database.run_in_transaction(insert_entities_batch)
        logging.info(f"Inserted {len(batch)} entities into Spanner.")

    for batch in chunk_list(relationships_to_insert, BATCH_SIZE):
        def insert_relationships_batch(transaction):
            transaction.insert_or_update(
                table="Relationships",
                columns=("EntityId", "RelationshipId", "TargetEntityId", "Type", "Properties"),
                values=batch,
            )
        spanner_database.run_in_transaction(insert_relationships_batch)
        logging.info(f"Inserted {len(batch)} relationships into Spanner.")

    for batch in chunk_list(communities_to_insert, BATCH_SIZE):
        def insert_communities_batch(transaction):
            transaction.insert_or_update(
                table="Communities",
                columns=("CommunityId", "Summary", "Embedding"),
                values=batch,
            )
        spanner_database.run_in_transaction(insert_communities_batch)
        logging.info(f"Inserted {len(batch)} communities into Spanner.")

    for batch in chunk_list(entity_community_to_insert, BATCH_SIZE):
        def insert_entity_community_batch(transaction):
            transaction.insert_or_update(
                table="EntityCommunity",
                columns=("EntityId", "CommunityId"),
                values=batch,
            )
        spanner_database.run_in_transaction(insert_entity_community_batch)
        logging.info(f"Inserted {len(batch)} entity-community relationships into Spanner.")

    logging.info("Data migrated to Spanner successfully.")
    return data

def log_redis_counts(data):
    """Logs the number of partial results fetched from Redis."""
    partial_results_count = len(data.get("partial_results", []))
    logging.info(f"Fetched {partial_results_count} partial results from Redis.")
    return data

def log_memgraph_counts(data):
    """Logs the number of nodes and relationships in Memgraph."""
    try:
        node_count_result = memgraph_graph.query("MATCH (n) RETURN count(n) AS count")
        node_count = node_count_result[0]['count'] if node_count_result else 0
        
        rel_count_result = memgraph_graph.query("MATCH ()-[r]->() RETURN count(r) AS count")
        rel_count = rel_count_result[0]['count'] if rel_count_result else 0
        
        logging.info(f"Memgraph contains {node_count} nodes and {rel_count} relationships.")
    except Exception as e:
        logging.error(f"Error counting objects in Memgraph: {e}", exc_info=True)
    return data

def log_spanner_counts(data):
    """Queries and logs the row counts from Spanner tables."""
    logging.info("Querying Spanner for row counts...")
    tables_to_query = ["Entities", "Communities", "Relationships", "EntityCommunity"]
    
    try:
        for table in tables_to_query:
            with spanner_database.snapshot() as snapshot:
                try:
                    results = snapshot.execute_sql(f"SELECT COUNT(*) FROM {table}")
                    for row in results:
                        logging.info(f"Spanner table '{table}' contains {row[0]} rows.")
                except Exception as e:
                    logging.error(f"Error querying row count for table {table}: {e}")
    except Exception as e:
        logging.error(f"Error creating Spanner snapshot: {e}", exc_info=True)
    return data

# def cleanup_redis(data):
#     batch_id = data["batch_id"]
#     redis_client.delete(f"batch:{batch_id}:results", f"batch:{batch_id}:counter")
#     return data

def cleanup_memgraph(data):
    """Deletes all nodes and relationships from Memgraph."""
    logging.info("Cleaning up Memgraph...")
    try:
        memgraph_graph.query("MATCH (n) DETACH DELETE n")
        logging.info("Memgraph cleaned up successfully.")
    except Exception as e:
        logging.error(f"Error cleaning up Memgraph: {e}", exc_info=True)
    return data

# --- LangChain Sequence ---
consolidation_chain = RunnableSequence(
    decode_pubsub_message,
    fetch_from_redis,
    log_redis_counts,
    aggregate_results,
    cluster_and_merge_entities,
    load_to_memgraph,
    log_memgraph_counts,
    run_community_detection,
    generate_hierarchical_summaries,
    store_summaries,
    migrate_to_spanner,
    log_spanner_counts,
    cleanup_redis,
    cleanup_memgraph
)

# --- Main Function ---
@functions_framework.cloud_event
def consolidator(cloud_event):
    try:
        initialize_clients()  # Initialize clients on each invocation
        consolidation_chain.invoke(cloud_event)
        return "OK", 200
    except Exception as e:
        logging.error(f'An error occurred in the consolidator: {e}', exc_info=True)
        return "Internal Server Error", 500
