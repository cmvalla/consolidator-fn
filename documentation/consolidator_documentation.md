# Consolidator Function Documentation

## 1. Overall Purpose

The `consolidator` function is a key component of the GraphRAG pipeline. Its primary responsibility is to take the partially processed graph data from the `worker` functions, consolidate it into a single graph, perform further analysis and enrichment, and finally store the resulting graph in Cloud Spanner.

## 2. Triggering

The `consolidator` function is triggered by a message published to a Pub/Sub topic. The message contains a `batch_id` that identifies the batch of documents to be processed. The trigger is defined in `terraform/pubsub.tf` and the Cloud Run service is configured to be invoked by this trigger in `functions/consolidator/terraform/main.tf`.

## 3. Initialization

Upon invocation, the `consolidator` function performs the following initialization steps in the `initialize_clients` function:

*   **Logging**: Initializes the Google Cloud Logging client.
*   **Environment Variables**: Reads essential configuration from environment variables, including GCP project ID, region, Redis host/port, and Spanner instance/database IDs.
*   **Secret Manager**: Initializes the Secret Manager client to access secrets like the Redis password and the Gemini API key.
*   **Redis**: Initializes a connection to the Redis instance, which is used for intermediate storage of graph data.
*   **Vertex AI (Gemini)**: Initializes the `ChatGoogleGenerativeAI` client for interacting with the Gemini LLM. It authenticates using the API key retrieved from Secret Manager.
*   **Spanner (SQLAlchemy)**: Initializes a connection to the Cloud Spanner database using SQLAlchemy. It creates a database engine and a session for interacting with the database.
*   **Schema Creation**: Calls the `ensure_spanner_schema` function to create all the necessary tables, the property graph, and vector indexes in Spanner if they don't already exist.

## 4. Processing Pipeline

The core logic of the `consolidator` is implemented as a LangChain `RunnableSequence`. This pipeline processes the data in a series of steps:

1.  **`aggregate_results`**: Fetches the partial graph data for a given `batch_id` from Redis and aggregates all entities and relationships into a single data structure.

2.  **`generate_embeddings`**: Generates vector embeddings for all entities and communities using the `graphrag-embedding` service. For `Chunk` entities, it generates a summary of the chunk's text if one doesn't exist and then creates an embedding for the summary. For other entities, it creates an embedding from their type and properties.

3.  **`cluster_and_merge_entities`**: This step performs two key tasks:
    *   **Clustering**: It clusters similar entities based on the cosine similarity of their embeddings.
    *   **Class Creation**: For each cluster, it uses the Gemini LLM to generate a `Class` entity that represents the common theme of the cluster. It also creates `INSTANCE_OF` relationships between the original entities and their new `Class` entities.

4.  **`deduplicate_entities`**: This function resolves duplicate entity IDs (`Eid`) that might have been created during the distributed processing by the `worker` functions. It handles duplicate `Class` and `Instance` entities by merging or renaming them.

5.  **`run_igraph_community_detection`**: Uses the `igraph` library to perform community detection on the graph. It identifies maximal cliques as overlapping communities and creates `Community` entities for each one.

6.  **`store_consolidated_results_in_redis`**: Stores the final, consolidated graph data (entities and relationships) back into Redis. This serves as a cache and allows for reprocessing from this point without re-running the entire pipeline.

7.  **`migrate_to_spanner`**: Migrates the final graph data to Cloud Spanner using SQLAlchemy. It performs "upsert" operations to insert or update entities, relationships, and `InstanceOf` relationships in the Spanner database.

## 5. Schema Management

The `consolidator` function is responsible for managing the Cloud Spanner schema. The `ensure_spanner_schema` function reads the DDL statements from the `schema.sql` file and executes them. This includes creating the following tables, property graph, and vector indexes:

*   **Tables**: `Entities`, `Relationships`, `InstanceOf`, `ProcessedDocuments`, `Communities`, `EntityCommunity`, `WorkflowStatus`.
*   **Property Graph**: `my_graph` which defines the graph structure with `Entities` and `Communities` as nodes and `Relationships`, `InstanceOf`, and `EntityCommunity` as edges.
*   **Vector Indexes**: `EntitiesEmbeddingIndex` and `CommunitiesEmbeddingIndex` on the `Embedding` columns of the respective tables to enable efficient vector similarity searches.

## 6. Error Handling

The `consolidator` function has robust error handling:

*   **Initialization**: If any of the global clients fail to initialize, a critical error is logged, and the function execution is halted.
*   **Processing**: If an error occurs during the main processing pipeline, the error is logged, and the `WorkflowStatus` for the given `batch_id` is updated to `FAILED` in Spanner.
*   **Spanner Migration**: The `migrate_to_spanner` function uses a transaction, and if any error occurs during the database operations, the transaction is rolled back to maintain data consistency.

## 7. Configuration

The `consolidator` function is configured using the following key environment variables:

*   `GOOGLE_CLOUD_PROJECT`: The GCP project ID.
*   `REDIS_HOST`: The hostname or IP address of the Redis instance.
*   `REDIS_PORT`: The port number for the Redis instance.
*   `SPANNER_INSTANCE_ID`: The ID of the Cloud Spanner instance.
*   `SPANNER_DATABASE_ID`: The ID of the Cloud Spanner database.
*   `GEMINI_API_KEY_SECRET_ID`: The resource ID of the secret in Secret Manager that stores the Gemini API key.
*   `EMBEDDING_SERVICE_URL`: The URL of the `graphrag-embedding` service.
*   `MAX_WORKERS`: The number of parallel threads to use for LLM calls during class property generation.