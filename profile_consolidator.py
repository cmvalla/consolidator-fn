import os
import sys
import json
import base64
import cProfile
import pstats
from unittest.mock import Mock

# Add the consolidator directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import consolidator

def create_mock_cloud_event(batch_id: str, gcs_path: str, chunk_number: int, total_chunks: int) -> Mock:
    inner_payload = {
        "batch_id": batch_id,
        "gcs_paths": [gcs_path],
        "chunk_number": chunk_number,
        "total_chunks": total_chunks
    }
    base64_payload = base64.b64encode(json.dumps(inner_payload).encode("utf-8")).decode("utf-8")

    mock_message = {
        "data": base64_payload,
        "messageId": f"local-test-{int(time.time())}"
    }
    mock_cloud_event = Mock()
    mock_cloud_event.data = mock_message
    return mock_cloud_event

if __name__ == "__main__":
    # Set environment variables for local debugging
    os.environ["LOCAL_DEBUG"] = "true"
    os.environ["GOOGLE_CLOUD_PROJECT"] = "spanner-demo-kid"
    os.environ["GCP_LOCATION"] = "europe-west1"
    os.environ["SPANNER_INSTANCE_ID"] = "demo-instance-kid"
    os.environ["SPANNER_DATABASE_ID"] = "spanner-graphrag-db"
    os.environ["EMBEDDING_SERVICE_URL"] = "https://graphrag-embedding-kg7odfkvta-ew.a.run.app"
    os.environ["PERSISTOR_TOPIC_NAME"] = "consolidator-processed-topic-kid"
    os.environ["GRAPH_DATA_BUCKET_NAME"] = "spanner-demo-graph-data-kid"
    os.environ["CONSOLIDATOR_MAX_WORKERS"] = "5"
    os.environ["LLM_BATCH_SIZE"] = "10"
    os.environ["LLM_MODEL_NAME"] = "gemini-2.5-pro"
    os.environ["GEMINI_API_KEY_SECRET_ID"] = "projects/spanner-demo-kid/secrets/gemini-api-key/versions/latest"

    # --- Auto-configure batch_id and GCS path from latest GCS object ---
    # This part mimics the logic from trigger_local_consolidator.sh
    PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
    GRAPH_DATA_BUCKET_NAME = os.environ["GRAPH_DATA_BUCKET_NAME"]

    BATCH_ID = ""
    CHUNK_NUMBER = 0
    TOTAL_CHUNKS = 1
    GCS_PATH = ""

    print("Batch ID not provided. Searching for the latest GCS object...")
    # This part requires gsutil to be configured and accessible
    # For profiling, you might need to manually provide a GCS_PATH if gsutil is not easily integrated
    # For simplicity in this script, let's assume a hardcoded path for now or require it as an arg
    # In a real scenario, you'd run a shell command here to get the latest GCS path
    # For now, let's use a placeholder or expect the user to provide it.
    
    # Placeholder for GCS_PATH - USER MUST REPLACE WITH A VALID PATH FOR PROFILING
    # Example: GCS_PATH = "gs://spanner-demo-graph-data-kid/graph_data/my-batch-123/0_1678886400.pkl"
    # For a real run, you'd execute gsutil ls and parse the latest file.
    # For this profiling script, let's assume a valid GCS_PATH is provided or hardcoded.
    
    # To make this script runnable without manual intervention for GCS path, 
    # we'll need to execute a shell command to find the latest GCS path.
    # This is a bit complex to do purely in Python without external dependencies or direct gsutil calls.
    # For now, let's make it a placeholder and instruct the user.
    
    # For a quick test, you can manually set a valid GCS_PATH here:
    # GCS_PATH = "gs://spanner-demo-graph-data-kid/graph_data/test-batch-123/0_1701000000000000.pkl"
    
    # To dynamically get the latest GCS path, we'd need to run a shell command:
    import subprocess
    try:
        all_gcs_paths_cmd = f"gsutil ls gs://{GRAPH_DATA_BUCKET_NAME}/graph_data/*/*.pkl | sort -r"
        all_gcs_paths_output = subprocess.check_output(all_gcs_paths_cmd, shell=True, text=True)
        if not all_gcs_paths_output.strip():
            print("Error: No GCS objects found. Please ensure worker has run and uploaded data.")
            sys.exit(1)
        LATEST_GCS_PATH = all_gcs_paths_output.splitlines()[0]
        print(f"Found latest GCS object: {LATEST_GCS_PATH}")

        # Extract BATCH_ID and CHUNK_NUMBER from the GCS path
        # Expected format: gs://bucket_name/graph_data/{batch_id}/{chunk_number}_{timestamp}.pkl
        filename = os.path.basename(LATEST_GCS_PATH)
        batch_id_match = re.search(r'graph_data/([^/]+)/', LATEST_GCS_PATH)
        chunk_number_match = re.search(r'^([0-9]+)_', filename)

        if batch_id_match:
            BATCH_ID = batch_id_match.group(1)
        if chunk_number_match:
            CHUNK_NUMBER = int(chunk_number_match.group(1))
        
        GCS_PATH = LATEST_GCS_PATH
        print(f"Auto-configured: BATCH_ID={BATCH_ID}, CHUNK_NUMBER={CHUNK_NUMBER}")

    except subprocess.CalledProcessError as e:
        print(f"Error running gsutil: {e}")
        print("Please ensure gsutil is installed and authenticated.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during GCS path auto-configuration: {e}")
        sys.exit(1)

    if not BATCH_ID or not GCS_PATH:
        print("Error: Could not determine BATCH_ID or GCS_PATH automatically. Please provide them manually if needed.")
        sys.exit(1)

    # Create a mock CloudEvent
    mock_cloud_event = create_mock_cloud_event(BATCH_ID, GCS_PATH, CHUNK_NUMBER, TOTAL_CHUNKS)

    # Run the consolidator function with cProfile
    print("Starting consolidator profiling...")
    profiler = cProfile.Profile()
    profiler.enable()
    consolidator(mock_cloud_event)
    profiler.disable()
    print("Consolidator profiling finished.")

    # Save and print profiling statistics
    stats_file = "consolidator_profile.pstats"
    with open(stats_file, "w") as f:
        ps = pstats.Stats(profiler, stream=f)
        ps.sort_stats("cumulative").print_stats(20) # Top 20 functions by cumulative time
    print(f"Profiling results saved to {stats_file}")
    print("You can visualize the results using snakeviz: snakeviz consolidator_profile.pstats")

