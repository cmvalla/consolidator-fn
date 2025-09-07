#!/bin/bash

# Navigate to the script's directory
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

# Define the path to the virtual environment
VENV_PATH="./.venv"

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

echo "Sending test payload to consolidator local server..."
curl -X POST http://localhost:8080/ \
-H "Content-Type: application/json" \
-H "ce-specversion: 1.0" \
-H "ce-type: google.cloud.pubsub.topic.v1.messagePublished" \
-H "ce-source: //pubsub.googleapis.com/projects/your-gcp-project-id/topics/your-pubsub-topic-name" \
-H "ce-id: $(uuidgen)" \
-d '{ "message": { "data": "eyJidWNrZXROYW1lIjogIm15LWJ1Y2tldCIsICJmaWxlTmFtZSI6ICJteS1maWxlIn0=", "attributes": { "batch_id": "your_test_batch_id" } } }'
