#!/bin/bash

# Navigate to the script's directory
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

# Define the path to the service account key
KEY_PATH="./credentials.json"

# Check if the key file exists
if [ ! -f "$KEY_PATH" ]; then
    echo "Error: Service account key file not found at $KEY_PATH"
    echo "Please download your consolidator service account key and place it in this directory."
    exit 1
fi

echo "Building Docker image 'consolidator-local'"...
podman build -t consolidator-local .

echo "Starting consolidator local server in Docker container"...
podman run -p 8080:8080 \
  -v "$(pwd)/credentials.json:/tmp/credentials.json" \
  -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/credentials.json \
  consolidator-local
