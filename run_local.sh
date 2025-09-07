#!/bin/bash

# Navigate to the script's directory
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

# Define the path to the virtual environment
VENV_PATH="./.venv"

# Check if the virtual environment exists, if not, create it
if [ ! -d "$VENV_PATH" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv "$VENV_PATH"
fi

# Activate the virtual environment (for subsequent commands in this script)
source "$VENV_PATH/bin/activate"

# Install dependencies from requirements.txt
echo "Installing/updating dependencies..."
pip install -r requirements.txt

# Install pytest (if not already installed)
echo "Installing pytest..."
pip install pytest

# Inform the user how to start the server
echo ""
echo "Environment setup complete. To start the consolidator local server, run:"
echo "functions-framework --target consolidator --port 8080"
echo ""
echo "Remember to send a Cloud Event payload via POST request to http://localhost:8080/"
