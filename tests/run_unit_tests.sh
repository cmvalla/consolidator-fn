#!/bin/bash

# Run unit tests for ConsolidatorService

# Change to the consolidator directory
cd functions/consolidator

pytest tests/test_consolidator_service.py
