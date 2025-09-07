# Configuration for the consolidator function
import os

class Config:
    GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
    REDIS_HOST = os.environ.get("REDIS_HOST")
    REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
    SPANNER_INSTANCE_ID = os.environ.get("SPANNER_INSTANCE_ID")
    SPANNER_DATABASE_ID = os.environ.get("SPANNER_DATABASE_ID")
    LOCATION = os.environ.get("LOCATION")
    GEMINI_API_KEY_SECRET_ID = os.environ.get("GEMINI_API_KEY_SECRET_ID")
    EMBEDDING_SERVICE_URL = os.environ.get("EMBEDDING_SERVICE_URL")
    MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 5))
    EMBEDDING_DIMENSION = 768