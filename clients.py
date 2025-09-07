# Client factory for the consolidator function

import redis
import google.cloud.secretmanager as secretmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from langchain_google_genai import ChatGoogleGenerativeAI

from config import Config

class ClientFactory:
    def __init__(self):
        self._redis_client = None
        self._llm = None
        self._db_session = None
        self._sm_client = None

    def get_sm_client(self):
        if self._sm_client is None:
            self._sm_client = secretmanager.SecretManagerServiceClient()
        return self._sm_client

    def get_redis_client(self):
        if self._redis_client is None:
            sm_client = self.get_sm_client()
            redis_password = sm_client.access_secret_version(request={"name": f"projects/{Config.GCP_PROJECT}/secrets/redis-password/versions/latest"}).payload.data.decode("UTF-8")
            self._redis_client = redis.Redis(host=Config.REDIS_HOST, port=Config.REDIS_PORT, password=redis_password, ssl=False, ssl_cert_reqs=None, decode_responses=True, socket_connect_timeout=10)
        return self._redis_client

    def get_llm(self):
        if self._llm is None:
            gemini_api_key = None
            if Config.GEMINI_API_KEY_SECRET_ID:
                sm_client = self.get_sm_client()
                try:
                    gemini_api_key = sm_client.access_secret_version(request={"name": Config.GEMINI_API_KEY_SECRET_ID}).payload.data.decode("UTF-8")
                except Exception as e:
                    # log the error
                    pass
            self._llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", convert_system_message_to_human=True, google_api_key=gemini_api_key)
        return self._llm

    def get_db_session(self):
        if self._db_session is None:
            db_uri = f"spanner+spanner:///projects/{Config.GCP_PROJECT}/instances/{Config.SPANNER_INSTANCE_ID}/databases/{Config.SPANNER_DATABASE_ID}"
            engine = create_engine(db_uri)
            Session = sessionmaker(bind=engine)
            self._db_session = Session()
        return self._db_session