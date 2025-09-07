import os
import threading
from flask import Flask

health_app = Flask(__name__)

@health_app.route('/')
def health_check():
    return "OK", 200

def run_health_check_server():
    port = int(os.environ.get("PORT", 8080))
    health_app.run(host='0.0.0.0', port=port)

# Start the Flask health check server in a separate thread
health_check_thread = threading.Thread(target=run_health_check_server)
health_check_thread.daemon = True
health_check_thread.start()