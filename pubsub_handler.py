# Pub/Sub message handling for the consolidator function
import base64
import json
import logging

def decode_pubsub_message(cloud_event):
    logging.info(f"Received cloud_event: {cloud_event}")

    if isinstance(cloud_event, dict) and "batch_id" in cloud_event and "message" not in cloud_event:
        return {"batch_id": cloud_event["batch_id"]}

    event_data = cloud_event.get("data") if isinstance(cloud_event, dict) else cloud_event.data

    message_data = base64.b64decode(event_data["message"]["data"]).decode("utf-8")
    message_json = json.loads(message_data)
    return {"batch_id": message_json.get("batch_id")}