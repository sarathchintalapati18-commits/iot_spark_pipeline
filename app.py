import pandas as pd
import joblib
import paho.mqtt.client as mqtt
import os
from flask import Flask, jsonify

# Flask app
app = Flask(__name__)

# Load trained XGBoost pipeline
MODEL_PATH = "xgboost_pipeline.pkl"
model = joblib.load()

# HiveMQ credentials from environment variables
BROKER = os.getenv("BROKER")
PORT = int(os.getenv("PORT", 8883))
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

# Placeholder for latest prediction
latest_prediction = None

# MQTT callback
def on_message(client, userdata, message):
    global latest_prediction
    import json
    payload = json.loads(message.payload.decode())
    df = pd.DataFrame([payload])
    pred = model.predict(df)
    latest_prediction = int(pred[0])
    print("New prediction:", latest_prediction)

# MQTT client setup
client = mqtt.Client()
client.username_pw_set(USERNAME, PASSWORD)
client.tls_set()  # Use TLS for secure connection
client.on_message = on_message
client.connect(BROKER, PORT)
client.subscribe("device/data")  # Subscribe to your topic
client.loop_start()

# Flask route to get latest prediction
@app.route("/latest_prediction", methods=["GET"])
def get_prediction():
    if latest_prediction is None:
        return jsonify({"prediction": "No data yet"})
    return jsonify({"prediction": latest_prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
