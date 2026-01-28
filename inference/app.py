import json
import os
import joblib

from flask import Flask, request, jsonify

MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/ml/model")


app = Flask(__name__)

# Global holders (loaded once)
preprocess = None
label_map = None
metadata = None
artifacts_loaded = False


def load_artifacts():
    global preprocess, label_map, metadata, artifacts_loaded

    try:
        preprocess_path = os.path.join(MODEL_DIR, "preprocess.joblib")
        label_map_path = os.path.join(MODEL_DIR, "label_map.json")
        metadata_path = os.path.join(MODEL_DIR, "metadata.json")

        preprocess = joblib.load(preprocess_path)

        with open(label_map_path, "r") as f:
            label_map = json.load(f)

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        artifacts_loaded = True
        print("Artifacts loaded successfully")

    except Exception as e:
        print("Artifact loading failed:", str(e))
        artifacts_loaded = False


# Load artifacts at import time (container startup)
load_artifacts()


@app.route("/ping", methods=["GET"])
def ping():
    """
    Health check.
    Should only return 200 if artifacts are loaded.
    """
    if artifacts_loaded:
        return jsonify(status="ok"), 200
    else:
        return jsonify(status="error", reason="artifacts not loaded"), 500


@app.route("/invocations", methods=["POST"])
def invocations():
    """
    Dummy inference.
    Just prove we can access loaded artifacts.
    """
    if not artifacts_loaded:
        return jsonify(error="Model not ready"), 503

    payload = request.get_json(silent=True)

    response = {
        "received": payload,
        "model_version": metadata.get("model_version"),
        "available_labels": list(label_map["label_to_id"].keys())
    }

    return jsonify(response), 200
