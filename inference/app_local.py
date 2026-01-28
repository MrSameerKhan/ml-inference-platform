import json
import os
import joblib
from flask import Flask, request, jsonify

# Change version here when needed
LOCAL_MODEL_DIR = r"C:\Users\samee\Desktop\ml-inference-platform\artifacts\versioned\v3.0.0"

app = Flask(__name__)

preprocess = None
label_map = None
metadata = None
artifacts_loaded = False
load_error = None


def load_artifacts():
    global preprocess, label_map, metadata, artifacts_loaded, load_error

    try:
        preprocess_path = os.path.join(LOCAL_MODEL_DIR, "preprocess.joblib")
        label_map_path = os.path.join(LOCAL_MODEL_DIR, "label_map.json")
        metadata_path = os.path.join(LOCAL_MODEL_DIR, "metadata.json")

        preprocess = joblib.load(preprocess_path)

        with open(label_map_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        artifacts_loaded = True
        load_error = None
        print(f"[LOCAL] Artifacts loaded from: {LOCAL_MODEL_DIR}")

    except Exception as e:
        artifacts_loaded = False
        load_error = str(e)
        print(f"[LOCAL] Artifact loading failed from: {LOCAL_MODEL_DIR}")
        print(f"[LOCAL] ERROR: {load_error}")


# Load once at startup
load_artifacts()


@app.route("/ping", methods=["GET"])
def ping():
    if artifacts_loaded:
        return jsonify(status="ok", model_dir=LOCAL_MODEL_DIR), 200
    return jsonify(status="error", reason="artifacts not loaded", model_dir=LOCAL_MODEL_DIR, error=load_error), 500


@app.route("/invocations", methods=["POST"])
def invocations():
    if not artifacts_loaded:
        return jsonify(error="Model not ready", model_dir=LOCAL_MODEL_DIR, reason=load_error), 503

    payload = request.get_json(silent=True)

    # Return artifact info only (no ML yet)
    response = {
        "received": payload,
        "model_version": (metadata or {}).get("model_version"),
        "available_labels": list((label_map or {}).get("label_to_id", {}).keys()),
        "model_dir": LOCAL_MODEL_DIR,
    }
    return jsonify(response), 200


if __name__ == "__main__":
    # Local dev server
    app.run(host="0.0.0.0", port=8080, debug=False)
