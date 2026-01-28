import argparse
import json
import os
import tarfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--artifact_dir",
        required=True,
        help="Example: artifacts/versioned/v1.0.0"
    )
    args = ap.parse_args()

    artifact_dir = args.artifact_dir.rstrip("/\\")

    files_to_package = {
        "keras_model": os.path.join(artifact_dir, "keras_model"),
        "preprocess.joblib": os.path.join(artifact_dir, "preprocess.joblib"),
        "label_map.json": os.path.join(artifact_dir, "label_map.json"),
        "metadata.json": os.path.join(artifact_dir, "metadata.json"),
    }

    # Validate presence
    for name, path in files_to_package.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required artifact: {path}")

    # Read metadata for logging
    with open(files_to_package["metadata.json"], "r", encoding="utf-8") as f:
        meta = json.load(f)

    out_tar = os.path.join(artifact_dir, "saved_model.tar.gz")

    with tarfile.open(out_tar, "w:gz") as tar:
        tar.add(files_to_package["keras_model"], arcname="keras_model")
        tar.add(files_to_package["preprocess.joblib"], arcname="preprocess.joblib")
        tar.add(files_to_package["label_map.json"], arcname="label_map.json")
        tar.add(files_to_package["metadata.json"], arcname="metadata.json")

    print("PACKAGE_OK")
    print("MODEL_VERSION:", meta.get("model_version"))
    print("WROTE:", out_tar)

# python training\package_artifacts.py --artifact_dir artifacts\versioned\v3.0.0
if __name__ == "__main__":
    main()
