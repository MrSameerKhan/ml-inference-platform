import argparse
import hashlib
import json
import os
import time

import joblib
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_version", required=True, help="Example: v1.0.0")
    ap.add_argument("--out_dir", default="artifacts", help="Root artifact folder")
    args = ap.parse_args()

    version = args.model_version.strip()
    out_root = os.path.join(args.out_dir, "versioned", version)
    ensure_dir(out_root)

    # Demo dataset (replace later)
    texts = [
        "refund my money", "need a refund", "cancel subscription", "want to cancel",
        "app is crashing", "bug in app", "cannot login", "login fails",
        "pricing is too high", "need discount", "payment failed", "card declined",
        "how to change password", "reset my password", "forgot password", "password reset link not working",
    ]
    labels = [
        "refund", "refund", "cancel", "cancel",
        "bug", "bug", "login", "login",
        "pricing", "pricing", "payment", "payment",
        "account", "account", "account", "account",
    ]

    label_set = sorted(list(set(labels)))
    label_to_id = {l: i for i, l in enumerate(label_set)}
    id_to_label = {i: l for l, i in label_to_id.items()}
    y = np.array([label_to_id[l] for l in labels], dtype=np.int32)

    # Ensure stratified split has at least 1 sample per class in test set
    n_classes = len(label_set)
    min_test = n_classes  # at least one per class
    test_size = max(0.25, min_test / len(texts))

    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=test_size, random_state=42, stratify=y
)


    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    Xtr = vectorizer.fit_transform(X_train).toarray().astype(np.float32)
    Xte = vectorizer.transform(X_test).toarray().astype(np.float32)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(Xtr.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(label_set), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(Xtr, y_train, epochs=20, batch_size=8, verbose=0)

    probs = model.predict(Xte, verbose=0)
    pred = np.argmax(probs, axis=1)
    acc = float(accuracy_score(y_test, pred))

    keras_dir = os.path.join(out_root, "keras_model")
    preprocess_path = os.path.join(out_root, "preprocess.joblib")
    label_map_path = os.path.join(out_root, "label_map.json")
    metadata_path = os.path.join(out_root, "metadata.json")

    model.save(keras_dir)
    joblib.dump(vectorizer, preprocess_path)

    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, f, indent=2)

    metadata = {
        "model_version": version,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task": "text_classification",
        "framework": "tensorflow",
        "metrics": {"accuracy": acc},
        "artifacts": {
            "keras_model_dir": "keras_model",
            "preprocess": "preprocess.joblib",
            "label_map": "label_map.json",
            "saved_model_tar": "saved_model.tar.gz"
        },
        "hashes": {
            "preprocess.joblib": sha256_file(preprocess_path),
            "label_map.json": sha256_file(label_map_path),
        }
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("TRAIN_OK")
    print("VERSION:", version)
    print("OUT:", out_root)
    print("ACCURACY:", f"{acc:.4f}")
    print("METADATA:", metadata_path)

# python training/train.py --model_version v3.0.0 --out_dir artifacts
if __name__ == "__main__":
    main()
