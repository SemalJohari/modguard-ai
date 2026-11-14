
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

LABELS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]

DATA_DIR = Path("data/raw")
MODEL_DIR = Path("models/baseline")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_parquet(DATA_DIR / "train_dev.parquet")
valid = pd.read_parquet(DATA_DIR / "validation.parquet")

thresholds = [0.001, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5]

for label in LABELS:
    print(f"\n{'=' * 45}")
    print(f"LABEL: {label}")
    print(f"{'=' * 45}")

    for threshold in thresholds:
        train_count = (train[label] >= threshold).sum()
        valid_count = (valid[label] >= threshold).sum()

        print(
            f"Threshold {threshold:>5}: "
            f"Train = {train_count:>5}, "
            f"Validation = {valid_count:>4}"
        )

print("\n========== TRAINING DATA ==========")

for label in LABELS:
    print(f"\n{label}")
    print(train[label].describe())
    print("Unique values:", train[label].nunique())
    print("NaN count:", train[label].isna().sum())
    print("Values >= 0.5:", (train[label].fillna(0) >= 0.5).sum())

print("\n========== VALIDATION DATA ==========")

for label in LABELS:
    print(f"\n{label}")
    print(valid[label].describe())
    print("Unique values:", valid[label].nunique())
    print("NaN count:", valid[label].isna().sum())
    print("Values >= 0.5:", (valid[label].fillna(0) >= 0.5).sum())

# Use the same documented target definition throughout.
X_train = train["text"].fillna("").astype(str)
X_valid = valid["text"].fillna("").astype(str)

# These are operational thresholds, not universal ground truth.
Y_train = (train[LABELS].fillna(0).to_numpy() >= 0.5).astype(int)
Y_valid = (valid[LABELS].fillna(0).to_numpy() >= 0.5).astype(int)

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=3,
            max_features=300_000,
            sublinear_tf=True,
            strip_accents="unicode",
        ),
    ),
    (
        "classifier",
        MultiOutputClassifier(
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="liblinear",
            ),
            n_jobs=1,
        ),
    ),
])

model.fit(X_train, Y_train)

probabilities = np.column_stack([
    estimator.predict_proba(
        model.named_steps["tfidf"].transform(X_valid)
    )[:, 1]
    for estimator in model.named_steps["classifier"].estimators_
])

predictions = (probabilities >= 0.5).astype(int)

results = {}

for i, label in enumerate(LABELS):
    results[label] = {
        "precision": float(
            precision_score(
                Y_valid[:, i], predictions[:, i], zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                Y_valid[:, i], predictions[:, i], zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                Y_valid[:, i], predictions[:, i], zero_division=0
            )
        ),
        "average_precision": (
            float(average_precision_score(
                Y_valid[:, i], probabilities[:, i]
            ))
            if Y_valid[:, i].sum() > 0
            else None
        ),
        "support": int(Y_valid[:, i].sum()),
    }

print(json.dumps(results, indent=2))

joblib.dump(model, MODEL_DIR / "model.joblib")

with open(MODEL_DIR / "metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved baseline model.")

print("\nPositive samples in training set:")
print(Y_train.sum(axis=0))

print("\nPositive samples in test set:")
print(Y_valid.sum(axis=0))

print("\nTraining distribution:")
print(Y_train.mean(axis=0))

print("\nTesting distribution:")
print(Y_valid.mean(axis=0))

print("X_train:", X_train.shape)
print("X_test:", X_valid.shape)

print("\nTraining positive counts:")
print(Y_train.sum(axis=0))

print("\nTesting positive counts:")
print(Y_valid.sum(axis=0))