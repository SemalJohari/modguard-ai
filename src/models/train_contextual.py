from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

from tqdm.auto import tqdm


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

LABELS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit"
]

CHECKPOINT = "distilbert-base-uncased"

DATA_DIR = Path("data/contextual")
MODEL_DIR = Path("models/contextual_01")

MAX_LENGTH = 192
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42


# --------------------------------------------------
# REPRODUCIBILITY
# --------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Device:", DEVICE)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

train_df = pd.read_csv(
    DATA_DIR / "contextual_train.csv"
)

val_df = pd.read_csv(
    DATA_DIR / "contextual_validation.csv"
)

required_columns = [
    "text",
    "parent_comment",
    *LABELS
]

for name, df in [
    ("training", train_df),
    ("validation", val_df)
]:
    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} data is missing columns: {missing}"
        )

    if df[["text", *LABELS]].isnull().any().any():
        raise ValueError(
            f"{name} data contains missing required values."
        )

    df["parent_comment"] = df["parent_comment"].fillna("")

    for label in LABELS:
        if not set(df[label].unique()).issubset({0, 1}):
            raise ValueError(
                f"{label} must contain only binary 0/1 labels."
            )

print("Training examples:", len(train_df))
print("Validation examples:", len(val_df))


# --------------------------------------------------
# CONTEXT FORMATTING
# --------------------------------------------------

def build_contextual_text(comment, parent):
    comment = str(comment).strip()
    parent = str(parent).strip()

    if parent:
        return (
            f"[CURRENT COMMENT] {comment} "
            f"[PARENT COMMENT] {parent}"
        )

    return f"[CURRENT COMMENT] {comment}"


# --------------------------------------------------
# TOKENIZER
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)


# --------------------------------------------------
# DATASET
# --------------------------------------------------

class ContextualDataset(Dataset):

    def __init__(self, dataframe):
        self.texts = [
            build_contextual_text(
                row["text"],
                row["parent_comment"]
            )
            for _, row in dataframe.iterrows()
        ]

        self.labels = dataframe[LABELS].values.astype(
            np.float32
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return {
            "text": self.texts[index],
            "labels": self.labels[index]
        }


def collate_fn(batch):

    texts = [item["text"] for item in batch]

    labels = torch.tensor(
        np.array([item["labels"] for item in batch]),
        dtype=torch.float32
    )

    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    tokens["labels"] = labels

    return tokens


train_dataset = ContextualDataset(train_df)
val_dataset = ContextualDataset(val_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0
)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

class ContextualClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            CHECKPOINT
        )

        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Linear(
            hidden_size,
            len(LABELS)
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(
            self.dropout(pooled_output)
        )

        loss = None

        if labels is not None:
            loss_function = nn.BCEWithLogitsLoss()
            loss = loss_function(logits, labels)

        return {
            "loss": loss,
            "logits": logits
        }


model = ContextualClassifier().to(DEVICE)


# --------------------------------------------------
# OPTIMIZER AND SCHEDULER
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

total_steps = len(train_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps
)


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate():

    model.eval()

    all_probabilities = []
    all_targets = []

    with torch.no_grad():

        for batch in val_loader:

            labels = batch.pop("labels").to(DEVICE)

            batch = {
                key: value.to(DEVICE)
                for key, value in batch.items()
            }

            outputs = model(**batch)

            probabilities = torch.sigmoid(
                outputs["logits"]
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_targets.append(
                labels.cpu().numpy()
            )

    probabilities = np.vstack(all_probabilities)
    targets = np.vstack(all_targets)

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {}

    for index, label in enumerate(LABELS):

        y_true = targets[:, index]
        y_pred = predictions[:, index]
        y_score = probabilities[:, index]

        if y_true.sum() > 0:
            average_precision = float(
                average_precision_score(
                    y_true,
                    y_score
                )
            )
        else:
            average_precision = None

        metrics[label] = {
            "precision": float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0
                )
            ),
            "f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0
                )
            ),
            "average_precision": average_precision,
            "support": int(y_true.sum())
        }

    valid_ap = [
        item["average_precision"]
        for item in metrics.values()
        if item["average_precision"] is not None
    ]

    metrics["macro_average_precision"] = (
        float(np.mean(valid_ap))
        if valid_ap else None
    )

    return metrics


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

best_macro_ap = -1

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for batch in progress:

        labels = batch.pop("labels").to(DEVICE)

        batch = {
            key: value.to(DEVICE)
            for key, value in batch.items()
        }

        optimizer.zero_grad()

        outputs = model(
            **batch,
            labels=labels
        )

        loss = outputs["loss"]

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        progress.set_postfix(
            loss=loss.item()
        )

    average_loss = total_loss / len(train_loader)

    metrics = evaluate()

    print(f"\nEpoch {epoch + 1}")
    print("Training loss:", average_loss)
    print(json.dumps(metrics, indent=2))

    current_ap = metrics["macro_average_precision"]

    if (
        current_ap is not None
        and current_ap > best_macro_ap
    ):

        best_macro_ap = current_ap

        torch.save(
            model.state_dict(),
            MODEL_DIR / "best_model.pt"
        )

        tokenizer.save_pretrained(
            MODEL_DIR / "tokenizer"
        )

        config = {
            "checkpoint": CHECKPOINT,
            "labels": LABELS,
            "max_length": MAX_LENGTH,
            "decision_threshold": 0.5,
            "dropout": 0.2,
            "best_epoch": epoch + 1
        }

        with open(
            MODEL_DIR / "config.json",
            "w"
        ) as file:
            json.dump(config, file, indent=4)

        with open(
            MODEL_DIR / "best_metrics.json",
            "w"
        ) as file:
            json.dump(metrics, file, indent=4)

print("\nTraining completed.")
print("Best macro AP:", best_macro_ap)