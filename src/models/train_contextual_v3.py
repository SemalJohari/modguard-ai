from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from tqdm.auto import tqdm

# V3 training run. The old contextual_01 checkpoint is NOT overwritten.
LABELS = [
    "toxicity", "severe_toxicity", "obscene", "threat",
    "insult", "identity_attack", "sexual_explicit",
]
DATA_DIR = Path("data/experiments/v3")
MODEL_DIR = Path("models/contextual_02")
BASE_CHECKPOINT = "distilbert-base-uncased"

MAX_LENGTH = 192
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_text(comment, parent):
    comment = str(comment).strip()
    parent = "" if pd.isna(parent) else str(parent).strip()
    if parent:
        return f"[CURRENT COMMENT] {comment} [PARENT COMMENT] {parent}"
    return f"[CURRENT COMMENT] {comment}"


def load_split(filename):
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    df = pd.read_csv(path)
    required = {"text", "parent_comment", *LABELS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{filename} is missing columns: {sorted(missing)}")
    if df["text"].isna().any():
        raise ValueError(f"{filename} contains missing comment text.")
    df["parent_comment"] = df["parent_comment"].fillna("")
    for label in LABELS:
        values = set(df[label].dropna().unique())
        if not values.issubset({0, 1}) or df[label].isna().any():
            raise ValueError(f"{filename}: {label} must contain only 0/1 values.")
    return df


class CommentDataset(Dataset):
    def __init__(self, frame):
        self.texts = [
            build_text(row.text, row.parent_comment)
            for row in frame.itertuples(index=False)
        ]
        self.labels = frame[LABELS].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def make_collate(tokenizer):
    def collate(batch):
        texts, labels = zip(*batch)
        tokens = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        tokens.pop("token_type_ids", None)
        tokens["labels"] = torch.tensor(np.asarray(labels), dtype=torch.float32)
        return tokens
    return collate


class ContextualClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(BASE_CHECKPOINT)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, len(LABELS))

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = output.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(logits, labels)
        return loss, logits


def evaluate(model, loader):
    model.eval()
    probs_all, targets_all = [], []
    with torch.no_grad():
        for batch in loader:
            targets = batch.pop("labels").to(DEVICE)
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            _, logits = model(**batch)
            probs_all.append(torch.sigmoid(logits).cpu().numpy())
            targets_all.append(targets.cpu().numpy())

    probs = np.vstack(probs_all)
    targets = np.vstack(targets_all)
    metrics = {}
    aps = []

    for i, label in enumerate(LABELS):
        y_true = targets[:, i].astype(int)
        y_score = probs[:, i]
        y_pred = (y_score >= 0.5).astype(int)
        ap = float(average_precision_score(y_true, y_score)) if y_true.sum() else None
        if ap is not None:
            aps.append(ap)
        metrics[label] = {
            "precision_at_0.5": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall_at_0.5": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_at_0.5": float(f1_score(y_true, y_pred, zero_division=0)),
            "average_precision": ap,
            "positive_support": int(y_true.sum()),
        }

    metrics["macro_average_precision"] = float(np.mean(aps)) if aps else None
    return metrics


def main():
    set_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_split("train.csv")
    val_df = load_split("validation.csv")

    print(f"Device: {DEVICE}")
    print(f"Train rows: {len(train_df)} | Validation rows: {len(val_df)}")
    print("Training label positives:")
    print(train_df[LABELS].sum().to_dict())

    tokenizer = AutoTokenizer.from_pretrained(BASE_CHECKPOINT)
    collate = make_collate(tokenizer)

    train_loader = DataLoader(
        CommentDataset(train_df), batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=collate, num_workers=0,
    )
    val_loader = DataLoader(
        CommentDataset(val_df), batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=collate, num_workers=0,
    )

    model = ContextualClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.1)),
        num_training_steps=total_steps,
    )

    best_ap = -1.0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for batch in progress:
            labels = batch.pop("labels").to(DEVICE)
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model(**batch, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        metrics = evaluate(model, val_loader)
        print(f"\nEpoch {epoch + 1} mean training loss: {running_loss / len(train_loader):.4f}")
        print(json.dumps(metrics, indent=2))

        current_ap = metrics["macro_average_precision"]
        if current_ap is not None and current_ap > best_ap:
            best_ap = current_ap
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pt")
            tokenizer.save_pretrained(MODEL_DIR / "tokenizer")
            config = {
                "checkpoint": BASE_CHECKPOINT,
                "labels": LABELS,
                "max_length": MAX_LENGTH,
                "decision_threshold": 0.5,
                "dropout": 0.2,
                "best_epoch": epoch + 1,
                "training_dataset": "data/experiments/v3/train.csv",
                "validation_dataset": "data/experiments/v3/validation.csv",
            }
            (MODEL_DIR / "config.json").write_text(json.dumps(config, indent=2))
            (MODEL_DIR / "best_metrics.json").write_text(json.dumps(metrics, indent=2))
            print(f"Saved best checkpoint to {MODEL_DIR}")

    print(f"\nTraining finished. Best validation macro AP: {best_ap:.4f}")
    print(f"New model directory: {MODEL_DIR}")
    print("The existing models/contextual_01 checkpoint was not modified.")


if __name__ == "__main__":
    main()
