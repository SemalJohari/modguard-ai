
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

from tqdm.auto import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

LABELS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]

CHECKPOINT = "distilbert-base-uncased"

DATA_DIR = Path("data/raw")
MODEL_DIR = Path("models/transformer_02")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

ANNOTATION_THRESHOLD = 0.2
MAX_LENGTH = 192
EPOCHS = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
GRADIENT_CLIP = 1.0
GRAD_ACCUMULATION = 1

# Automatically select hardware.
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 16 if DEVICE.type == "cuda" else 4

print("=" * 60)
print("MODGUARD AI - DISTILBERT TRAINING")
print("=" * 60)

print("Device:", DEVICE)
print("Checkpoint:", CHECKPOINT)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("Annotation threshold:", ANNOTATION_THRESHOLD)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_parquet(DATA_DIR / "train_dev.parquet")
valid = pd.read_parquet(DATA_DIR / "validation.parquet")

print("\nDataset sizes:")
print("Training:", len(train))
print("Validation:", len(valid))

# Preserve original annotation scores.
train_scores = train[LABELS].fillna(0).to_numpy(
    dtype=np.float32
)

valid_scores = valid[LABELS].fillna(0).to_numpy(
    dtype=np.float32
)

# Convert continuous annotation scores into binary targets.
Y_train = (
    train_scores >= ANNOTATION_THRESHOLD
).astype(np.float32)

Y_valid = (
    valid_scores >= ANNOTATION_THRESHOLD
).astype(np.float32)

# Check label distributions.
print("\nTraining positive counts:")

for i, label in enumerate(LABELS):
    print(f"{label}: {int(Y_train[:, i].sum())}")

print("\nValidation positive counts:")

for i, label in enumerate(LABELS):
    print(f"{label}: {int(Y_valid[:, i].sum())}")


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)


# ============================================================
# PYTORCH DATASET
# ============================================================

class ToxicityDataset(Dataset):

    def __init__(self, texts, labels):
        self.texts = texts.fillna("").astype(str).tolist()
        self.labels = torch.tensor(
            labels,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return (
            self.texts[index],
            self.labels[index]
        )


# ============================================================
# BATCH COLLATOR
# ============================================================

def collate_fn(batch):

    texts, labels = zip(*batch)

    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": torch.stack(labels),
    }


# ============================================================
# DATALOADERS
# ============================================================

train_dataset = ToxicityDataset(
    train["text"],
    Y_train
)

valid_dataset = ToxicityDataset(
    valid["text"],
    Y_valid
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
)


# ============================================================
# DISTILBERT MULTI-LABEL CLASSIFIER
# ============================================================

class ToxicityClassifier(nn.Module):

    def __init__(self, num_labels=7):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            CHECKPOINT
        )

        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Linear(
            hidden_size,
            num_labels
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # First token representation.
        pooled = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(
            self.dropout(pooled)
        )

        loss = None

        if labels is not None:

            loss_function = nn.BCEWithLogitsLoss()

            loss = loss_function(
                logits,
                labels.float()
            )

        return {
            "loss": loss,
            "logits": logits,
        }


model = ToxicityClassifier(
    num_labels=len(LABELS)
).to(DEVICE)


# ============================================================
# OPTIMIZER AND SCHEDULER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

steps_per_epoch = math.ceil(
    len(train_loader) / GRAD_ACCUMULATION
)

total_steps = steps_per_epoch * EPOCHS

warmup_steps = int(
    total_steps * WARMUP_RATIO
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

criterion = nn.BCEWithLogitsLoss()


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(model, loader):

    model.eval()

    total_loss = 0.0

    all_probabilities = []
    all_labels = []

    with torch.no_grad():

        for batch in tqdm(
            loader,
            desc="Evaluating",
            leave=False
        ):

            input_ids = batch["input_ids"].to(DEVICE)

            attention_mask = batch[
                "attention_mask"
            ].to(DEVICE)

            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = criterion(
                outputs["logits"],
                labels
            )

            total_loss += loss.item()

            probabilities = torch.sigmoid(
                outputs["logits"]
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_labels.append(
                labels.cpu().numpy()
            )

    probabilities = np.concatenate(
        all_probabilities,
        axis=0
    )

    true_labels = np.concatenate(
        all_labels,
        axis=0
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    results = {}

    average_precisions = []

    for i, label in enumerate(LABELS):

        support = int(
            true_labels[:, i].sum()
        )

        ap = (
            float(
                average_precision_score(
                    true_labels[:, i],
                    probabilities[:, i]
                )
            )
            if support > 0
            else None
        )

        if ap is not None:
            average_precisions.append(ap)

        results[label] = {
            "precision": float(
                precision_score(
                    true_labels[:, i],
                    predictions[:, i],
                    zero_division=0
                )
            ),

            "recall": float(
                recall_score(
                    true_labels[:, i],
                    predictions[:, i],
                    zero_division=0
                )
            ),

            "f1": float(
                f1_score(
                    true_labels[:, i],
                    predictions[:, i],
                    zero_division=0
                )
            ),

            "average_precision": ap,

            "support": support,
        }

    macro_ap = (
        float(np.mean(average_precisions))
        if average_precisions
        else None
    )

    return {
        "validation_loss": total_loss / len(loader),
        "macro_average_precision": macro_ap,
        "per_label_metrics": results,
    }


# ============================================================
# TRAINING LOOP
# ============================================================

best_macro_ap = -float("inf")

history = []

for epoch in range(EPOCHS):

    print("\n" + "=" * 60)

    print(
        f"EPOCH {epoch + 1}/{EPOCHS}"
    )

    print("=" * 60)

    model.train()

    total_train_loss = 0.0

    optimizer.zero_grad(set_to_none=True)

    progress_bar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc="Training"
    )

    for step, batch in progress_bar:

        input_ids = batch["input_ids"].to(DEVICE)

        attention_mask = batch[
            "attention_mask"
        ].to(DEVICE)

        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs["loss"]

        total_train_loss += loss.item()

        (loss / GRAD_ACCUMULATION).backward()

        should_update = (
            (step + 1) % GRAD_ACCUMULATION == 0
            or (step + 1) == len(train_loader)
        )

        if should_update:

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP
            )

            optimizer.step()

            scheduler.step()

            optimizer.zero_grad(set_to_none=True)

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    average_train_loss = (
        total_train_loss / len(train_loader)
    )

    print(
        f"\nTraining loss: {average_train_loss:.4f}"
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    evaluation = evaluate(
        model,
        valid_loader
    )

    macro_ap = evaluation[
        "macro_average_precision"
    ]

    print(
        f"Validation loss: "
        f"{evaluation['validation_loss']:.4f}"
    )

    print(
        f"Validation macro AP: {macro_ap:.4f}"
    )

    for label, metrics in evaluation[
        "per_label_metrics"
    ].items():

        print(
            f"{label:20s} "
            f"F1: {metrics['f1']:.4f} | "
            f"AP: {metrics['average_precision']}"
        )

    history.append({
        "epoch": epoch + 1,
        "training_loss": average_train_loss,
        **evaluation,
    })

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if (
        macro_ap is not None
        and macro_ap > best_macro_ap
    ):

        best_macro_ap = macro_ap

        torch.save(
            model.state_dict(),
            MODEL_DIR / "best_model.pt"
        )

        tokenizer.save_pretrained(
            MODEL_DIR / "tokenizer"
        )

        config = {
            "checkpoint": CHECKPOINT,
            "num_labels": len(LABELS),
            "labels": LABELS,
            "max_length": MAX_LENGTH,
            "annotation_threshold": ANNOTATION_THRESHOLD,
            "decision_threshold": 0.5,
            "hidden_size": model.encoder.config.hidden_size,
            "dropout": 0.2,
            "best_epoch": epoch + 1,
        }

        with open(
            MODEL_DIR / "config.json",
            "w"
        ) as f:

            json.dump(config, f, indent=2)

        with open(
            MODEL_DIR / "best_metrics.json",
            "w"
        ) as f:

            json.dump(evaluation, f, indent=2)

        print("\nSaved new best model!")


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

with open(
    MODEL_DIR / "training_history.json",
    "w"
) as f:

    json.dump(history, f, indent=2)


print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print("Best validation macro AP:", best_macro_ap)

print("Model directory:", MODEL_DIR.resolve())
