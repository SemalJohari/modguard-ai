import json
from pathlib import Path

import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel

from app.context import build_contextual_text


MODEL_DIR = Path("models/contextual_02")


class ContextualClassifier(nn.Module):

    def __init__(self, checkpoint, num_labels, dropout=0.2):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(checkpoint)

        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            hidden_size,
            num_labels
        )

    def forward(self, input_ids, attention_mask):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.last_hidden_state[:, 0, :]

        return self.classifier(
            self.dropout(pooled_output)
        )


class ModerationModel:

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        with open(
            MODEL_DIR / "config.json",
            "r"
        ) as file:
            self.config = json.load(file)

        self.labels = self.config["labels"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR / "tokenizer"
        )

        self.model = ContextualClassifier(
            checkpoint=self.config["checkpoint"],
            num_labels=len(self.labels),
            dropout=self.config["dropout"]
        )

        state_dict = torch.load(
            MODEL_DIR / "best_model.pt",
            map_location=self.device
        )

        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        current_comment,
        parent_comment=None
    ):

        contextual_text = build_contextual_text(
            current_comment,
            parent_comment
        )

        inputs = self.tokenizer(
            contextual_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config["max_length"]
        )

        inputs.pop("token_type_ids", None)

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            logits = self.model(**inputs)

            probabilities = torch.sigmoid(
                logits
            )[0].cpu().numpy()

        return {
            label: float(probability)
            for label, probability in zip(
                self.labels,
                probabilities
            )
        }
        