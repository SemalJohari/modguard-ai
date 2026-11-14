
import pandas as pd

df = pd.read_parquet("data/raw/train_dev.parquet")

label_columns = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]

print("Rows:", len(df))
print("Missing text:", df["text"].isna().sum())
print("Duplicate text:", df["text"].duplicated().sum())

for label in label_columns:
    print(
        label,
        "mean score =", df[label].mean(),
        "positive at 0.5 =", (df[label] >= 0.5).mean()
    )
