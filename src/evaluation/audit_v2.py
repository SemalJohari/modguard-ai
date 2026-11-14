
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/experiments/v2")

files = {
    "train": DATA_DIR / "train.csv",
    "validation": DATA_DIR / "validation.csv",
    "test": DATA_DIR / "test.csv",
}

LABELS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]


def normalize(series):
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


datasets = {}

for name, path in files.items():
    df = pd.read_csv(path)

    df["normalized_text"] = normalize(df["text"])
    df["normalized_parent"] = normalize(df["parent_comment"])

    datasets[name] = df

    print(f"\n{'=' * 55}")
    print(name.upper())

    print("Rows:", len(df))

    print(
        "Duplicate current comments:",
        df["normalized_text"].duplicated().sum()
    )

    print(
        "Duplicate comment-parent pairs:",
        df.duplicated(
            ["normalized_text", "normalized_parent"]
        ).sum()
    )

    print("\nLabel distribution:")
    print(df[LABELS].sum())

    print("\nScenario distribution:")
    print(df["scenario_type"].value_counts())


print("\n" + "=" * 55)
print("CROSS-SPLIT OVERLAP")
print("=" * 55)

pairs = [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
]

for left, right in pairs:

    a = datasets[left]
    b = datasets[right]

    text_overlap = (
        set(a["normalized_text"])
        & set(b["normalized_text"])
    )

    parent_overlap = (
        set(a["normalized_parent"])
        & set(b["normalized_parent"])
    )

    pair_a = set(
        zip(a["normalized_text"], a["normalized_parent"])
    )

    pair_b = set(
        zip(b["normalized_text"], b["normalized_parent"])
    )

    pair_overlap = pair_a & pair_b

    print(f"\n{left} vs {right}")
    print("Current-comment overlap:", len(text_overlap))
    print("Parent-comment overlap:", len(parent_overlap))
    print("Comment-parent pair overlap:", len(pair_overlap))


print("\n" + "=" * 55)
print("LABEL VALIDITY CHECK")
print("=" * 55)

for name, df in datasets.items():

    invalid = (
        ~df[LABELS].isin([0, 1])
    ).any().any()

    print(
        name,
        "Binary labels valid:",
        not invalid
    )

    print(
        name,
        "Rows with no positive labels:",
        (df[LABELS].sum(axis=1) == 0).sum()
    )

print("\nAudit completed.")
