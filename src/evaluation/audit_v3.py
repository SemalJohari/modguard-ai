
import pandas as pd
from pathlib import Path

BASE = Path("data/experiments/v3")

SPLITS = {
    "train": BASE / "train.csv",
    "validation": BASE / "validation.csv",
    "test": BASE / "test.csv",
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

SECONDARY_LABELS = LABELS[1:]

EXPECTED_ROWS = {
    "train": 1000,
    "validation": 250,
    "test": 250,
}


def pair_set(df):
    return set(
        zip(
            df["text"].fillna(""),
            df["parent_comment"].fillna("")
        )
    )


datasets = {}

print("=" * 60)
print("MODGUARD AI - V3 DATASET AUDIT")
print("=" * 60)

# --------------------------------------------------
# 1. Individual split checks
# --------------------------------------------------

for split, path in SPLITS.items():

    print(f"\n{'=' * 20} {split.upper()} {'=' * 20}")

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    df = pd.read_csv(path)
    datasets[split] = df

    print("Rows:", len(df))
    print("Expected rows:", EXPECTED_ROWS[split])
    print("Unique comment IDs:", df["comment_id"].nunique())
    print("Missing values:", int(df.isnull().sum().sum()))

    print("\nDuplicate current comments:")
    print(df["text"].duplicated().sum())

    print("Duplicate comment-parent pairs:")
    print(
        df.duplicated(
            subset=["text", "parent_comment"]
        ).sum()
    )

    print("\nScenario distribution:")
    print(df["scenario_type"].value_counts())

    print("\nPositive label counts:")
    actual_counts = df[LABELS].sum().astype(int)
    print(actual_counts)

    expected_toxicity = len(df) // 2
    expected_secondary = len(df) // 4

    expected_counts = {
        "toxicity": expected_toxicity,
        **{
            label: expected_secondary
            for label in SECONDARY_LABELS
        }
    }

    print("\nLabel distribution valid:",
          actual_counts.to_dict() == expected_counts)

    # Basic structural checks
    assert len(df) == EXPECTED_ROWS[split]
    assert df["comment_id"].is_unique
    assert df[LABELS].isin([0, 1]).all().all()
    assert df[LABELS].notna().all().all()

    # Every secondary positive must also be toxic.
    hierarchy_valid = (
        df[SECONDARY_LABELS].eq(1).any(axis=1)
        <= df["toxicity"].eq(1)
    ).all()

    print("Toxicity hierarchy valid:", hierarchy_valid)
    assert hierarchy_valid

    # --------------------------------------------------
    # 2. Contextual pair validation
    # --------------------------------------------------

    contextual = df[
        df["scenario_type"] == "context_sensitive"
    ]

    pair_errors = []

    for thread_id, group in contextual.groupby("thread_id"):

        if len(group) != 2:
            pair_errors.append(
                f"{thread_id}: expected 2 rows, got {len(group)}"
            )
            continue

        if group["text"].nunique() != 1:
            pair_errors.append(
                f"{thread_id}: current comments differ"
            )
            continue

        benign = group[
            group["toxicity"] == 0
        ]

        toxic = group[
            group["toxicity"] == 1
        ]

        if len(benign) != 1 or len(toxic) != 1:
            pair_errors.append(
                f"{thread_id}: expected one benign and one toxic row"
            )
            continue

        benign_row = benign.iloc[0]
        toxic_row = toxic.iloc[0]

        if benign_row[LABELS].sum() != 0:
            pair_errors.append(
                f"{thread_id}: benign row has positive labels"
            )

        if (
            toxic_row["toxicity"] != 1
            or toxic_row["insult"] != 1
            or toxic_row["threat"] != 0
        ):
            pair_errors.append(
                f"{thread_id}: unexpected toxic-row labels"
            )

    print("\nContextual pairs:", contextual["thread_id"].nunique())
    print("Contextual pair errors:", len(pair_errors))

    for error in pair_errors[:10]:
        print("ERROR:", error)

    assert not pair_errors, "Contextual pair validation failed"


# --------------------------------------------------
# 3. Cross-split leakage checks
# --------------------------------------------------

print("\n" + "=" * 60)
print("CROSS-SPLIT LEAKAGE AUDIT")
print("=" * 60)

for first, second in [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
]:

    a = datasets[first]
    b = datasets[second]

    comment_overlap = (
        set(a["text"]) & set(b["text"])
    )

    parent_overlap = (
        set(a["parent_comment"]) &
        set(b["parent_comment"])
    )

    pair_overlap = pair_set(a) & pair_set(b)

    thread_overlap = (
        set(a["thread_id"]) &
        set(b["thread_id"])
    )

    print(f"\n{first} vs {second}")
    print("Current-comment overlap:", len(comment_overlap))
    print("Parent-comment overlap:", len(parent_overlap))
    print("Comment-parent pair overlap:", len(pair_overlap))
    print("Thread ID overlap:", len(thread_overlap))

    assert not pair_overlap, "Exact pair leakage detected"
    assert not thread_overlap, "Thread leakage detected"


print("\n" + "=" * 60)
print("V3 AUDIT COMPLETED SUCCESSFULLY")
print("=" * 60)
