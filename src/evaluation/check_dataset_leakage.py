
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/contextual")

files = {
    "train": DATA_DIR / "contextual_train.csv",
    "validation": DATA_DIR / "contextual_validation.csv",
    "test": DATA_DIR / "contextual_test.csv",
}

datasets = {}

for name, path in files.items():
    df = pd.read_csv(path)

    # Normalize text for duplicate detection.
    df["normalized_text"] = (
        df["text"]
        .fillna("")
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    df["normalized_parent"] = (
        df["parent_comment"]
        .fillna("")
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    datasets[name] = df

    print(f"\n{name.upper()}")
    print("Rows:", len(df))
    print(
        "Duplicate current comments:",
        df["normalized_text"].duplicated().sum()
    )
    print(
        "Duplicate comment-parent pairs:",
        df.duplicated(
            subset=["normalized_text", "normalized_parent"]
        ).sum()
    )

print("\n" + "=" * 55)
print("CROSS-SPLIT OVERLAP")
print("=" * 55)

pairs = [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
]

for left, right in pairs:
    df_left = datasets[left]
    df_right = datasets[right]

    left_texts = set(df_left["normalized_text"])
    right_texts = set(df_right["normalized_text"])

    text_overlap = left_texts & right_texts

    left_pairs = set(
        zip(
            df_left["normalized_text"],
            df_left["normalized_parent"],
        )
    )

    right_pairs = set(
        zip(
            df_right["normalized_text"],
            df_right["normalized_parent"],
        )
    )

    pair_overlap = left_pairs & right_pairs

    print(f"\n{left} vs {right}")
    print("Overlapping unique comments:", len(text_overlap))
    print("Overlapping unique comment-parent pairs:", len(pair_overlap))

    if pair_overlap:
        print("Example overlapping pair:")
        example = next(iter(sorted(pair_overlap)))
        print("Current:", example[0][:150])
        print("Parent:", example[1][:150])

print("\nAudit completed.")
