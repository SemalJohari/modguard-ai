
from datasets import load_dataset
from pathlib import Path

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

dataset = load_dataset("google/civil_comments")

print(dataset)
print(dataset["train"].column_names)

# Randomly sample 50,000 training examples.
train = dataset["train"].shuffle(seed=42).select(
    range(50_000)
)

# Preserve the complete official validation and test splits.
validation = dataset["validation"]
test = dataset["test"]

train.to_parquet(OUTPUT_DIR / "train_dev.parquet")
validation.to_parquet(OUTPUT_DIR / "validation.parquet")
test.to_parquet(OUTPUT_DIR / "test.parquet")

print("\nDataset preparation complete.")
print("Training:", len(train))
print("Validation:", len(validation))
print("Test:", len(test))
