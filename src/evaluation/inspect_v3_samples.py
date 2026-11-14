
import pandas as pd

df = pd.read_csv("data/experiments/v3/train.csv")

labels = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]

# Exclude contextual examples for this inspection.
regular = df[
    df["scenario_type"] == "unambiguous"
]

for label in labels:

    print("\n" + "=" * 70)
    print("LABEL:", label.upper())
    print("=" * 70)

    positive = regular[regular[label] == 1].sample(
        n=min(3, len(regular[regular[label] == 1])),
        random_state=42
    )

    negative = regular[regular[label] == 0].sample(
        n=min(2, len(regular[regular[label] == 0])),
        random_state=42
    )

    print("\n--- POSITIVE EXAMPLES ---")

    for _, row in positive.iterrows():
        print("\nComment:", row["text"])
        print("Parent:", row["parent_comment"])
        print(
            "Labels:",
            {
                name: int(row[name])
                for name in labels
            }
        )

    print("\n--- NEGATIVE EXAMPLES ---")

    for _, row in negative.iterrows():
        print("\nComment:", row["text"])
        print("Parent:", row["parent_comment"])
        print(
            "Labels:",
            {
                name: int(row[name])
                for name in labels
            }
        )
