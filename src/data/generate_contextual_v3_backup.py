
import random
import json
from pathlib import Path

import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

OUTPUT_DIR = Path("data/experiments/v3")

SPLIT_SIZES = {
    "train": 1000,
    "validation": 250,
    "test": 250,
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

SECONDARY_LABELS = [
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]

COMMUNITIES = [
    "community_alpha",
    "community_beta",
    "community_demo",
]


# Separate vocabulary pools reduce exact lexical overlap
# between the three data splits.

VOCABULARY = {
    "train": {
        "objects": [
            "proposal", "analysis", "design", "report",
            "implementation", "presentation", "argument",
            "research", "solution", "decision",
            "software", "experiment", "document",
            "workflow", "project"
        ],
        "adjectives": [
            "technical", "detailed", "initial", "revised",
            "complex", "practical", "academic",
            "creative", "formal", "experimental"
        ],
        "topics": [
            "software development", "project planning",
            "technical review", "research discussion",
            "team coordination"
        ],
    },

    "validation": {
        "objects": [
            "manuscript", "prototype", "forecast",
            "architecture", "assessment", "strategy",
            "submission", "workflow", "campaign",
            "experiment", "application", "framework"
        ],
        "adjectives": [
            "preliminary", "advanced", "structured",
            "analytical", "proposed", "updated",
            "comprehensive", "innovative"
        ],
        "topics": [
            "product launch", "committee meeting",
            "design evaluation", "deployment planning",
            "performance assessment"
        ],
    },

    "test": {
        "objects": [
            "blueprint", "draft", "model", "framework",
            "assessment", "plan", "documentation",
            "methodology", "proposal", "configuration",
            "technical specification", "prototype"
        ],
        "adjectives": [
            "independent", "modified", "evaluated",
            "systematic", "conceptual", "reviewed",
            "operational", "comparative"
        ],
        "topics": [
            "technical assessment", "research meeting",
            "release planning", "project evaluation",
            "system verification"
        ],
    },
}


# ============================================================
# 2. CONTEXT-SENSITIVE EXAMPLES
# ============================================================


def make_context_pairs(split, count, rng):

    pair_count = count // 10
    vocab = VOCABULARY[split]

    pairs = []

    for i in range(pair_count):

        obj = rng.choice(vocab["objects"])
        adjective = rng.choice(vocab["adjectives"])
        topic = rng.choice(vocab["topics"])

        # Same current comment within a pair,
        # but varied across datasets.
        current_comment = (
            f"I will handle this matter after "
            f"the {adjective} review of {obj} "
            f"case number {i + 1}."
        )

        benign_parent = (
            f"We need someone to coordinate {topic}. "
            f"Can you take responsibility for the "
            f"remaining tasks in case {i + 1}?"
        )

        threatening_parent = (
            f"During the discussion about {topic}, "
            f"you threatened to hurt me if I reported "
            f"your actions in case {i + 1}."
        )

        benign_labels = {
            label: 0 for label in LABELS
        }

        threatening_labels = {
            label: 0 for label in LABELS
        }

        threatening_labels["toxicity"] = 1
        threatening_labels["threat"] = 1

        thread_id = f"{split}_context_thread_{i + 1:05d}"

        pairs.append({
            "comment_id": f"{split}_context_{i + 1:05d}_benign",
            "thread_id": thread_id,
            "text": current_comment,
            "parent_comment": benign_parent,
            "community_id": rng.choice(COMMUNITIES),
            "language": "en",
            **benign_labels,
            "scenario_type": "context_sensitive",
        })

        pairs.append({
            "comment_id": f"{split}_context_{i + 1:05d}_threat",
            "thread_id": thread_id,
            "text": current_comment,
            "parent_comment": threatening_parent,
            "community_id": rng.choice(COMMUNITIES),
            "language": "en",
            **threatening_labels,
            "scenario_type": "context_sensitive",
        })

    return pairs

# ============================================================
# 3. LABEL ASSIGNMENT FOR REGULAR EXAMPLES
# ============================================================


def assign_labels(rows, rng, total_count, context_pair_count):

    # Target counts across the COMPLETE split.
    target_toxicity = total_count // 2
    target_secondary = total_count // 4

    # Context-sensitive pairs already contribute:
    # one benign example and one threatening example.
    regular_toxicity = target_toxicity - context_pair_count

    regular_secondary = {
        label: target_secondary
        for label in SECONDARY_LABELS
    }

    # Threatening context examples already contribute
    # one positive threat label per pair.
    regular_secondary["threat"] -= context_pair_count

    labels = {
        index: {label: 0 for label in LABELS}
        for index in range(len(rows))
    }

    # Select the required number of toxic examples.
    toxic_indices = list(range(len(rows)))
    rng.shuffle(toxic_indices)

    toxic_indices = toxic_indices[:regular_toxicity]

    for index in toxic_indices:
        labels[index]["toxicity"] = 1

    # Give every toxic example at least one
    # secondary moderation label.
    shuffled_toxic = toxic_indices.copy()
    rng.shuffle(shuffled_toxic)

    for position, index in enumerate(shuffled_toxic):
        category = SECONDARY_LABELS[
            position % len(SECONDARY_LABELS)
        ]
        labels[index][category] = 1

    # Add secondary labels until each target is reached.
    for category in SECONDARY_LABELS:

        current_positive = [
            index
            for index in toxic_indices
            if labels[index][category] == 1
        ]

        remaining = (
            regular_secondary[category]
            - len(current_positive)
        )

        candidates = [
            index
            for index in toxic_indices
            if labels[index][category] == 0
        ]

        rng.shuffle(candidates)

        if remaining > len(candidates):
            raise ValueError(
                f"Insufficient examples for label {category}"
            )

        for index in candidates[:remaining]:
            labels[index][category] = 1

    for index, row in enumerate(rows):
        row.update(labels[index])

    return rows

# ============================================================
# 4. GENERATE COMMENT TEXT
# ============================================================

def build_comment(row, split, index, rng):
    """
    Construct a synthetic current comment using
    the assigned labels.
    """

    vocab = VOCABULARY[split]

    obj = rng.choice(vocab["objects"])
    adjective = rng.choice(vocab["adjectives"])

    sentences = []

    if row["severe_toxicity"]:
        sentences.append(
            "You are a disgusting and vile person. "
            "Everyone should despise you."
        )

    elif row["insult"]:
        sentences.append(
            f"You are an incompetent fool who cannot "
            f"understand this {adjective} {obj}."
        )

    elif row["toxicity"]:
        sentences.append(
            f"Your contribution is worthless and "
            f"your {obj} is completely unacceptable."
        )

    else:
        sentences.append(
            f"I have reviewed the {adjective} {obj} "
            f"and would recommend a different approach."
        )

    if row["obscene"]:
        sentences.append(
            "This contains vulgar and obscene language."
        )

    if row["threat"]:
        sentences.append(
            "I will find you and hurt you if you continue."
        )

    if row["identity_attack"]:
        sentences.append(
            "People from your community are inferior "
            "and should not participate."
        )

    if row["sexual_explicit"]:
        sentences.append(
            "The message contains explicit sexual content."
        )

    return " ".join(sentences)


def build_parent(split, rng):
    vocab = VOCABULARY[split]

    topic = rng.choice(vocab["topics"])

    return (
        f"We are discussing {topic}. "
        f"Please share your thoughts on the matter."
    )


# ============================================================
# 5. GENERATE REGULAR EXAMPLES
# ============================================================

def make_regular_examples(split, count, rng):

    rows = [
        {
            "comment_id": f"{split}_regular_{i + 1:05d}",
            "thread_id": f"{split}_regular_thread_{i + 1:05d}",
            "community_id": rng.choice(COMMUNITIES),
            "language": "en",
            "parent_comment": build_parent(split, rng),
            "scenario_type": "unambiguous",
        }
        for i in range(count)
    ]

    rows = assign_labels(rows, rng)

    for index, row in enumerate(rows):
        row["text"] = build_comment(
            row,
            split,
            index,
            rng
        )

    return rows


# ============================================================
# 6. BUILD EACH DATASET
# ============================================================


def make_dataset(split, count):

    rng = random.Random(SEED + count + len(split))

    context_pairs = make_context_pairs(
        split,
        count,
        rng
    )

    pair_count = count // 10

    regular_count = count - len(context_pairs)

    vocab = VOCABULARY[split]

    rows = [
        {
            "comment_id": f"{split}_regular_{i + 1:05d}",
            "thread_id": f"{split}_regular_thread_{i + 1:05d}",
            "community_id": rng.choice(COMMUNITIES),
            "language": "en",
            "parent_comment": build_parent(split, rng),
            "scenario_type": "unambiguous",
        }
        for i in range(regular_count)
    ]

    rows = assign_labels(
        rows,
        rng,
        total_count=count,
        context_pair_count=pair_count
    )

    for index, row in enumerate(rows):

        row["text"] = build_comment(
            row,
            split,
            index,
            rng
        )

    rows = context_pairs + rows

    rng.shuffle(rows)

    return pd.DataFrame(rows)


# ============================================================
# 7. VALIDATE DATASETS
# ============================================================

def validate_datasets(datasets):

    print("\n" + "=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    for name, df in datasets.items():

        assert len(df) == SPLIT_SIZES[name]

        assert df["comment_id"].is_unique

        assert df["thread_id"].notna().all()

        assert df["text"].notna().all()

        assert df["parent_comment"].notna().all()

        assert df[LABELS].isin([0, 1]).all().all()

        assert (
            df["toxicity"] >= df[SECONDARY_LABELS].max(axis=1)
        ).all()

        print(f"\n{name.upper()}")
        print("Rows:", len(df))
        print("Unique IDs:", df["comment_id"].nunique())
        print("Unique threads:", df["thread_id"].nunique())
        print("Missing values:", df.isnull().sum().sum())

        print("\nPositive label counts:")
        print(df[LABELS].sum().to_dict())

        # Verify exact target label counts.
        expected_counts = {
            "toxicity": len(df) // 2,
            "severe_toxicity": len(df) // 4,
            "obscene": len(df) // 4,
            "threat": len(df) // 4,
            "insult": len(df) // 4,
            "identity_attack": len(df) // 4,
            "sexual_explicit": len(df) // 4,
        }

        actual_counts = df[LABELS].sum().to_dict()

        assert actual_counts == expected_counts, (
            f"Label imbalance in {name}: "
            f"expected {expected_counts}, got {actual_counts}"
        )

        print("\nScenario distribution:")
        print(df["scenario_type"].value_counts().to_dict())

    # Check cross-split exact comment-parent pairs.
    names = list(datasets.keys())

    for i in range(len(names)):
        for j in range(i + 1, len(names)):

            left_name = names[i]
            right_name = names[j]

            left = datasets[left_name]
            right = datasets[right_name]

            left_pairs = set(
                zip(left["text"], left["parent_comment"])
            )

            right_pairs = set(
                zip(right["text"], right["parent_comment"])
            )

            overlap = left_pairs & right_pairs

            assert len(overlap) == 0, (
                f"Pair leakage: {left_name} vs {right_name}"
            )

            left_threads = set(left["thread_id"])
            right_threads = set(right["thread_id"])

            assert not (
                left_threads & right_threads
            ), "Thread ID leakage detected."

    print("\nAll structural and cross-split checks passed.")


# ============================================================
# 8. MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {}

    for split, count in SPLIT_SIZES.items():

        print(f"\nGenerating {split} dataset...")

        df = make_dataset(split, count)

        datasets[split] = df

    validate_datasets(datasets)

    # Save only after validation succeeds.
    for split, df in datasets.items():

        output_path = OUTPUT_DIR / f"{split}.csv"

        df.to_csv(output_path, index=False)

        print(f"Saved: {output_path}")

    # Save a reproducibility report.
    report = {
        "dataset_version": "v3",
        "random_seed": SEED,
        "synthetic": True,
        "label_design": {
            "toxicity": "50% positive",
            "secondary_labels": "25% positive each",
            "hierarchy": "secondary labels are subsets of toxicity",
        },
        "split_sizes": SPLIT_SIZES,
        "labels": LABELS,
        "limitations": [
            "Programmatically generated examples",
            "Rule-assigned labels, not human annotations",
            "Synthetic parent-comment relationships",
            "Not representative of real-world prevalence",
        ],
    }

    with open(
        OUTPUT_DIR / "dataset_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=4)

    print("\nV3 generation completed successfully.")


if __name__ == "__main__":
    main()
