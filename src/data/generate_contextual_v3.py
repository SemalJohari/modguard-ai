import json
import random
from pathlib import Path

import pandas as pd

# Synthetic, rule-generated dataset for pipeline development and demos.
# These labels are NOT human annotations or evidence of real-world accuracy.

SEED = 42
OUTPUT_DIR = Path("data/experiments/v3")
SPLIT_SIZES = {"train": 1000, "validation": 250, "test": 250}

LABELS = [
    "toxicity", "severe_toxicity", "obscene", "threat",
    "insult", "identity_attack", "sexual_explicit",
]
SECONDARY_LABELS = LABELS[1:]
COMMUNITIES = ["community_alpha", "community_beta", "community_demo"]

VOCABULARY = {
    "train": {
        "objects": ["proposal", "analysis", "design", "report", "implementation",
                    "presentation", "argument", "research", "solution", "decision",
                    "software", "experiment", "document", "workflow", "project"],
        "adjectives": ["technical", "detailed", "initial", "revised", "complex",
                       "practical", "academic", "creative", "formal", "experimental"],
        "topics": ["software development", "project planning", "technical review",
                   "research discussion", "team coordination"],
    },
    "validation": {
        "objects": ["manuscript", "prototype", "forecast", "architecture",
                    "assessment", "strategy", "submission", "workflow", "campaign",
                    "experiment", "application", "framework"],
        "adjectives": ["preliminary", "advanced", "structured", "analytical",
                       "proposed", "updated", "comprehensive", "innovative"],
        "topics": ["product launch", "committee meeting", "design evaluation",
                   "deployment planning", "performance assessment"],
    },
    "test": {
        "objects": ["blueprint", "draft", "model", "framework", "assessment", "plan",
                    "documentation", "methodology", "proposal", "configuration",
                    "technical specification", "prototype"],
        "adjectives": ["independent", "modified", "evaluated", "systematic",
                       "conceptual", "reviewed", "operational", "comparative"],
        "topics": ["technical assessment", "research meeting", "release planning",
                   "project evaluation", "system verification"],
    },
}


def make_context_pairs(split, count, rng):
    """Illustrative context pairs; labels require human validation."""
    pair_count = count // 10
    vocab = VOCABULARY[split]
    pairs = []

    for i in range(pair_count):
        topic = rng.choice(vocab["topics"])
        obj = rng.choice(vocab["objects"])
        adjective = rng.choice(vocab["adjectives"])
        current = "That is pathetic."

        benign_parent = (
            f"The team's performance during the {topic} was disappointing. "
            f"We lost the final round despite months of preparation for "
            f"the {adjective} {obj} project."
        )
        toxic_parent = (
            f"I shared my personal struggles with the group during the "
            f"{topic}. I was explaining my situation and asking everyone "
            f"for support with the {adjective} {obj} project."
        )

        benign_labels = {label: 0 for label in LABELS}
        toxic_labels = {label: 0 for label in LABELS}
        toxic_labels["toxicity"] = 1
        toxic_labels["insult"] = 1

        thread_id = f"{split}_context_thread_{i + 1:05d}"
        community_id = rng.choice(COMMUNITIES)

        pairs.extend([
            {
                "comment_id": f"{split}_context_{i + 1:05d}_benign",
                "thread_id": thread_id, "text": current,
                "parent_comment": benign_parent,
                "community_id": community_id, "language": "en",
                **benign_labels, "scenario_type": "context_sensitive",
            },
            {
                "comment_id": f"{split}_context_{i + 1:05d}_toxic",
                "thread_id": thread_id, "text": current,
                "parent_comment": toxic_parent,
                "community_id": community_id, "language": "en",
                **toxic_labels, "scenario_type": "context_sensitive",
            },
        ])
    return pairs


def assign_labels(rows, rng, total_count, context_pair_count):
    """Assign exact target counts; all secondary labels imply toxicity."""
    target_toxicity = total_count // 2
    target_secondary = total_count // 4
    regular_toxicity = target_toxicity - context_pair_count

    regular_targets = {label: target_secondary for label in SECONDARY_LABELS}
    regular_targets["insult"] -= context_pair_count

    if regular_toxicity < 0 or any(v < 0 for v in regular_targets.values()):
        raise ValueError("Context pairs exceed a label target.")

    assigned = {i: {label: 0 for label in LABELS} for i in range(len(rows))}
    toxic_indices = list(range(len(rows)))
    rng.shuffle(toxic_indices)
    toxic_indices = toxic_indices[:regular_toxicity]

    for i in toxic_indices:
        assigned[i]["toxicity"] = 1

    for category in SECONDARY_LABELS:
        candidates = toxic_indices.copy()
        rng.shuffle(candidates)
        needed = regular_targets[category]
        if needed > len(candidates):
            raise ValueError(f"Insufficient toxic rows for {category}")
        for i in candidates[:needed]:
            assigned[i][category] = 1

    for i, row in enumerate(rows):
        row.update(assigned[i])
    return rows


def build_comment(row, split, index, rng):
    """Every positive label adds text intended to express that category."""
    vocab = VOCABULARY[split]
    obj = rng.choice(vocab["objects"])
    adjective = rng.choice(vocab["adjectives"])
    sentences = []

    if row["toxicity"]:
        sentences.append(
            f"Your contribution is worthless and your {obj} is unacceptable."
        )
    else:
        sentences.append(
            f"I reviewed the {adjective} {obj} and recommend a different approach."
        )

    if row["severe_toxicity"]:
        sentences.append(
            "I want to humiliate you and make you suffer for what you did."
        )
    if row["obscene"]:
        sentences.append("This is complete shit, and I am furious about it.")
    if row["threat"]:
        sentences.append("I will find you and hurt you if you continue.")
    if row["insult"]:
        sentences.append(
            f"You are an incompetent fool who cannot understand this {obj}."
        )
    if row["identity_attack"]:
        # Fictional target group: not a substitute for representative data.
        sentences.append(
            "People from the fictional Group Z are inferior and should be excluded."
        )
    if row["sexual_explicit"]:
        sentences.append("I want to have sex with you.")

    return " ".join(sentences)


def build_parent(split, rng):
    topic = rng.choice(VOCABULARY[split]["topics"])
    return f"We are discussing {topic}. Please share your thoughts on the matter."


def make_dataset(split, count):
    rng = random.Random(SEED + count + len(split))
    context_pairs = make_context_pairs(split, count, rng)
    pair_count = count // 10
    regular_count = count - len(context_pairs)

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

    rows = assign_labels(rows, rng, count, pair_count)
    for index, row in enumerate(rows):
        row["text"] = build_comment(row, split, index, rng)

    rows.extend(context_pairs)
    rng.shuffle(rows)
    return pd.DataFrame(rows)


def validate_datasets(datasets):
    print("\n" + "=" * 60)
    print("V3 DATASET VALIDATION")
    print("=" * 60)

    for split, df in datasets.items():
        assert len(df) == SPLIT_SIZES[split], f"{split}: wrong row count"
        assert df["comment_id"].is_unique, f"{split}: duplicate comment IDs"
        assert df["thread_id"].notna().all(), f"{split}: missing thread IDs"
        assert df["text"].notna().all(), f"{split}: missing text"
        assert df["parent_comment"].notna().all(), f"{split}: missing parents"
        assert df[LABELS].notna().all().all(), f"{split}: missing labels"
        assert df[LABELS].isin([0, 1]).all().all(), f"{split}: non-binary labels"
        assert (
            df[SECONDARY_LABELS].max(axis=1) <= df["toxicity"]
        ).all(), f"{split}: secondary label without toxicity"

        expected = {
            "toxicity": len(df) // 2,
            **{label: len(df) // 4 for label in SECONDARY_LABELS},
        }
        actual = df[LABELS].sum().astype(int).to_dict()
        assert actual == expected, (
            f"{split}: expected label counts {expected}, got {actual}"
        )

        contextual = df[df["scenario_type"] == "context_sensitive"]
        errors = []
        for thread_id, group in contextual.groupby("thread_id"):
            if len(group) != 2 or group["text"].nunique() != 1:
                errors.append(thread_id)
                continue
            benign = group[group["toxicity"] == 0]
            toxic = group[group["toxicity"] == 1]
            if len(benign) != 1 or len(toxic) != 1:
                errors.append(thread_id)
                continue
            if benign.iloc[0][LABELS].sum() != 0:
                errors.append(thread_id)
            if not (
                toxic.iloc[0]["toxicity"] == 1
                and toxic.iloc[0]["insult"] == 1
            ):
                errors.append(thread_id)

        assert not errors, f"{split}: contextual pair errors {errors[:5]}"

        print(f"\n{split.upper()}")
        print("Rows:", len(df))
        print("Unique comment IDs:", df["comment_id"].nunique())
        print("Missing values:", int(df.isnull().sum().sum()))
        print("Positive label counts:", actual)
        print("Contextual pairs:", contextual["thread_id"].nunique())
        print("Contextual pair errors:", len(errors))

    names = list(datasets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            left_name, right_name = names[i], names[j]
            left, right = datasets[left_name], datasets[right_name]
            left_pairs = set(zip(left["text"], left["parent_comment"]))
            right_pairs = set(zip(right["text"], right["parent_comment"]))
            assert not (left_pairs & right_pairs), (
                f"Comment-parent pair leakage: {left_name} vs {right_name}"
            )
            assert not (set(left["thread_id"]) & set(right["thread_id"])), (
                f"Thread ID leakage: {left_name} vs {right_name}"
            )

    print("\nAll structural and cross-split checks passed.")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {}

    for split, count in SPLIT_SIZES.items():
        print(f"Generating {split} dataset...")
        datasets[split] = make_dataset(split, count)

    # Save only after every split passes validation.
    validate_datasets(datasets)

    for split, df in datasets.items():
        path = OUTPUT_DIR / f"{split}.csv"
        df.to_csv(path, index=False)
        print(f"Saved: {path}")

    report = {
        "dataset_version": "v3_revised",
        "random_seed": SEED,
        "synthetic": True,
        "annotation_method": "programmatic label-driven templates",
        "split_sizes": SPLIT_SIZES,
        "labels": LABELS,
        "label_targets": {
            "toxicity": "50% positive",
            "each_secondary_label": "25% positive",
        },
        "limitations": [
            "Rule-generated examples, not human annotations",
            "Fictional Group Z used for identity-attack examples",
            "Contextual pairs are illustrative hypotheses requiring review",
            "Not representative of real-world prevalence or language",
            "Requires independent human-annotated evaluation",
        ],
    }
    with (OUTPUT_DIR / "dataset_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\nV3 revised generation completed successfully.")


if __name__ == "__main__":
    main()
