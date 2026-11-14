
import random
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

OUTPUT_DIR = Path("data/experiments/v2")

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

COMMUNITIES = [
    "community_alpha",
    "community_beta",
    "community_demo",
]

# Each split receives different vocabulary and templates.
# This reduces exact template leakage across splits.
SPLIT_VOCAB = {
    "train": {
        "objects": [
            "proposal", "analysis", "design", "report",
            "implementation", "presentation", "argument",
            "research", "solution", "decision",
        ],
        "targets": [
            "your explanation", "your suggestion",
            "your argument", "your approach",
            "your contribution",
        ],
        "topics": [
            "the software release", "the project",
            "the discussion", "the experiment",
            "the review",
        ],
    },
    "validation": {
        "objects": [
            "manuscript", "prototype", "forecast",
            "architecture", "assessment", "strategy",
            "submission", "workflow", "campaign",
            "experiment",
        ],
        "targets": [
            "your interpretation", "your proposal",
            "your reasoning", "your recommendation",
            "your conclusion",
        ],
        "topics": [
            "the product launch", "the committee meeting",
            "the design review", "the deployment",
            "the evaluation",
        ],
    },
    "test": {
        "objects": [
            "blueprint", "draft", "model", "framework",
            "assessment", "plan", "documentation",
            "presentation", "methodology", "proposal",
        ],
        "targets": [
            "your analysis", "your decision",
            "your reasoning", "your recommendation",
            "your interpretation",
        ],
        "topics": [
            "the technical review", "the research meeting",
            "the release", "the project discussion",
            "the assessment",
        ],
    },
}


# ============================================================
# SCENARIO TEMPLATES
# ============================================================

def make_examples(split, count):
    vocab = SPLIT_VOCAB[split]
    rng = random.Random(SEED + count)

    examples = []

    # Eight scenario categories:
    # clean + seven moderation labels.
    categories = [
        "clean",
        "toxicity",
        "severe_toxicity",
        "obscene",
        "threat",
        "insult",
        "identity_attack",
        "sexual_explicit",
    ]

    # Balance the categories as evenly as possible.
    category_sequence = [
        categories[i % len(categories)]
        for i in range(count)
    ]

    rng.shuffle(category_sequence)

    for index, category in enumerate(category_sequence):

        obj = rng.choice(vocab["objects"])
        target = rng.choice(vocab["targets"])
        topic = rng.choice(vocab["topics"])

        # Default: no moderation labels.
        labels = {label: 0 for label in LABELS}

        scenario_type = "unambiguous"

        if category == "clean":
            current = (
                f"I have reviewed the {obj} and "
                f"would recommend a different approach."
            )
            parent = (
                f"We are discussing {topic}. "
                f"What do you think?"
            )

        elif category == "toxicity":
            current = (
                f"Your {obj} is completely worthless "
                f"and you should be ashamed."
            )
            parent = f"Please provide feedback about {topic}."
            labels["toxicity"] = 1
            labels["insult"] = 1

        elif category == "severe_toxicity":
            current = (
                f"You are a disgusting and vile person. "
                f"Everyone should despise you."
            )
            parent = f"We are discussing {topic}."
            labels["toxicity"] = 1
            labels["severe_toxicity"] = 1
            labels["insult"] = 1

        elif category == "obscene":
            current = (
                f"This is a vulgar and obscene piece of work."
            )
            parent = f"Please review {topic}."
            labels["toxicity"] = 1
            labels["obscene"] = 1

        elif category == "threat":
            current = (
                f"I will find you and hurt you "
                f"if you continue this."
            )
            parent = f"The discussion concerns {topic}."
            labels["toxicity"] = 1
            labels["threat"] = 1

        elif category == "insult":
            current = (
                f"You are an incompetent fool "
                f"who cannot understand {obj}."
            )
            parent = f"We are reviewing {topic}."
            labels["toxicity"] = 1
            labels["insult"] = 1

        elif category == "identity_attack":
            current = (
                f"People from your identity group "
                f"are inferior and unwelcome here."
            )
            parent = f"We are discussing {topic}."
            labels["toxicity"] = 1
            labels["identity_attack"] = 1

        elif category == "sexual_explicit":
            current = (
                f"The message contains explicit sexual descriptions."
            )
            parent = f"Please review the content in {topic}."
            labels["toxicity"] = 1
            labels["sexual_explicit"] = 1

        # Add context-sensitive examples.
        # The current comment alone is ambiguous;
        # the parent supplies the intended context.
        if index % 5 == 0:
            scenario_type = "context_sensitive"

            ambiguous_comments = [
                "That is sick.",
                "You are killing it.",
                "I will deal with you.",
                "That was a dirty move.",
                "This is insane.",
            ]

            current = rng.choice(ambiguous_comments)

            if category == "clean":
                parent = (
                    f"We are celebrating the excellent result "
                    f"of {topic}."
                )

            elif category == "threat":
                current = "I will deal with you."
                parent = (
                    f"You threatened to hurt me during {topic}."
                )
                labels["toxicity"] = 1
                labels["threat"] = 1

            elif category == "insult":
                current = "That is sick."
                parent = (
                    f"You are mocking me during {topic}."
                )
                labels["toxicity"] = 1
                labels["insult"] = 1

            elif category == "obscene":
                current = "That was a dirty move."
                parent = (
                    f"The discussion concerns obscene language "
                    f"used during {topic}."
                )
                labels["toxicity"] = 1
                labels["obscene"] = 1

            else:
                # Keep other category labels consistent with
                # the category, while retaining an ambiguous
                # current-comment example.
                pass

        examples.append({
            "comment_id": f"{split}_{index + 1:05d}",
            "thread_id": f"{split}_thread_{index + 1:05d}",
            "text": current,
            "parent_comment": parent,
            "community_id": rng.choice(COMMUNITIES),
            "language": "en",
            **labels,
            "scenario_type": scenario_type,
        })

    return examples


# ============================================================
# GENERATE DATASETS
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split, count in SPLIT_SIZES.items():

        examples = make_examples(split, count)

        df = pd.DataFrame(examples)

        output_path = OUTPUT_DIR / f"{split}.csv"

        df.to_csv(output_path, index=False)

        print(f"\nCreated: {output_path}")
        print("Rows:", len(df))
        print("Unique comments:", df["text"].nunique())
        print(
            "Unique comment-parent pairs:",
            df[["text", "parent_comment"]]
            .drop_duplicates()
            .shape[0],
        )

        print("\nLabel distribution:")
        print(df[LABELS].sum().to_string())

        print("\nScenario distribution:")
        print(df["scenario_type"].value_counts().to_string())

    print("\nAll synthetic datasets generated.")


if __name__ == "__main__":
    main()
