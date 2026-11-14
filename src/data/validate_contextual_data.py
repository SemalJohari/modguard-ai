from pathlib import Path
import csv
import pandas as pd

DATA_DIR = Path("data/contextual")

FILES = [
    "contextual_train.csv",
    "contextual_validation.csv",
    "contextual_test.csv"
]

LABELS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit"
]

REQUIRED_COLUMNS = [
    "comment_id",
    "text",
    "parent_comment",
    "community_id",
    "language",
    *LABELS
]

EXPECTED_FIELDS = len(REQUIRED_COLUMNS)


def check_csv_structure(path):

    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)

        header = next(reader)

        if len(header) != EXPECTED_FIELDS:
            raise ValueError(
                f"{path}: Expected {EXPECTED_FIELDS} header columns, "
                f"found {len(header)}."
            )

        for line_number, row in enumerate(reader, start=2):

            if len(row) != EXPECTED_FIELDS:
                raise ValueError(
                    f"\nCSV formatting error in {path}\n"
                    f"Line: {line_number}\n"
                    f"Expected fields: {EXPECTED_FIELDS}\n"
                    f"Actual fields: {len(row)}\n"
                    f"Problematic row:\n{row}\n"
                    f"Check for unquoted commas."
                )


for filename in FILES:

    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    check_csv_structure(path)

    df = pd.read_csv(path)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"{filename}: Missing columns {missing}"
        )

    if df["comment_id"].duplicated().any():
        raise ValueError(
            f"{filename}: Duplicate comment IDs"
        )

    mandatory = [
        "text",
        "community_id",
        "language",
        *LABELS
    ]

    if df[mandatory].isnull().any().any():
        raise ValueError(
            f"{filename}: Missing mandatory values"
        )

    df["parent_comment"] = df["parent_comment"].fillna("")

    for label in LABELS:

        if not set(df[label].unique()).issubset({0, 1}):
            raise ValueError(
                f"{filename}: Invalid values in {label}"
            )

    print(f"\n{filename}")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Positive labels:")
    print(df[LABELS].sum().to_string())

print("\nAll contextual CSV files passed structural validation.")