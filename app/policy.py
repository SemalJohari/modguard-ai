
from dataclasses import dataclass
from pathlib import Path

import yaml


POLICY_PATH = Path("configs/moderation_policy.yaml")


@dataclass
class Decision:
    action: str
    reasons: list[str]
    review_required: bool


def load_policy():
    """Load community moderation policies from YAML."""

    with open(POLICY_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def moderation_decision(
    probabilities: dict[str, float],
    thresholds: dict[str, float],
    actions: dict[str, str],
    uncertainty_band: float = 0.10,
) -> Decision:

    violations = []
    uncertain = []

    for label, probability in probabilities.items():

        threshold = thresholds.get(label, 0.5)

        if abs(probability - threshold) <= uncertainty_band:
            uncertain.append(label)

        if probability >= threshold:
            violations.append(label)

    # Uncertain predictions always require human review.
    if uncertain:
        return Decision(
            action="manual_review",
            reasons=sorted(set(uncertain + violations)),
            review_required=True,
        )

    # Determine the strongest configured action.
    if violations:

        triggered_actions = [
            actions.get(label, "flag").lower()
            for label in violations
        ]

        if "escalate" in triggered_actions:
            action = "escalate"
        elif "remove" in triggered_actions:
            action = "remove"
        else:
            action = "flag"

        # Keep automatic removal/escalation disabled
        # until the model and policies are validated.
        return Decision(
            action="manual_review",
            reasons=violations,
            review_required=True,
        )

    return Decision(
        action="allow",
        reasons=[],
        review_required=False,
    )


def evaluate_policy(
    predictions: dict[str, float],
    community_id: str,
) -> dict:

    policy = load_policy()

    communities = policy.get("communities", {})

    community = communities.get(
        community_id,
        policy["default_community"]
    )

    thresholds = community.get("thresholds", {})
    actions = community.get("actions", {})

    result = moderation_decision(
        probabilities=predictions,
        thresholds=thresholds,
        actions=actions,
    )

    return {
        "decision": result.action,
        "triggered_rules": result.reasons,
        "review_required": result.review_required,
        "policy_version": policy.get("policy_version", "1.0"),
    }

