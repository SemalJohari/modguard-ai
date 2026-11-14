
from typing import Optional
from pydantic import BaseModel, Field


class ModerationRequest(BaseModel):
    comment_id: str = Field(..., min_length=1)
    author_id: str = Field(..., min_length=1)

    text: str = Field(..., min_length=1, max_length=10000)

    parent_comment: Optional[str] = Field(
        default=None,
        max_length=10000,
    )

    parent_comment_id: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    community_id: str = Field(..., min_length=1)
    language: str = "en"


class ModerationResponse(BaseModel):
    comment_id: str
    community_id: str
    predictions: dict[str, float]
    spam_detection: dict
    decision: str
    triggered_rules: list[str]
    model_version: str
    policy_version: str


class ModerationReviewRequest(BaseModel):
    moderator_id: str = Field(..., min_length=1)

    moderator_decision: str = Field(
        ...,
        pattern="^(allow|remove|escalate)$",
    )

    reason: Optional[str] = Field(
        default=None,
        max_length=1000,
    )
