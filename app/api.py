
from fastapi import FastAPI, HTTPException

from app.schemas import (
    ModerationRequest,
    ModerationResponse,
    ModerationReviewRequest,
)
from app.moderation import ModerationModel
from app.policy import evaluate_policy
from app.database import (
    create_tables,
    SessionLocal,
    ModerationRecord,
    ModerationReview,
)
from app.spam_detection import detect_spam
from app.spam_history import detect_repeated_promotional_spam


app = FastAPI(
    title="ModGuard AI",
    description="Context-aware AI content moderation API"
)

moderator = ModerationModel()


@app.on_event("startup")
def initialize_database():
    create_tables()


@app.post(
    "/moderate",
    response_model=ModerationResponse
)
def moderate_comment(request: ModerationRequest):

    db = SessionLocal()

    try:
        # Prevent duplicate comment IDs from being stored.
        existing_record = (
            db.query(ModerationRecord)
            .filter(
                ModerationRecord.comment_id == request.comment_id
            )
            .first()
        )

        if existing_record:
            raise HTTPException(
                status_code=409,
                detail="This comment_id has already been processed."
            )

        # 1. Detect promotional content.
        spam_result = detect_spam(request.text)

        # 2. Check posting history only for promotional content.
        if spam_result["promotional_content"]:
            history_result = detect_repeated_promotional_spam(
                db=db,
                community_id=request.community_id,
                author_id=request.author_id,
                text_hash=spam_result["text_hash"]
            )
        else:
            history_result = {
                "repeated_promotional_spam": False,
                "matching_posts": 0,
                "threshold": 3,
                "window_minutes": 10,
                "reason": None
            }

        # 3. Run the contextual moderation model.
        if request.language.lower() == "en":

            predictions = moderator.predict(
                current_comment=request.text,
                parent_comment=request.parent_comment
            )

            policy_result = evaluate_policy(
                predictions=predictions,
                community_id=request.community_id
            )

            decision = policy_result["decision"]
            triggered_rules = policy_result["triggered_rules"]
            policy_version = policy_result["policy_version"]

        else:
            predictions = {}
            decision = "manual_review"
            triggered_rules = ["unsupported_language"]
            policy_version = "1.0"

        # 4. Escalate repeated promotional posting for review.
        if history_result["repeated_promotional_spam"]:
            decision = "manual_review"

            if "repeated_promotional_spam" not in triggered_rules:
                triggered_rules.append(
                    "repeated_promotional_spam"
                )

        # 5. Save the comment and its moderation result.
        record = ModerationRecord(
            comment_id=request.comment_id,
            community_id=request.community_id,
            author_id=request.author_id,

            text_hash=spam_result["text_hash"],
            promotional_content=spam_result["promotional_content"],
            text=request.text,

            decision=decision,

            # Audit information
            model_version="contextual_01",
            policy_version=policy_version,
            predictions=predictions,
            triggered_rules=triggered_rules,

            # Parent comment reference
            parent_comment_id=request.parent_comment_id,

            # Review status
            review_status=(
                "pending"
                if decision == "manual_review"
                else "not_required"
            ),
        )

        db.add(record)
        db.commit()

        # 6. Return the moderation response.
        return {
            "comment_id": request.comment_id,
            "community_id": request.community_id,
            "predictions": predictions,
            "spam_detection": {
                "promotional_content": spam_result["promotional_content"],
                "spam_signal": spam_result["spam_signal"],
                "repeated_promotional_spam": history_result[
                    "repeated_promotional_spam"
                ],
                "matching_posts": history_result["matching_posts"],
                "threshold": history_result["threshold"],
                "window_minutes": history_result["window_minutes"],
                "reason": (
                    history_result["reason"]
                    or spam_result["reason"]
                )
            },
            "decision": decision,
            "triggered_rules": triggered_rules,
            "model_version": "contextual_01",
            "policy_version": policy_version
        }

    finally:
        db.close()


@app.post("/moderation/{comment_id}/review")
def review_moderation(
    comment_id: str,
    request: ModerationReviewRequest,
):
    db = SessionLocal()

    try:
        record = (
            db.query(ModerationRecord)
            .filter(
                ModerationRecord.comment_id == comment_id
            )
            .first()
        )

        if record is None:
            raise HTTPException(
                status_code=404,
                detail="Moderation record not found."
            )

        # Preserve the original model decision.
        original_decision = record.decision

        # Store the latest moderator decision.
        record.moderator_decision = request.moderator_decision
        record.review_status = "completed"

        # Append a separate review-history record.
        review = ModerationReview(
            comment_id=comment_id,
            moderator_id=request.moderator_id,
            moderator_decision=request.moderator_decision,
            reason=request.reason,
        )

        db.add(review)
        db.commit()

        return {
            "comment_id": comment_id,
            "original_decision": original_decision,
            "moderator_decision": request.moderator_decision,
            "review_status": record.review_status,
            "message": "Moderator decision recorded."
        }

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()
