
from datetime import datetime, timedelta, timezone

from sqlalchemy import func

from app.database import ModerationRecord


MAX_POSTS = 3
TIME_WINDOW_MINUTES = 10


def detect_repeated_promotional_spam(
    db,
    community_id: str,
    author_id: str,
    text_hash: str,
) -> dict:

    now = datetime.now(timezone.utc)

    window_start = now - timedelta(
        minutes=TIME_WINDOW_MINUTES
    )

    matching_posts = (
        db.query(func.count(ModerationRecord.id))
        .filter(
            ModerationRecord.community_id == community_id,
            ModerationRecord.author_id == author_id,
            ModerationRecord.text_hash == text_hash,
            ModerationRecord.promotional_content.is_(True),
            ModerationRecord.created_at >= window_start,
        )
        .scalar()
    )

    # Include the current post, which has not yet been saved.
    total_posts = matching_posts + 1

    detected = total_posts >= MAX_POSTS

    return {
        "repeated_promotional_spam": detected,
        "matching_posts": total_posts,
        "threshold": MAX_POSTS,
        "window_minutes": TIME_WINDOW_MINUTES,
        "reason": (
            "repeated_promotional_posting"
            if detected
            else None
        ),
    }
