
import os
from datetime import datetime, timezone

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Boolean,
    DateTime,
    Integer,
    Index,
    JSON,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, sessionmaker


os.makedirs("data", exist_ok=True)

DATABASE_URL = "sqlite:///./data/modguard.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


class ModerationRecord(Base):
    __tablename__ = "moderation_records"

    id = Column(Integer, primary_key=True, index=True)

    comment_id = Column(String, unique=True, nullable=False, index=True)
    community_id = Column(String, nullable=False, index=True)
    author_id = Column(String, nullable=False, index=True)

    text_hash = Column(String, nullable=False, index=True)
    promotional_content = Column(Boolean, default=False)

    # Retained for this prototype.
    # Review privacy and retention before production deployment.
    text = Column(String, nullable=False)

    decision = Column(String, nullable=False)

    # Audit information
    model_version = Column(String, nullable=True)
    policy_version = Column(String, nullable=True)
    predictions = Column(JSON, nullable=True)
    triggered_rules = Column(JSON, nullable=True)

    # Optional reference to the parent comment
    parent_comment_id = Column(String, nullable=True)

    # Review tracking
    review_status = Column(
        String,
        nullable=False,
        default="not_required",
    )

    moderator_decision = Column(String, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class ModerationReview(Base):
    """
    Append-only record of moderator decisions.
    Keeps a history if a comment is reviewed more than once.
    """

    __tablename__ = "moderation_reviews"

    id = Column(Integer, primary_key=True, index=True)

    comment_id = Column(
        String,
        ForeignKey("moderation_records.comment_id"),
        nullable=False,
        index=True,
    )

    moderator_id = Column(String, nullable=False)
    moderator_decision = Column(String, nullable=False)
    reason = Column(String, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


Index(
    "idx_spam_history",
    ModerationRecord.community_id,
    ModerationRecord.author_id,
    ModerationRecord.text_hash,
    ModerationRecord.created_at,
)


def create_tables():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()

