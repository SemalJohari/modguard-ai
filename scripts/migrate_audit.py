
from sqlalchemy import inspect, text

from app.database import engine, create_tables


def migrate_audit_columns():
    # Ensure the existing tables are present.
    create_tables()

    inspector = inspect(engine)
    existing_columns = {
        column["name"]
        for column in inspector.get_columns("moderation_records")
    }

    new_columns = {
        "model_version": "VARCHAR(100)",
        "policy_version": "VARCHAR(100)",
        "predictions": "JSON",
        "triggered_rules": "JSON",
        "parent_comment_id": "VARCHAR(255)",
        "review_status": "VARCHAR(50) DEFAULT 'not_required'",
        "moderator_decision": "VARCHAR(50)",
    }

    with engine.begin() as connection:
        for column_name, column_type in new_columns.items():

            if column_name not in existing_columns:
                connection.execute(
                    text(
                        f"ALTER TABLE moderation_records "
                        f"ADD COLUMN {column_name} {column_type}"
                    )
                )

                print(f"Added column: {column_name}")
            else:
                print(f"Already exists: {column_name}")

    print("Audit database migration completed.")


if __name__ == "__main__":
    migrate_audit_columns()
