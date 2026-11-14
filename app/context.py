
def build_contextual_text(
    current_comment: str,
    parent_comment: str | None = None
) -> str:

    current_comment = current_comment.strip()

    if parent_comment:
        parent_comment = parent_comment.strip()

        contextual_text = (
            "[CURRENT COMMENT] "
            + current_comment
            + " [PARENT COMMENT] "
            + parent_comment
        )

    else:
        contextual_text = (
            "[CURRENT COMMENT] "
            + current_comment
        )

    return contextual_text

