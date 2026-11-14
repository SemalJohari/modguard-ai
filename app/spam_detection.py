
import re
import hashlib


PROMOTIONAL_TERMS = [
    "buy now",
    "limited offer",
    "discount",
    "click here",
    "subscribe now",
]


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "<URL>", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_promotional_content(text: str) -> bool:
    normalized = normalize_text(text)

    return any(
        term in normalized
        for term in PROMOTIONAL_TERMS
    )


def generate_text_hash(text: str) -> str:
    normalized = normalize_text(text)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def detect_spam(text: str) -> dict:
    promotional = detect_promotional_content(text)

    return {
        "promotional_content": promotional,
        "normalized_text": normalize_text(text),
        "text_hash": generate_text_hash(text),
        "spam_signal": promotional,
        "reason": (
            "promotional_phrase_detected"
            if promotional
            else None
        ),
    }
