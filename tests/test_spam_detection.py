
from app.spam_detection import (
    normalize_text,
    detect_promotional_content,
    generate_text_hash,
    detect_spam,
)


def test_normalization():
    text = "BUY NOW   https://example.com"
    normalized = normalize_text(text)

    assert normalized == "buy now <URL>"


def test_promotional_detection():
    assert detect_promotional_content(
        "Buy now for a limited offer!"
    )


def test_non_promotional_comment():
    assert not detect_promotional_content(
        "I enjoyed reading this discussion."
    )


def test_hash_consistency():
    first = generate_text_hash("BUY NOW")
    second = generate_text_hash("buy now")

    assert first == second


def test_spam_result():
    result = detect_spam("Click here to get a discount")

    assert result["promotional_content"] is True
    assert result["spam_signal"] is True
    assert result["reason"] == "promotional_phrase_detected"
