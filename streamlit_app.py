from __future__ import annotations

import uuid
from typing import Any

import requests
import streamlit as st

st.set_page_config(
    page_title="ModGuard AI | Content Moderation",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1250px;}
.hero {padding: 1.35rem 1.5rem; border-radius: 16px;
background: linear-gradient(120deg, #172554 0%, #1e3a8a 55%, #0f766e 100%);
color: white; margin-bottom: 1.2rem;}
.hero h1 {margin: 0; font-size: 2rem; color: white;}
.hero p {margin: .45rem 0 0 0; color: #dbeafe;}
.small-muted {color: #64748b; font-size: .88rem;}
div[data-testid="stMetric"] {background: rgba(120,120,120,.07);
border: 1px solid rgba(120,120,120,.16); padding: 12px 14px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

for key, default in {
    "last_result": None,
    "review_result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def api_url(path: str) -> str:
    return st.session_state.api_base.rstrip("/") + path


def show_json(value: Any) -> None:
    st.json(value, expanded=True)


def show_decision(decision: str) -> None:
    if decision == "allow":
        st.success("ALLOW — comment can remain visible.")
    elif decision == "remove":
        st.error("REMOVE — comment should be removed.")
    elif decision == "manual_review":
        st.warning("MANUAL REVIEW — moderator attention is required.")
    else:
        st.info(f"Decision returned by API: {decision or 'not provided'}")


with st.sidebar:
    st.title("🛡️ ModGuard AI")
    st.caption("Context-aware content moderation prototype")
    st.divider()
    st.subheader("API connection")
    st.text_input(
        "FastAPI base URL",
        value="http://127.0.0.1:8000",
        key="api_base",
        help="Keep FastAPI running in a separate terminal.",
    )
    if st.button("Check API connection", use_container_width=True):
        try:
            r = requests.get(api_url("/openapi.json"), timeout=5)
            if r.ok:
                st.success("API is reachable.")
            else:
                st.error(f"API returned HTTP {r.status_code}.")
        except requests.RequestException as exc:
            st.error(f"Could not reach API: {exc}")
    st.divider()
    st.markdown("**Moderation pipeline**")
    st.markdown(
        "1. Comment and context input  \n"
        "2. Promotional-spam detection  \n"
        "3. Contextual model inference  \n"
        "4. Community policy decision  \n"
        "5. Audit logging and moderator review"
    )
    st.caption("Demo prototype • Model: contextual_01")

st.markdown("""
<div class="hero">
<h1>🛡️ ModGuard AI</h1>
<p>Context-aware moderation for safer online communities</p>
</div>
""", unsafe_allow_html=True)

st.write(
    "Submit a comment with its conversation context. The API evaluates the "
    "comment, checks promotional spam signals, applies community policy, and "
    "records the result."
)
st.caption("Backend: FastAPI  |  Frontend: Streamlit  |  Storage: SQLite")
st.divider()

st.subheader("1. Submit a comment")
st.caption("Comment ID, Author ID, Comment text, and Community ID are required.")

with st.form("moderation_form"):
    col1, col2 = st.columns(2)
    with col1:
        comment_id = st.text_input(
            "Comment ID",
            value=f"demo-{uuid.uuid4().hex[:10]}",
            help="The API rejects comment IDs that have already been processed.",
        )
        author_id = st.text_input("Author ID", value="demo_user_01")
        community_id = st.text_input("Community ID", value="general")
    with col2:
        language = st.selectbox("Language", ["en", "hi", "other"])
        parent_comment_id = st.text_input("Parent comment ID (optional)")
        st.caption("The contextual model currently runs only for English.")
    comment_text = st.text_area(
        "Comment text", height=130, max_chars=10000,
        placeholder="Enter the comment you want ModGuard AI to evaluate...",
    )
    parent_comment = st.text_area(
        "Parent comment / conversation context (optional)",
        height=100, max_chars=10000,
        placeholder="Paste the parent comment to provide conversational context...",
    )
    submitted = st.form_submit_button(
        "Run moderation", type="primary", use_container_width=True
    )

if submitted:
    if not all([comment_id.strip(), author_id.strip(), community_id.strip(), comment_text.strip()]):
        st.error("Please complete all required fields.")
    else:
        payload = {
            "comment_id": comment_id.strip(),
            "author_id": author_id.strip(),
            "text": comment_text.strip(),
            "parent_comment": parent_comment.strip() or None,
            "parent_comment_id": parent_comment_id.strip() or None,
            "community_id": community_id.strip(),
            "language": language,
        }
        try:
            with st.spinner("Running spam detection, model inference, and policy evaluation..."):
                response = requests.post(
                    api_url("/moderate"), json=payload, timeout=120
                )
            if response.ok:
                st.session_state.last_result = response.json()
                st.session_state.review_result = None
                st.success("Moderation completed. The result has been returned by the API.")
            else:
                st.error(f"API error — HTTP {response.status_code}")
                try:
                    show_json(response.json())
                except ValueError:
                    st.code(response.text)
        except requests.Timeout:
            st.error("The API request timed out. Check whether the model is loading.")
        except requests.RequestException as exc:
            st.error(f"Could not connect to the API: {exc}")

result = st.session_state.last_result
if result:
    st.divider()
    st.subheader("2. Moderation result")
    decision = result.get("decision", "unknown")
    show_decision(decision)

    a, b, c = st.columns(3)
    a.metric("Decision", decision.replace("_", " ").upper())
    b.metric("Model version", result.get("model_version", "—"))
    c.metric("Policy version", result.get("policy_version", "—"))
    st.caption(
        f"Comment ID: `{result.get('comment_id', '—')}`  |  "
        f"Community: `{result.get('community_id', '—')}`"
    )

    predictions = result.get("predictions") or {}
    spam = result.get("spam_detection") or {}
    left, right = st.columns(2)

    with left:
        st.markdown("**Contextual model predictions**")
        if predictions:
            for label, score in sorted(
                predictions.items(), key=lambda item: float(item[1]), reverse=True
            ):
                try:
                    value = max(0.0, min(1.0, float(score)))
                    st.write(f"{label.replace('_', ' ').title()} — {value:.3f}")
                    st.progress(value)
                except (TypeError, ValueError):
                    st.write(f"{label}: {score}")
        else:
            st.info("No model predictions returned. Unsupported languages are routed to manual review.")

    with right:
        st.markdown("**Promotional spam analysis**")
        st.metric(
            "Promotional content",
            "Detected" if spam.get("promotional_content") else "Not detected",
        )
        st.metric(
            "Repeated promotional spam",
            "Yes" if spam.get("repeated_promotional_spam") else "No",
        )
        x, y = st.columns(2)
        x.metric("Matching posts", spam.get("matching_posts", 0))
        y.metric("Threshold", spam.get("threshold", 3))
        if spam.get("spam_signal"):
            st.write("Spam signal:", spam["spam_signal"])
        if spam.get("reason"):
            st.caption(f"Reason: {spam['reason']}")

    st.markdown("**Triggered policy rules**")
    rules = result.get("triggered_rules") or []
    if rules:
        for rule in rules:
            st.markdown(f"- `{rule}`")
    else:
        st.write("No policy rules were triggered.")

    with st.expander("View complete API response"):
        show_json(result)

    if decision == "manual_review":
        st.divider()
        st.subheader("3. Moderator review")
        st.warning(
            "This prototype records moderator decisions but does not implement "
            "moderator authentication or authorization."
        )
        with st.form("review_form"):
            moderator_id = st.text_input("Moderator ID", value="demo_moderator")
            moderator_decision = st.selectbox(
                "Final moderator decision",
                ["allow", "remove", "escalate"],
                format_func=lambda value: {
                    "allow": "Allow comment",
                    "remove": "Remove comment",
                    "escalate": "Escalate for further review",
                }[value],
            )
            reason = st.text_area("Review reason (optional)", max_chars=1000)
            review_submitted = st.form_submit_button(
                "Submit moderator decision", use_container_width=True
            )

        if review_submitted:
            if not moderator_id.strip():
                st.error("Moderator ID is required.")
            else:
                review_payload = {
                    "moderator_id": moderator_id.strip(),
                    "moderator_decision": moderator_decision,
                    "reason": reason.strip() or None,
                }
                try:
                    rr = requests.post(
                        api_url(f"/moderation/{result.get('comment_id')}/review"),
                        json=review_payload,
                        timeout=30,
                    )
                    if rr.ok:
                        st.session_state.review_result = rr.json()
                        st.success("Moderator decision recorded.")
                    else:
                        st.error(f"Review API error — HTTP {rr.status_code}")
                        try:
                            show_json(rr.json())
                        except ValueError:
                            st.code(rr.text)
                except requests.RequestException as exc:
                    st.error(f"Could not submit moderator review: {exc}")

        if st.session_state.review_result:
            st.markdown("**Review confirmation**")
            show_json(st.session_state.review_result)

st.divider()
st.caption(
    "Academic demonstration prototype. Model predictions are not a substitute "
    "for human moderation. Synthetic training data and the absence of reviewer "
    "authentication are important limitations."
)
