# ui/streamlit_app.py

import streamlit as st
from datetime import datetime
from html import escape

from src.safety.guardrails import validate_query
from src.orchestrator import run_query
from src.ui.components.chat_window import render_chat_message
from src.ui.components.charts import render_chart
from src.ui.components.trace_viewer import render_trace
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Fraud Intelligence Agent",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    "<h1 style='margin-bottom: 0.2rem;'>🧠 Mekari Fraud Intelligence Agent</h1>",
    unsafe_allow_html=True,
)
st.caption("Fraud analytics · Fraud Doc RAG · Multilingual assistant")


# ======================================================
# RECURSIVE SANITIZATION
# ======================================================
def sanitize_recursive(obj):
    """Escape all strings deeply in dicts / lists."""
    if isinstance(obj, str):
        return escape(obj)

    if isinstance(obj, dict):
        return {k: sanitize_recursive(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [sanitize_recursive(i) for i in obj]

    return obj


def sanitize_message_history():
    """Sanitize all existing messages in session state to remove legacy HTML."""
    if "messages" not in st.session_state:
        return
    for m in st.session_state.messages:
        m["content"] = sanitize_recursive(m.get("content"))


# ======================================================
# SESSION STATE
# ======================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

# Clean old legacy messages
sanitize_message_history()


# ======================================================
# ORCHESTRATOR WRAPPER
# ======================================================
def _handle_query(query: str):

    # ----- Guardrails -----
    is_valid, cleaned, lang = validate_query(query)

    if not is_valid:
        return {
            "type": "error",
            "error": escape(cleaned),
            "details": None,
        }

    # ----- Orchestrator -----
    out = run_query(cleaned, detected_lang=lang)

    if out.get("error"):
        return {
            "type": "error",
            "error": "Orchestrator failed.",
            "details": escape(str(out.get("error"))),
        }

    # ----- Build final payload -----
    result = out.get("result", {})
    result["_query"] = escape(str(out.get("query")))
    result["_intent"] = escape(str(out.get("intent")))
    result["_lang"] = lang

    # CRITICAL — sanitize before returning
    return sanitize_recursive(result)


# ======================================================
# LAYOUT
# ======================================================
col_chat, col_side = st.columns([0.68, 0.32])


# ======================================================
# LEFT PANEL — CHAT
# ======================================================
with col_chat:

    st.markdown("### 💬 Conversation")

    chat_container = st.container()

    with chat_container:
        msgs = st.session_state.messages

        if not msgs:
            st.info("Ask a question to begin.")
        else:
            # ---- HISTORY ----
            if len(msgs) > 2:
                with st.expander("📜 History (previous turns)"):
                    for m in msgs[:-2]:
                        render_chat_message(m)

            # ---- LATEST TURN ----
            if len(msgs) >= 2:
                render_chat_message(msgs[-2])
                render_chat_message(msgs[-1])
            else:
                render_chat_message(msgs[0])

        st.markdown(
            "<script>window.scrollTo(0, document.body.scrollHeight);</script>",
            unsafe_allow_html=True,
        )

    # ---- CLEAR BUTTON ----
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_input = ""
        st.session_state.clear_input = False
        st.rerun()

    # ---- USER INPUT ----
    st.markdown("---")
    user_input = st.text_input(
        "Your question:",
        placeholder="Ask about fraud rates, merchant risk, cross-border patterns, or regulations…",
        key="chat_input",
        label_visibility="collapsed",
    )

    send = st.button("Ask", use_container_width=True)


# ======================================================
# SUBMIT HANDLER
# ======================================================
if send and user_input.strip():

    safe_query = escape(user_input.strip())

    # Save user message (HTML safe)
    st.session_state.messages.append(
        {"role": "user", "content": safe_query, "time": datetime.now()}
    )

    # Request
    with st.spinner("Analyzing..."):
        result = _handle_query(safe_query)

    # Save assistant message (ALREADY SANITIZED)
    st.session_state.messages.append(
        {"role": "assistant", "content": result, "time": datetime.now()}
    )

    st.session_state.clear_input = True
    st.rerun()


# ======================================================
# RIGHT PANEL — DETAILS
# ======================================================
with col_side:

    st.markdown("### 📊 Analytics Result")

    # Find latest assistant message
    last_assistant = next(
        (m for m in reversed(st.session_state.messages) if m["role"] == "assistant"),
        None,
    )

    if not last_assistant:
        st.info("Ask a question to see results here.")
        st.stop()

    payload = last_assistant.get("content", {})

    if not isinstance(payload, dict):
        st.info("No structured data.")
        st.stop()

    ptype = payload.get("type")

    # ----- ERROR -----
    if ptype == "error":
        st.error(payload.get("error", "Error"))
        if payload.get("details"):
            st.code(payload["details"])
        st.stop()

    # ----- ANALYTICS -----
    if ptype == "analytics":
        analytics = payload.get("analytics", {})

        st.markdown("#### 📈 Summary")
        summary = escape(analytics.get("answer") or payload.get("answer") or "")
        st.markdown(f"<div style='white-space:pre-wrap;'>{summary}</div>", unsafe_allow_html=True)

        conf = analytics.get("confidence")
        if isinstance(conf, (int, float)):
            st.caption(f"Confidence: **{conf:.2f}**")

        chart_data = analytics.get("chart_data")
        if chart_data:
            st.markdown("#### 📊 Chart")
            render_chart(chart_data)
        else:
            st.info("No chart data.")

        st.stop()

    # ----- RAG -----
    if ptype == "rag":
        st.markdown("#### 💡 Insight")

        insight_text = escape(payload.get("insight") or payload.get("answer") or "")
        st.markdown(f"<div style='white-space:pre-wrap;'>{insight_text}</div>", unsafe_allow_html=True)

        score = payload.get("score", {}).get("final_score")
        if isinstance(score, (int, float)):
            st.caption(f"Answer confidence: **{score:.2f}**")

        st.markdown("#### 📘 Citations")
        cites = payload.get("citations") or []
        if cites:
            render_trace(cites)
        else:
            st.info("No citations.")

        st.stop()

    # ----- FALLBACK -----
    st.info("No structured information available.")
