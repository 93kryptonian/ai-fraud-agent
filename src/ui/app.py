"""
STREAMLIT UI — AI FRAUD INTELLIGENCE AGENT
-----------------------------------------

Responsibilities:
- User interaction & visualization
- Guardrails enforcement before orchestration
- Safe rendering (XSS-protected)
- Conversation state management
- Analytics & RAG result presentation

Design principles:
- Never trust user input
- Never render raw HTML from LLM
- Stateless backend, stateful UI
"""

import streamlit as st
from datetime import datetime
from html import escape
from typing import Any

from src.safety.guardrails import validate_query
from src.orchestrator import run_query
from src.ui.components.chat_window import render_chat_message
from src.ui.components.charts import render_chart
from src.ui.components.trace_viewer import render_trace
from src.utils.logger import get_logger

logger = get_logger(__name__)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Fraud Intelligence Agent",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    "<h1 style='margin-bottom:0.2rem;'>🧠 Fraud Intelligence AI Platform</h1>",
    unsafe_allow_html=True,
)
st.caption("Fraud analytics · Fraud Doc RAG · Multilingual assistant")

# =============================================================================
# SANITIZATION UTILITIES (XSS PROTECTION)
# =============================================================================

def sanitize_recursive(obj: Any):
    """
    Recursively escape all strings inside nested structures.
    Prevents XSS when rendering LLM or user-generated content.
    """
    if isinstance(obj, str):
        return escape(obj)

    if isinstance(obj, dict):
        return {k: sanitize_recursive(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [sanitize_recursive(i) for i in obj]

    return obj


def sanitize_message_history():
    """
    Sanitize existing session messages (backward compatibility
    for legacy unsanitized content).
    """
    if "messages" not in st.session_state:
        return

    for msg in st.session_state.messages:
        msg["content"] = sanitize_recursive(msg.get("content"))

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

sanitize_message_history()

# =============================================================================
# ORCHESTRATOR WRAPPER
# =============================================================================

def handle_query(query: str) -> dict:
    """
    End-to-end query handler:
    - Guardrails
    - Orchestration
    - Sanitization
    """

    # ---------------- Guardrails ----------------
    is_valid, cleaned, lang = validate_query(query)

    if not is_valid:
        return {
            "type": "error",
            "error": escape(cleaned),
            "details": None,
        }

    # ---------------- Orchestrator ----------------
    result = run_query(cleaned, detected_lang=lang)

    if result.get("error"):
        return {
            "type": "error",
            "error": "Orchestrator failed.",
            "details": escape(str(result.get("error"))),
        }

    payload = result.get("result", {})
    payload["_query"] = escape(str(result.get("query")))
    payload["_intent"] = escape(str(result.get("intent")))
    payload["_lang"] = lang

    # CRITICAL: sanitize before returning to UI
    return sanitize_recursive(payload)

# =============================================================================
# LAYOUT
# =============================================================================

col_chat, col_side = st.columns([0.68, 0.32])

# =============================================================================
# LEFT PANEL — CHAT INTERFACE
# =============================================================================

with col_chat:
    st.markdown("### 💬 Conversation")
    chat_container = st.container()

    with chat_container:
        messages = st.session_state.messages

        if not messages:
            st.info("Ask a question to begin.")
        else:
            # -------- Conversation History --------
            if len(messages) > 2:
                with st.expander("📜 History (previous turns)"):
                    for m in messages[:-2]:
                        render_chat_message(m)

            # -------- Latest Turn --------
            if len(messages) >= 2:
                render_chat_message(messages[-2])
                render_chat_message(messages[-1])
            else:
                render_chat_message(messages[0])

        # Auto-scroll
        st.markdown(
            "<script>window.scrollTo(0, document.body.scrollHeight);</script>",
            unsafe_allow_html=True,
        )

    # -------- Clear Conversation --------
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_input = ""
        st.session_state.clear_input = False
        st.rerun()

    # -------- User Input --------
    st.markdown("---")
    user_input = st.text_input(
        "Your question:",
        placeholder=(
            "Ask about fraud rates, merchant risk, "
            "cross-border patterns, or regulations…"
        ),
        key="chat_input",
        label_visibility="collapsed",
    )

    send = st.button("Ask", use_container_width=True)

# =============================================================================
# SUBMIT HANDLER
# =============================================================================

if send and user_input.strip():
    safe_query = escape(user_input.strip())

    # Save user message
    st.session_state.messages.append(
        {"role": "user", "content": safe_query, "time": datetime.now()}
    )

    with st.spinner("Analyzing..."):
        response = handle_query(safe_query)

    # Save assistant message (already sanitized)
    st.session_state.messages.append(
        {"role": "assistant", "content": response, "time": datetime.now()}
    )

    st.session_state.clear_input = True
    st.rerun()

# =============================================================================
# RIGHT PANEL — RESULT DETAILS
# =============================================================================

with col_side:
    st.markdown("### 📊 Result")

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

    # ---------------- ERROR ----------------
    if ptype == "error":
        st.error(payload.get("error", "Error"))
        if payload.get("details"):
            st.code(payload["details"])
        st.stop()

    # ---------------- ANALYTICS ----------------
    if ptype == "analytics":
        analytics = payload.get("analytics", {})

        st.markdown("#### 📈 Summary")
        summary = escape(
            analytics.get("answer")
            or payload.get("answer")
            or ""
        )
        st.markdown(
            f"<div style='white-space:pre-wrap;'>{summary}</div>",
            unsafe_allow_html=True,
        )

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

    # ---------------- RAG ----------------
    if ptype == "rag":
        st.markdown("#### 💡 Insight")

        insight = escape(
            payload.get("insight")
            or payload.get("answer")
            or ""
        )
        st.markdown(
            f"<div style='white-space:pre-wrap;'>{insight}</div>",
            unsafe_allow_html=True,
        )

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

    # ---------------- FALLBACK ----------------
    st.info("No structured information available.")
