import streamlit as st
from datetime import datetime
from html import escape


# ======================================================
# CSS Styling (safe)
# ======================================================
if "chat_css_injected" not in st.session_state:
    st.markdown("""
    <style>

    .chat-bubble-user {
        background-color: #bbf7d0;
        border: 1px solid #86efac;
        padding: 0.6rem 0.9rem;
        border-radius: 12px;
        margin: 0.4rem 0;
        width: 100%;
        white-space: pre-wrap;
        word-break: break-word;
    }

    .chat-bubble-assistant {
        background-color: #f3f4f6;
        border: 1px solid #d1d5db;
        padding: 0.6rem 0.9rem;
        border-radius: 12px;
        margin: 0.4rem 0;
        width: 100%;
        white-space: pre-wrap;
        word-break: break-word;
    }

    .chat-meta {
        font-size: 0.7rem;
        color: #6b7280;
        text-align: right;
    }

    </style>
    """, unsafe_allow_html=True)

    st.session_state.chat_css_injected = True


def _fmt_time(dt: datetime):
    return dt.strftime("%H:%M") if isinstance(dt, datetime) else ""


def render_chat_message(msg: dict):
    role = msg.get("role", "assistant")
    content = msg.get("content")
    timestamp = _fmt_time(msg.get("time"))

    if role == "user":
        _render_user(content, timestamp)
        return

    # assistant
    if isinstance(content, dict):
        ptype = content.get("type")

        if ptype == "rag":
            text = content.get("answer") or ""
            insight = content.get("insight")
            _render_assistant(text, timestamp, "RAG Answer", insight)
            return

        if ptype == "analytics":
            text = content.get("analytics", {}).get("answer") or content.get("answer") or ""
            _render_assistant(text, timestamp, "Analytics")
            return
        if ptype == "reject":
            text = content.get("message", "Request rejected.")
            _render_assistant(text, timestamp, "Safety")
            return

        if ptype == "error":
            text = f"❗ {content.get('error', 'Error')}"
            _render_assistant(text, timestamp, "System")
            return

        _render_assistant(str(content), timestamp)
        return

    _render_assistant(str(content), timestamp)


# ======================================================
# NEW SAFE BUBBLE RENDERING (Markdown-only, NO HTML embedding)
# ======================================================
def _render_user(text, timestamp):
    safe = escape(str(text))
    st.markdown(f"""
<div class="chat-bubble-user">
🕵️‍♂️ You:  
{safe}

<div class="chat-meta">{timestamp}</div>
</div>
""", unsafe_allow_html=True)


def _render_assistant(text, timestamp, footer="Assistant", insight=None):
    safe_text = escape(str(text))
    safe_insight = escape(str(insight)) if insight else None

    insight_block = f"\n\n**Insight:**\n{safe_insight}" if safe_insight else ""

    # Entire message is markdown-safe, not HTML-embedded
    st.markdown(f"""
<div class="chat-bubble-assistant">
🤖 Agent:  
{safe_text}
{insight_block}

<div class="chat-meta">🧠 {footer} · {timestamp}</div>
</div>
""", unsafe_allow_html=True)
