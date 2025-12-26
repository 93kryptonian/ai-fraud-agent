"""
CHAT WINDOW COMPONENT
---------------------

Responsibilities:
- Render user & assistant messages
- Enforce HTML/XSS safety
- Keep UI logic isolated from orchestration logic
- Provide consistent visual structure for RAG / Analytics / Errors

Design principles:
- Never trust message content
- Escape everything before rendering
- UI-only (no business logic)
"""

import streamlit as st
from datetime import datetime
from html import escape
from typing import Dict, Any, Optional

# =============================================================================
# CSS STYLING (INJECTED ONCE)
# =============================================================================

if "chat_css_injected" not in st.session_state:
    st.markdown(
        """
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
        """,
        unsafe_allow_html=True,
    )
    st.session_state.chat_css_injected = True

# =============================================================================
# UTILITIES
# =============================================================================

def _format_time(ts: Optional[datetime]) -> str:
    """
    Format timestamp for chat metadata.
    """
    return ts.strftime("%H:%M") if isinstance(ts, datetime) else ""


def _safe(text: Any) -> str:
    """
    Convert to string and escape HTML.
    """
    return escape(str(text)) if text is not None else ""

# =============================================================================
# PUBLIC RENDER FUNCTION
# =============================================================================

def render_chat_message(message: Dict[str, Any]):
    """
    Render a single chat message.

    Expected message format:
        {
            "role": "user" | "assistant",
            "content": str | dict,
            "time": datetime
        }
    """
    role = message.get("role", "assistant")
    content = message.get("content")
    timestamp = _format_time(message.get("time"))

    if role == "user":
        _render_user(content, timestamp)
        return

    # ---------------- Assistant ----------------
    if isinstance(content, dict):
        msg_type = content.get("type")

        if msg_type == "rag":
            text = content.get("answer") or ""
            insight = content.get("insight")
            _render_assistant(
                text=text,
                timestamp=timestamp,
                footer="RAG Answer",
                insight=insight,
            )
            return

        if msg_type == "analytics":
            text = (
                content.get("analytics", {}).get("answer")
                or content.get("answer")
                or ""
            )
            _render_assistant(
                text=text,
                timestamp=timestamp,
                footer="Analytics",
            )
            return

        if msg_type == "reject":
            text = content.get("message", "Request rejected.")
            _render_assistant(
                text=text,
                timestamp=timestamp,
                footer="Safety",
            )
            return

        if msg_type == "error":
            text = f"❗ {content.get('error', 'Error')}"
            _render_assistant(
                text=text,
                timestamp=timestamp,
                footer="System",
            )
            return

        # Fallback (unexpected dict)
        _render_assistant(
            text=str(content),
            timestamp=timestamp,
        )
        return

    # Plain string fallback
    _render_assistant(
        text=str(content),
        timestamp=timestamp,
    )

# =============================================================================
# INTERNAL RENDERERS (SAFE)
# =============================================================================

def _render_user(text: Any, timestamp: str):
    """
    Render user chat bubble.
    """
    safe_text = _safe(text)

    st.markdown(
        f"""
        <div class="chat-bubble-user">
        🕵️‍♂️ You:  
        {safe_text}

        <div class="chat-meta">{timestamp}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_assistant(
    text: Any,
    timestamp: str,
    footer: str = "Assistant",
    insight: Optional[Any] = None,
):
    """
    Render assistant chat bubble.
    """
    safe_text = _safe(text)
    safe_insight = _safe(insight) if insight else None

    insight_block = (
        f"\n\n**Insight:**\n{safe_insight}" if safe_insight else ""
    )

    st.markdown(
        f"""
        <div class="chat-bubble-assistant">
        🤖 Agent:  
        {safe_text}
        {insight_block}

        <div class="chat-meta">🧠 {footer} · {timestamp}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
