"""
Live demo UI for the fraud-intelligence API.

A thin client: it needs one setting, API_URL, and holds no provider or
database credentials. Everything it knows about the backend comes back from
POST /query (see api_client.py).

Run locally:  API_URL=http://127.0.0.1:8000 streamlit run live_ui/app.py
"""

import os
import re

import streamlit as st

import api_client as client

DEFAULT_API_URL = "https://ai-fraud-agent.onrender.com"
API_URL = os.environ.get("API_URL", DEFAULT_API_URL).strip()
ARTICLE_URL = "https://medium.com/@93Kryptonian/one-slow-request-shouldnt-freeze-everything-7267e2345f48"
REPO_URL = "https://github.com/93kryptonian/ai-fraud-agent"
MAX_QUESTION_CHARS = 500

EXAMPLES = [
    "Which merchants have the highest fraud incidence?",
    "What is the monthly fraud trend?",
    "What are the main fraud typologies described in the documents?",
    "Which merchants have the highest fraud incidence, and why does card-not-present fraud occur?",
]

st.set_page_config(page_title="Fraud intelligence live demo", layout="centered")

# Static styling only. Nothing from the API or the user is ever placed in here.
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Literata:opsz,wght@7..72,400;7..72,600&family=Public+Sans:wght@400;600&display=swap');
html, body, [class*="st-"], button, input, textarea { font-family: 'Public Sans', system-ui, sans-serif; }
.block-container { max-width: 720px; padding-top: 3rem; }
h1.demo-title { font-family: 'Literata', Georgia, serif; font-weight: 600; font-size: 2.4rem;
                line-height: 1.15; letter-spacing: -0.01em; margin: 0 0 .6rem; color: #15202B; }
p.demo-lede { color: #3b4856; font-size: 1.05rem; line-height: 1.55; margin: 0 0 1.2rem; }
.answer p, .answer li { font-family: 'Literata', Georgia, serif; font-size: 1.12rem; line-height: 1.7; }
.evidence-note { color: #3b4856; font-size: .9rem; margin: .2rem 0 .8rem; }
#MainMenu, footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<h1 class="demo-title">Ask the fraud data</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="demo-lede">Ask a question about the fraud transactions or the fraud documents. '
    "Every answer is shown with the evidence it came from.</p>",
    unsafe_allow_html=True,
)
st.info(
    "Live demo on a free tier, not production. After a quiet period the first question "
    "can take up to a minute while the API wakes up."
)


def _plain(text: str) -> str:
    """Model text is rendered as markdown but never as HTML; keep $ from becoming math."""
    return text.replace("$", r"\$")


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _sql_evidence(outcome: client.Outcome, timing: bool = True) -> None:
    with st.container(border=True):
        if outcome.sql:
            st.markdown("**The query that produced the numbers**")
            st.code(outcome.sql, language="sql")
            note = f"{len(outcome.rows)} rows returned"
            if timing:
                note += f" · answered in {outcome.elapsed_seconds:.1f} s, measured from this page"
            st.markdown(f'<p class="evidence-note">{note}</p>', unsafe_allow_html=True)
            if outcome.rows:
                st.dataframe(outcome.rows, use_container_width=True, hide_index=True)
        else:
            st.markdown("**No SQL was run for this answer**, so there is no query to check it against.")


def _passages_evidence(outcome: client.Outcome) -> None:
    with st.container(border=True):
        if outcome.citations:
            st.markdown("**Passages the explanation was built from**")
            for c in outcome.citations:
                if not isinstance(c, dict):
                    continue
                st.markdown(f"{_plain(str(c.get('source', 'Unknown source')))}, page {c.get('page', '?')}")
                st.caption(_squash(str(c.get("preview", "")))[:280])
        else:
            st.markdown("**No passages were cited** for this answer.")


def render_outcome(outcome: client.Outcome) -> None:
    kind = outcome.kind

    if kind == client.ANALYTICS:
        if outcome.answer:
            st.markdown(f'<div class="answer">\n\n{_plain(outcome.answer)}\n\n</div>', unsafe_allow_html=True)
        _sql_evidence(outcome)
        if outcome.confidence is not None:
            st.caption(f"Confidence reported by the API: {outcome.confidence:.2f}")
        return

    if kind == client.RAG:
        if outcome.answer:
            st.markdown(f'<div class="answer">\n\n{_plain(outcome.answer)}\n\n</div>', unsafe_allow_html=True)
        if outcome.message:
            st.warning(outcome.message)
        _passages_evidence(outcome)
        st.caption(f"Answered in {outcome.elapsed_seconds:.1f} s, measured from this page")
        if outcome.confidence is not None:
            st.caption(f"Confidence reported by the API: {outcome.confidence:.2f}")
        return

    if kind == client.HYBRID:
        if outcome.answer:
            st.markdown(f'<div class="answer">\n\n{_plain(outcome.answer)}\n\n</div>', unsafe_allow_html=True)
        if outcome.parts.get("analytics") in {"failed", "no_data"}:
            st.warning("The transaction data could not answer the numbers part of this question.")
        if outcome.parts.get("rag") in {"failed", "degraded"}:
            st.warning("The documents did not have enough to explain this, so the explanation part is incomplete.")
        _sql_evidence(outcome, timing=False)
        _passages_evidence(outcome)
        st.caption(f"Answered in {outcome.elapsed_seconds:.1f} s, measured from this page. Two sources were used, one after the other.")
        return

    if kind == client.REJECTED:
        st.warning(f"This question was not answered. {outcome.message}")
        st.caption("Try a question about fraud rates, merchants, trends, or the fraud documents.")
        return

    if kind == client.RATE_LIMITED:
        wait = f"Try again in {outcome.retry_after_seconds} seconds." if outcome.retry_after_seconds is not None else "Try again in a minute."
        st.warning(f"The API is limiting requests right now. {wait}")
        return

    if kind == client.TIMEOUT:
        st.error(f"{outcome.message} The API may still be waking up. Ask again in a moment.")
        return

    if kind == client.UNREACHABLE:
        st.error("Could not reach the API. Check that it is running, then ask again.")
        return

    if kind == client.API_ERROR:
        st.error("The API failed while answering this question. Ask again, or try a different question.")
        return

    st.error("The API sent back something this page does not understand. Ask again.")


# ---- Input ---------------------------------------------------------------

if "question" not in st.session_state:
    st.session_state.question = ""
if "outcome" not in st.session_state:
    st.session_state.outcome = None

for idx, example in enumerate(EXAMPLES):
    if st.button(example, key=f"example-{idx}", use_container_width=True):
        st.session_state.question = example
        st.session_state.run_now = True

with st.form("ask", border=False):
    text = st.text_area(
        "Your question",
        value=st.session_state.question,
        max_chars=MAX_QUESTION_CHARS,
        height=90,
        placeholder="Which merchants have the highest fraud incidence?",
    )
    submitted = st.form_submit_button("Ask", type="primary")

run_now = st.session_state.pop("run_now", False)
if (submitted or run_now) and (text if submitted else st.session_state.question).strip():
    question = (text if submitted else st.session_state.question).strip()
    st.session_state.question = question
    with st.spinner("Waiting for the API. If it was idle, it may need up to 45 seconds to wake."):
        st.session_state.outcome = client.ask(API_URL, question)

if st.session_state.outcome is not None:
    st.divider()
    render_outcome(st.session_state.outcome)

st.divider()
st.caption(
    f"Runs against the live API. Read how it was built: [the Part 5 article]({ARTICLE_URL}) · "
    f"[the repository]({REPO_URL})"
)
