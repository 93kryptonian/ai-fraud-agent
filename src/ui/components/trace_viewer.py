"""
TRACE / CITATION VIEWER COMPONENT
--------------------------------

Renders document citations used by the RAG pipeline.

Each citation item is expected to contain:
- source : document name
- page   : page number (optional)
- preview: short text snippet

Design goals:
- Simple, readable citation display
- Defensive against missing / noisy data
- UI-only (no business logic)
"""

import re
import streamlit as st
from typing import List, Dict, Any


# =============================================================================
# TEXT CLEANUP
# =============================================================================

_PAGE_FOOTER_PATTERN = re.compile(
    r"Page\s+\d+\s+of\s+\d+",
    flags=re.IGNORECASE,
)


def clean_preview(text: str) -> str:
    """
    Remove common PDF footer artifacts (e.g. 'Page X of Y')
    from preview snippets.
    """
    if not text:
        return ""
    return _PAGE_FOOTER_PATTERN.sub("", text).strip()


# =============================================================================
# PUBLIC RENDER FUNCTION
# =============================================================================

def render_trace(citations: List[Dict[str, Any]]):
    """
    Render a list of citations in expandable sections.

    Expected citation schema:
        {
            "source": str,
            "page": int | None,
            "preview": str | None
        }
    """
    if not citations:
        st.info("No citations available.")
        return

    for idx, citation in enumerate(citations, start=1):
        source = citation.get("source", "Unknown source")
        page = citation.get("page")
        preview = clean_preview(
            (citation.get("preview") or "").strip()
        )

        # Build expander title
        title = f"{idx}. {source}"
        if isinstance(page, int) and page >= 0:
            title += f" · page {page}"

        with st.expander(title, expanded=False):
            if preview:
                st.write(preview)
            else:
                st.write("_No preview snippet available._")
