# ui/components/trace_viewer.py

import streamlit as st
import re


def clean_preview(text: str) -> str:
    return re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE).strip()


def render_trace(citations):
    """
    Render list of citations:

    Each item:
      {
        "source": str,
        "page": int | None,
        "preview": str
      }
    """
    if not citations:
        st.info("No citations available.")
        return

    for i, cit in enumerate(citations, start=1):
        source = cit.get("source", "Unknown source")
        page = cit.get("page")
        preview = clean_preview(cit.get("preview", "").strip())

        title = f"{i}. {source}"
        if page is not None and page != -1:
            title += f" · page {page}"

        with st.expander(title, expanded=False):
            if preview:
                st.write(preview)
            else:
                st.write("_No preview snippet available._")
