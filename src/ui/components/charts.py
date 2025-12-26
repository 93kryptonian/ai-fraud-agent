"""
CHART RENDERING COMPONENT
-------------------------

Renders simple analytics charts from structured data.

Expected input:
    chart_data = [
        {"label": <str>, "value": <number>},
        ...
    ]

Rendering rules:
- Date-like labels → time-series line chart
- Otherwise → horizontal bar chart (ranking)
"""

from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import altair as alt


# =============================================================================
# PUBLIC API
# =============================================================================

def render_chart(chart_data: List[Dict[str, Any]]):
    """
    Render analytics chart from chart_data.

    Heuristics:
    - If labels can be parsed as datetime → line chart
    - Else → bar chart
    """
    if not chart_data:
        st.info("No chart data to display.")
        return

    try:
        df = pd.DataFrame(chart_data)

        # -------------------------------------------------
        # Validate schema
        # -------------------------------------------------
        if not {"label", "value"}.issubset(df.columns):
            st.warning("Chart data is not in the expected format.")
            st.json(chart_data)
            return

        # Normalize columns
        df["label_str"] = df["label"].astype(str)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # -------------------------------------------------
        # Detect date-like labels
        # -------------------------------------------------
        is_time_series = False
        try:
            df["label_dt"] = pd.to_datetime(
                df["label_str"],
                errors="raise",
            )
            is_time_series = True
        except Exception:
            is_time_series = False

        # -------------------------------------------------
        # Build chart
        # -------------------------------------------------
        if is_time_series:
            chart = (
                alt.Chart(df)
                .mark_line(point=True)
                .encode(
                    x=alt.X(
                        "label_dt:T",
                        title="Date",
                    ),
                    y=alt.Y(
                        "value:Q",
                        title="Fraud rate / value",
                    ),
                    tooltip=["label_str", "value"],
                )
                .properties(height=260)
            )
        else:
            chart = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "value:Q",
                        title="Fraud count / value",
                    ),
                    y=alt.Y(
                        "label_str:N",
                        sort="-x",
                        title="Category / Merchant",
                    ),
                    tooltip=["label_str", "value"],
                )
                .properties(height=260)
            )

        st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.warning(f"Failed to render chart: {e}")
        st.json(chart_data)
