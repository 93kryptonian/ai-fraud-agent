# ui/components/charts.py

import streamlit as st
import pandas as pd
import altair as alt


def render_chart(chart_data):
    """
    Render chart_data (list of {"label": ..., "value": ...}) as a nice chart.

    - If labels look like dates → line chart (time-series)
    - Otherwise → bar chart (ranking)
    """
    if not chart_data:
        st.info("No chart data to display.")
        return

    try:
        df = pd.DataFrame(chart_data)
        if "label" not in df.columns or "value" not in df.columns:
            st.warning("Chart data is not in the expected format.")
            st.json(chart_data)
            return

        # Try to parse dates
        df["label_str"] = df["label"].astype(str)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # Heuristic: detect date-like labels
        is_date_like = False
        try:
            df["label_dt"] = pd.to_datetime(df["label_str"], errors="raise")
            is_date_like = True
        except Exception:
            is_date_like = False

        if is_date_like:
            c = (
                alt.Chart(df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("label_dt:T", title="Date"),
                    y=alt.Y("value:Q", title="Fraud rate / value"),
                    tooltip=["label_str", "value"],
                )
                .properties(height=260)
            )
        else:
            # Bar chart for categories/merchants
            c = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X("value:Q", title="Fraud count / value"),
                    y=alt.Y("label_str:N", sort="-x", title="Category / Merchant"),
                    tooltip=["label_str", "value"],
                )
                .properties(height=260)
            )

        st.altair_chart(c, use_container_width=True)

    except Exception as e:
        st.warning(f"Failed to render chart: {e}")
        st.json(chart_data)
