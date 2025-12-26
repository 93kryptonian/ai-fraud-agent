# src/analytics/fraud_analytics.py
"""
Fraud analytics engine.

This module translates natural-language fraud questions into:
- deterministic SQL queries (with strict guardrails),
- structured analytics results,
- explainable summaries and chart-friendly outputs.

Design principles:
- Safety first (SELECT-only SQL, bounded rows)
- Deterministic analytics before LLM refinement
- Graceful fallback paths
"""

import os
import re
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd

from src.db.supabase_client import DB
from src.llm.llm_client import llm
from src.llm.prompts import ANALYTICS_SYSTEM_PROMPT, NL_TO_SQL_PROMPT
from src.llm.response_schema import AnalyticsResponse, ErrorResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

USE_LLM_SQL = os.getenv("ANALYTICS_USE_LLM_SQL", "1") != "0"
USE_LLM_SUMMARY = os.getenv("ANALYTICS_USE_LLM_SUMMARY", "0") == "1"
MAX_ROWS_RETURN = int(os.getenv("ANALYTICS_MAX_ROWS", "1000"))

# =============================================================================
# SQL & TEXT SANITIZATION HELPERS
# =============================================================================

def strip_markdown(sql: str) -> str:
    """Remove markdown fences from LLM-generated SQL."""
    return sql.replace("```sql", "").replace("```", "").strip()


def normalize_sql(sql: str) -> str:
    """Normalize whitespace for logging and execution."""
    return re.sub(r"\s+", " ", sql).strip()


def is_safe_select(sql: str) -> bool:
    """
    Enforce SELECT-only SQL.
    Blocks DDL/DML to prevent destructive queries.
    """
    s = sql.strip().lower()
    if not s.startswith("select"):
        return False

    forbidden = ("insert ", "delete ", "update ", "drop ", "alter ", "create ")
    return not any(tok in s for tok in forbidden)


def strip_html(text: str) -> str:
    """Remove HTML tags from text output."""
    if not isinstance(text, str):
        return text
    return re.sub(r"<[^>]+>", "", text)

# =============================================================================
# ANALYTICS INTENT CLASSIFICATION
# =============================================================================

def classify_analytics_intent(nl_query: str) -> str:
    """
    Lightweight intent classifier for fraud analytics queries.

    Returns:
      - timeseries
      - merchant_rank
      - category_rank
      - generic
    """
    q = nl_query.lower()

    if any(k in q for k in ("merchant", "merchants", "toko")):
        return "merchant_rank"

    if any(k in q for k in ("category", "categories", "kategori")):
        return "category_rank"

    if any(k in q for k in (
        "daily", "harian", "per day",
        "monthly", "bulanan", "per month",
        "trend", "over time",
        "two year", "two years",
        "fluctuate", "fluctuation",
    )):
        return "timeseries"

    return "generic"

# =============================================================================
# FALLBACK SQL BUILDERS (DETERMINISTIC & SAFE)
# =============================================================================

def extract_years_from_query(nl_query: str) -> List[int]:
    years = re.findall(r"\b(20\d{2})\b", nl_query.lower())
    return sorted({int(y) for y in years})


def fallback_time_series_sql(nl_query: str, period: str = "month") -> str:
    """
    Opinionated SQL template for fraud-rate time series.
    """
    years = extract_years_from_query(nl_query)
    year_clause = ""

    if years:
        year_list = ",".join(map(str, years))
        year_clause = (
            f"WHERE EXTRACT(YEAR FROM trans_date_trans_time) IN ({year_list})"
        )

    date_expr = (
        "DATE(trans_date_trans_time)"
        if period == "day"
        else "DATE_TRUNC('month', trans_date_trans_time)"
    )

    return f"""
        SELECT
            {date_expr} AS date,
            COUNT(*) FILTER (WHERE isfraud = TRUE)::float /
            NULLIF(COUNT(*), 0)::float AS fraud_rate
        FROM fraud_transactions
        {year_clause}
        GROUP BY date
        ORDER BY date;
    """


def fallback_count_by_category_sql() -> str:
    return """
        SELECT
            category,
            COUNT(*) FILTER (WHERE isfraud = TRUE) AS fraud_count,
            COUNT(*) AS total_count,
            COUNT(*) FILTER (WHERE isfraud = TRUE)::float /
            NULLIF(COUNT(*), 0)::float AS fraud_rate
        FROM fraud_transactions
        GROUP BY category
        ORDER BY fraud_count DESC
        LIMIT 20;
    """


def fallback_merchant_fraud_sql() -> str:
    return """
        SELECT
            merchant,
            COUNT(*) FILTER (WHERE isfraud = TRUE) AS fraud_count,
            COUNT(*) AS total_count,
            COUNT(*) FILTER (WHERE isfraud = TRUE)::float /
            NULLIF(COUNT(*), 0)::float AS fraud_rate
        FROM fraud_transactions
        GROUP BY merchant
        HAVING COUNT(*) FILTER (WHERE isfraud = TRUE) > 0
        ORDER BY fraud_count DESC
        LIMIT 20;
    """

# =============================================================================
# NL → SQL TRANSLATION
# =============================================================================

def nl_to_sql(nl_query: str, intent: str) -> str:
    """
    Translate natural language to SQL using:
    - strict deterministic templates when possible
    - LLM only for generic queries
    """
    q = nl_query.lower()

    if intent == "merchant_rank":
        logger.info("[analytics] Using merchant ranking template")
        return fallback_merchant_fraud_sql()

    if intent == "category_rank":
        logger.info("[analytics] Using category ranking template")
        return fallback_count_by_category_sql()

    if intent == "timeseries":
        logger.info("[analytics] Using time-series template")
        period = "day" if any(k in q for k in ("daily", "harian", "per day")) else "month"
        return fallback_time_series_sql(nl_query, period)

    if not USE_LLM_SQL:
        raise ValueError("LLM-based SQL generation is disabled.")

    prompt = NL_TO_SQL_PROMPT.format(q=nl_query)
    raw = llm.run(prompt, temperature=0.0)
    sql = normalize_sql(strip_markdown(str(raw)))

    if not is_safe_select(sql):
        raise ValueError(f"Unsafe SQL generated: {sql}")

    return sql

# =============================================================================
# SQL EXECUTION
# =============================================================================

def execute_sql(sql: str) -> pd.DataFrame:
    """Execute SQL via Supabase and return a bounded DataFrame."""
    sql_clean = normalize_sql(sql)
    logger.info(f"[analytics] Executing SQL:\n{sql_clean}")

    rows = DB.sql(sql_clean)
    if not rows:
        logger.info("[analytics] Query returned no rows.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    if len(df) > MAX_ROWS_RETURN:
        df = df.head(MAX_ROWS_RETURN)
        logger.info(f"[analytics] Truncated result to {MAX_ROWS_RETURN} rows.")

    logger.info(
        f"[analytics] Query returned {len(df)} rows | columns={list(df.columns)}"
    )
    return df

# =============================================================================
# CHART TRANSFORMATION
# =============================================================================

def to_chart_data(df: pd.DataFrame) -> Optional[List[Dict[str, float]]]:
    """
    Convert DataFrame into UI-friendly chart format.
    """
    chart: List[Dict[str, float]] = []

    if "date" in df.columns and any(c in df.columns for c in ("fraud_rate", "value")):
        value_col = "fraud_rate" if "fraud_rate" in df.columns else "value"
        for _, r in df.iterrows():
            chart.append({"label": str(r["date"]), "value": float(r[value_col])})
        return chart

    if any(c in df.columns for c in ("merchant", "category")) and "fraud_count" in df.columns:
        label_col = "merchant" if "merchant" in df.columns else "category"
        for _, r in df.iterrows():
            chart.append({"label": str(r[label_col]), "value": float(r["fraud_count"])})
        return chart

    return None

# =============================================================================
# ANALYTICS SUMMARIZATION (DETERMINISTIC)
# =============================================================================

def summarize_timeseries(df: pd.DataFrame, lang: str) -> Tuple[str, float]:
    """
    Generate a deterministic summary for fraud-rate time series.
    """
    if not {"date", "fraud_rate"}.issubset(df.columns):
        msg = (
            "Data does not contain fraud rate over time."
            if lang == "en"
            else "Data tidak mengandung informasi tingkat fraud dari waktu ke waktu."
        )
        return msg, 0.2

    if len(df) < 2:
        msg = (
            "Data is insufficient for time-series analysis."
            if lang == "en"
            else "Data tidak memadai untuk analisis time-series."
        )
        return msg, 0.2

    fr = df["fraud_rate"].astype(float)

    if fr.nunique() <= 1:
        msg = (
            "Fraud rate appears almost constant; insufficient fluctuation for analysis."
            if lang == "en"
            else "Tingkat fraud tampak hampir konstan; tidak cukup fluktuasi."
        )
        return msg, 0.4

    lo, hi, avg = fr.min(), fr.max(), fr.mean()
    std = fr.std()
    volatility = std / (avg + 1e-8)

    x = np.arange(len(df))
    trend = np.polyfit(x, fr.values, 1)[0]

    if trend > 0.0005:
        trend_desc = "increasing" if lang == "en" else "meningkat"
    elif trend < -0.0005:
        trend_desc = "decreasing" if lang == "en" else "menurun"
    else:
        trend_desc = "roughly stable" if lang == "en" else "cenderung stabil"

    peak_date = df.loc[fr.idxmax(), "date"]
    low_date = df.loc[fr.idxmin(), "date"]

    if lang.startswith("id"):
        summary = (
            f"Tingkat fraud berfluktuasi antara {lo:.4f}-{hi:.4f} "
            f"dengan rata-rata {avg:.4f}. Tren keseluruhan {trend_desc}. "
            f"Puncak terjadi pada {peak_date}, terendah pada {low_date}. "
            f"Volatilitas relatif sekitar {volatility:.2f}."
        )
    else:
        summary = (
            f"Fraud rates fluctuate between {lo:.4f}-{hi:.4f} "
            f"with an average of {avg:.4f}. The overall trend is {trend_desc}. "
            f"The peak occurs on {peak_date}, with the lowest on {low_date}. "
            f"Relative volatility is approximately {volatility:.2f}."
        )

    confidence = 0.5 + min(0.4, np.log10(len(df) + 1) * 0.1 + min(0.3, volatility))
    return summary, float(min(1.0, max(0.0, confidence)))


def summarize_ranking(df: pd.DataFrame, lang: str) -> Tuple[str, float]:
    """
    Summarize merchant or category fraud rankings.
    """
    label_col = "merchant" if "merchant" in df.columns else "category"

    if label_col not in df.columns or "fraud_count" not in df.columns:
        msg = (
            "Data does not contain merchant/category fraud breakdown."
            if lang == "en"
            else "Data tidak mengandung rincian fraud per merchant/kategori."
        )
        return msg, 0.3

    df_sorted = df.sort_values("fraud_count", ascending=False)
    top = df_sorted.iloc[0]

    top_rate = float(
        top.get("fraud_rate", top["fraud_count"] / max(1, top.get("total_count", 1)))
    )

    if lang.startswith("id"):
        summary = (
            f"{top[label_col]} memiliki insiden fraud tertinggi "
            f"dengan {int(top['fraud_count'])} kasus "
            f"(fraud rate ~{top_rate:.2%})."
        )
    else:
        summary = (
            f"{top[label_col]} has the highest fraud incidence "
            f"with {int(top['fraud_count'])} cases "
            f"(fraud rate ~{top_rate:.2%})."
        )

    confidence = 0.6 + min(0.3, np.log10(len(df_sorted) + 1) * 0.1)
    return summary, float(min(1.0, max(0.0, confidence)))


def summarize_generic(df: pd.DataFrame, lang: str) -> Tuple[str, float]:
    """Fallback summary for unstructured results."""
    if df.empty:
        msg = (
            "No data available to answer the question."
            if lang == "en"
            else "Tidak ada data untuk menjawab pertanyaan."
        )
        return msg, 0.0

    summary = (
        f"Analysis processed {len(df)} rows and {len(df.columns)} columns."
        if lang == "en"
        else f"Analisis memproses {len(df)} baris dan {len(df.columns)} kolom."
    )
    return summary, 0.5

# =============================================================================
# OPTIONAL LLM REFINEMENT
# =============================================================================

def refine_summary_with_llm(summary: str, df: pd.DataFrame, lang: str) -> str:
    """Light LLM refinement layer (optional, non-authoritative)."""
    if not USE_LLM_SUMMARY:
        return summary

    sample = df.head(5).to_dict(orient="records")
    prompt = f"""{ANALYTICS_SYSTEM_PROMPT}

Language: {"Indonesian" if lang.startswith("id") else "English"}

Initial summary:
{summary}

Sample data (JSON):
{sample}

Refine wording only. Do not change numbers or facts.
"""

    try:
        resp = llm.run(prompt, temperature=0.1)
        return resp.strip() if isinstance(resp, str) and resp.strip() else summary
    except Exception as e:
        logger.warning(f"[analytics] LLM refinement failed: {e}")
        return summary

# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

def run_analytics(nl_query: str, lang: str = "en") -> Dict[str, Any]:
    """
    Main fraud analytics pipeline.
    """
    try:
        q_lower = nl_query.lower()

        fraud_keywords = (
            "fraud", "penipuan", "isfraud", "transaksi",
            "daily", "monthly", "harian", "bulanan",
            "trend", "over time", "fluctuate",
        )

        if not any(k in q_lower for k in fraud_keywords):
            msg = (
                "Sorry, I can only analyze fraud-related transaction data."
                if lang == "en"
                else "Maaf, saya hanya dapat menganalisis data transaksi terkait fraud."
            )
            return AnalyticsResponse(
                answer=msg,
                data_points=None,
                chart_data=None,
                confidence=0.0,
            ).model_dump()

        intent = classify_analytics_intent(nl_query)
        logger.info(f"[analytics] intent={intent} | query={nl_query!r}")

        df = pd.DataFrame()
        try:
            sql = nl_to_sql(nl_query, intent)
            df = execute_sql(sql)
        except Exception as e:
            logger.warning(f"[analytics] NL→SQL failed: {e}")

        if df.empty:
            logger.info("[analytics] Using fallback SQL.")
            if intent == "timeseries":
                df = execute_sql(fallback_time_series_sql(nl_query))
            elif intent == "category_rank":
                df = execute_sql(fallback_count_by_category_sql())
            elif intent == "merchant_rank":
                df = execute_sql(fallback_merchant_fraud_sql())

        if df.empty:
            msg = (
                "Insufficient data to answer the question."
                if lang == "en"
                else "Data tidak mencukupi untuk menjawab pertanyaan."
            )
            return AnalyticsResponse(
                answer=msg,
                data_points=None,
                chart_data=None,
                confidence=0.0,
            ).model_dump()

        if intent == "timeseries":
            summary, conf = summarize_timeseries(df, lang)
        elif intent in {"merchant_rank", "category_rank"}:
            summary, conf = summarize_ranking(df, lang)
        else:
            summary, conf = summarize_generic(df, lang)

        chart = to_chart_data(df)
        summary = refine_summary_with_llm(summary, df, lang)

        return AnalyticsResponse(
            answer=strip_html(summary),
            data_points=df.to_dict(orient="records"),
            chart_data=chart,
            confidence=conf,
        ).model_dump()

    except Exception as e:
        logger.error("[analytics] Pipeline failed", exc_info=True)
        return ErrorResponse(
            error="Analytics failed.",
            details=str(e),
        ).model_dump()
