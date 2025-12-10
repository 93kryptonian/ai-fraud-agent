# src/analytics/fraud_analytics.py

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
# CONFIG
# =============================================================================

USE_LLM_SQL = os.getenv("ANALYTICS_USE_LLM_SQL", "1") != "0"
USE_LLM_SUMMARY = os.getenv("ANALYTICS_USE_LLM_SUMMARY", "0") == "1"
MAX_ROWS_RETURN = int(os.getenv("ANALYTICS_MAX_ROWS", "1000"))


# =============================================================================
# SMALL HELPERS
# =============================================================================

def _strip_markdown(sql: str) -> str:
    """
    Remove ```sql fences etc from LLM output.
    """
    return sql.replace("```sql", "").replace("```", "").strip()


def _is_safe_select(sql: str) -> bool:
    """
    Enforce SELECT-only, prevent DDL/DML.
    """
    s = sql.strip().lower()
    if not s.startswith("select"):
        return False
    forbidden = ["insert ", "delete ", "update ", "drop ", "alter ", "create "]
    return not any(tok in s for tok in forbidden)


def _clean_sql_whitespace(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip()


def strip_html(text: str) -> str:
    if not isinstance(text, str):
        return text
    return re.sub(r"<[^>]+>", "", text)

# =============================================================================
# ANALYTICS INTENT CLASSIFICATION 
# =============================================================================

def classify_analytics_intent(nl_query: str) -> str:
    """
    Lightweight classifier for analytics questions.

    Returns one of:
      - "timeseries"   → fraud rate over time
      - "merchant_rank"
      - "category_rank"
      - "generic"
    """
    q = nl_query.lower()

    if any(k in q for k in ["merchant", "merchants", "toko"]):
        return "merchant_rank"

    if any(k in q for k in ["category", "categories", "kategori"]):
        return "category_rank"

    if any(k in q for k in [
        "daily", "harian", "per day",
        "monthly", "bulanan", "per month",
        "trend", "over time", "two year", "two years", "fluctuate", "fluctuation"
    ]):
        return "timeseries"

    return "generic"


# =============================================================================
# FALLBACK SQL BUILDERS 
# =============================================================================

def _extract_years_from_query(nl_query: str) -> List[int]:
    years = re.findall(r"\b(20\d{2})\b", nl_query.lower())
    uniques = sorted({int(y) for y in years})
    return uniques


def fallback_time_series_sql(nl_query: str, period: str = "month") -> str:
    """
    Safe, opinionated SQL template for fraud rate time series.

    period: "day" or "month"
    """
    years = _extract_years_from_query(nl_query)
    year_clause = ""
    if years:
        year_list = ",".join(str(y) for y in years)
        year_clause = f"WHERE EXTRACT(YEAR FROM trans_date_trans_time) IN ({year_list})"

    if period == "day":
        date_expr = "DATE(trans_date_trans_time)"
    else:
        date_expr = "DATE_TRUNC('month', trans_date_trans_time)"

    return f"""
        SELECT
            {date_expr} AS date,
            COUNT(*) FILTER (WHERE isfraud = TRUE)::float /
            NULLIF(COUNT(*),0)::float AS fraud_rate
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
            COUNT(*) FILTER (WHERE isfraud = TRUE)::float / NULLIF(COUNT(*), 0)::float AS fraud_rate
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
            COUNT(*) FILTER (WHERE isfraud = TRUE)::float / NULLIF(COUNT(*), 0)::float AS fraud_rate
        FROM fraud_transactions
        GROUP BY merchant
        HAVING COUNT(*) FILTER (WHERE isfraud = TRUE) > 0
        ORDER BY fraud_count DESC
        LIMIT 20;
    """


# =============================================================================
# NL → SQL 
# =============================================================================

def nl_to_sql(nl_query: str, intent: str) -> str:
    """
    NL → SQL with:
      - hard overrides for ranking questions
      - controlled time-series patterns
      - safe LLM-only fallback for generic queries
    """
    q = nl_query.lower()

    # Hard override for merchant ranking
    if intent == "merchant_rank":
        logger.info("[analytics] Using strict merchant ranking SQL")
        return fallback_merchant_fraud_sql()

    # Hard override for category ranking
    if intent == "category_rank":
        logger.info("[analytics] Using strict category ranking SQL")
        return fallback_count_by_category_sql()

    # Time-series: fraud rate over time
    if intent == "timeseries":
        logger.info("[analytics] Using strict timeseries fraud_rate SQL")
        # decide daily vs monthly
        period = "day" if any(k in q for k in ["daily", "harian", "per day"]) else "month"
        return fallback_time_series_sql(nl_query, period=period)

    # Generic: use LLM to map question → SQL with strict schema
    if not USE_LLM_SQL:
        raise ValueError("LLM SQL generation disabled and no strict template available.")

    prompt = NL_TO_SQL_PROMPT.format(q=nl_query)
    raw = llm.run(prompt, temperature=0.0)
    if not isinstance(raw, str):
        raw = str(raw)

    sql = _strip_markdown(raw)
    sql = _clean_sql_whitespace(sql)

    if not _is_safe_select(sql):
        raise ValueError(f"Unsafe or invalid SQL generated: {sql}")

    return sql


# =============================================================================
# SQL EXECUTION LAYER
# =============================================================================

def execute_sql(sql: str) -> pd.DataFrame:
    """
    Execute SQL against Supabase and return DataFrame.
    """
    sql_clean = _clean_sql_whitespace(sql)
    logger.info(f"[analytics] EXECUTING SQL:\n{sql_clean}")

    rows = DB.sql(sql_clean)
    if not rows:
        logger.info("[analytics] SQL returned 0 rows.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if len(df) > MAX_ROWS_RETURN:
        df = df.head(MAX_ROWS_RETURN)
        logger.info(f"[analytics] Truncated rows to {MAX_ROWS_RETURN} for safety.")

    logger.info(f"[analytics] SQL returned {len(df)} rows | columns={list(df.columns)}")
    return df


# =============================================================================
# CHART BUILDER
# =============================================================================

def to_chart_data(df: pd.DataFrame) -> Optional[list]:
    """
    Convert a DataFrame into simple chart format understood by UI:
      [{ "label": ..., "value": ... }, ...]
    """
    chart = []

    # Time-series: date + fraud_rate or value
    if "date" in df.columns and ("fraud_rate" in df.columns or "value" in df.columns):
        val = "fraud_rate" if "fraud_rate" in df.columns else "value"
        for _, r in df.iterrows():
            chart.append({"label": str(r["date"]), "value": float(r[val])})
        return chart

    # Category / Merchant ranking
    if ("category" in df.columns or "merchant" in df.columns) and "fraud_count" in df.columns:
        label_col = "category" if "category" in df.columns else "merchant"
        for _, r in df.iterrows():
            chart.append({"label": str(r[label_col]), "value": float(r["fraud_count"])})
        return chart

    return None


# =============================================================================
# ANALYTICS SUMMARY (PYTHONIC, DETERMINISTIC)
# =============================================================================

def summarize_timeseries(df: pd.DataFrame, lang: str) -> Tuple[str, float]:
    """
    Summarize fraud_rate timeseries: min, max, avg, volatility, trend.
    Returns (summary, confidence).
    """
    if "fraud_rate" not in df.columns or "date" not in df.columns:
        msg = "Data does not contain fraud rate over time." if lang == "en" else \
              "Data tidak mengandung informasi tingkat fraud dari waktu ke waktu."
        return msg, 0.2

    if len(df) < 2:
        msg = "Data is insufficient for time-series analysis." if lang == "en" else \
              "Data tidak memadai untuk analisis time-series."
        return msg, 0.2

    fr = df["fraud_rate"].astype(float)
    if fr.nunique() <= 1:
        msg = "Fraud rate appears almost constant; insufficient fluctuation for analysis." if lang == "en" else \
              "Tingkat fraud tampak hampir konstan; tidak cukup fluktuasi untuk dianalisis."
        return msg, 0.4

    lo = fr.min()
    hi = fr.max()
    avg = fr.mean()
    std = fr.std()
    vol = std / (avg + 1e-8)

    x = np.arange(len(df))
    trend_coeff = np.polyfit(x, fr.values, 1)[0]
    if trend_coeff > 0.0005:
        trend_desc = "increasing" if lang == "en" else "meningkat"
    elif trend_coeff < -0.0005:
        trend_desc = "decreasing" if lang == "en" else "menurun"
    else:
        trend_desc = "roughly stable" if lang == "en" else "cenderung stabil"

    peak_idx = fr.idxmax()
    low_idx = fr.idxmin()
    peak_date = df.loc[peak_idx, "date"]
    low_date = df.loc[low_idx, "date"]

    if lang.startswith("id"):
        summary = (
            f"Tingkat fraud berfluktuasi sepanjang periode pengamatan dengan rentang "
            f"{lo:.4f}-{hi:.4f} dan rata-rata sekitar {avg:.4f}. "
            f"Secara umum tren terlihat {trend_desc}. "
            f"Nilai tertinggi terjadi pada {peak_date} sebesar {hi:.4f}, "
            f"sementara nilai terendah terjadi pada {low_date} sebesar {lo:.4f}. "
            f"Volatilitas relatif sekitar {vol:.2f}, berdasarkan {len(df)} titik waktu."
        )
    else:
        summary = (
            f"Fraud rates fluctuate over the observed period with a range of "
            f"{lo:.4f}-{hi:.4f} and an average of about {avg:.4f}. "
            f"The overall trend appears {trend_desc}. "
            f"The highest rate occurs on {peak_date} at {hi:.4f}, "
            f"while the lowest occurs on {low_date} at {lo:.4f}. "
            f"Relative volatility is approximately {vol:.2f}, based on {len(df)} time points."
        )

    # Confidence: more points + non-trivial variance -> higher
    conf = 0.5 + min(0.4, np.log10(len(df) + 1) * 0.1 + min(0.3, vol))
    conf = float(max(0.0, min(1.0, conf)))

    return summary, conf


def summarize_ranking(df: pd.DataFrame, lang: str) -> Tuple[str, float]:
    """
    Summarize merchant/category ranking table.
    Returns (summary, confidence).
    """
    if ("merchant" not in df.columns and "category" not in df.columns) or "fraud_count" not in df.columns:
        msg = "Data does not contain merchant/category fraud breakdown." if lang == "en" else \
              "Data tidak mengandung rincian fraud per merchant/kategori."
        return msg, 0.3

    label_col = "merchant" if "merchant" in df.columns else "category"
    df_sorted = df.sort_values("fraud_count", ascending=False)

    top = df_sorted.iloc[0]
    top_name = str(top[label_col])
    top_count = int(top["fraud_count"])
    top_total = int(top.get("total_count", top["fraud_count"]))
    top_rate = float(top.get("fraud_rate", top["fraud_count"] / max(1, top_total)))

    top5 = df_sorted.head(5)
    total_fraud = int(df_sorted["fraud_count"].sum())

    if lang.startswith("id"):
        summary = (
            f"Entitas dengan insiden fraud tertinggi adalah {top_name} "
            f"dengan {top_count} kasus fraud dari sekitar {top_total} transaksi "
            f"(fraud rate ~{top_rate:.2%}). "
            f"Secara keseluruhan, terdapat {total_fraud} kasus fraud "
            f"yang tersebar pada {len(df_sorted)} merchant/kategori. "
            f"Kelompok 5 besar memberikan kontribusi signifikan terhadap total insiden."
        )
    else:
        summary = (
            f"The entity with the highest fraud incidence is {top_name}, "
            f"with {top_count} fraud cases out of roughly {top_total} transactions "
            f"(fraud rate ~{top_rate:.2%}). "
            f"In total, there are {total_fraud} fraud cases across "
            f"{len(df_sorted)} merchants/categories, "
            f"with the top-5 contributing a significant portion of overall fraud volume."
        )

    conf = 0.6 + min(0.3, np.log10(len(df_sorted) + 1) * 0.1)
    conf = float(max(0.0, min(1.0, conf)))
    return summary, conf


def summarize_generic(df: pd.DataFrame, lang: str) -> Tuple[str, float]:
    """
    Generic catch-all summary when we don't recognize a special structure.
    """
    if df.empty:
        msg = "No data available to answer the question." if lang == "en" else \
              "Tidak ada data yang tersedia untuk menjawab pertanyaan."
        return msg, 0.0

    if lang.startswith("id"):
        summary = f"Analisis berhasil memproses {len(df)} baris data dan {len(df.columns)} kolom."
    else:
        summary = f"Analysis successfully processed {len(df)} rows and {len(df.columns)} columns."

    conf = 0.5
    return summary, conf


# =============================================================================
# OPTIONAL LLM-REFINED SUMMARY
# =============================================================================

def refine_summary_with_llm(summary: str, df: pd.DataFrame, lang: str) -> str:
    """
    Use ANALYTICS_SYSTEM_PROMPT to polish the Python-generated summary.
    Only run if USE_LLM_SUMMARY=True.
    """
    if not USE_LLM_SUMMARY:
        return summary

    # take small sample of data for context
    sample = df.head(5).to_dict(orient="records")

    prompt = ANALYTICS_SYSTEM_PROMPT + f"""

Language: {"Indonesian" if lang.startswith("id") else "English"}

Here is the initial summary:
{summary}

Here is a small sample of the underlying data (JSON):
{sample}

Refine the summary to be slightly clearer and more user-friendly,
but do NOT change any numbers or introduce new facts.
"""

    try:
        resp = llm.run(prompt, temperature=0.1)
        if isinstance(resp, str) and resp.strip():
            return resp.strip()
        return summary
    except Exception as e:
        logger.warning(f"[analytics] LLM summary refinement failed: {e}")
        return summary


# =============================================================================
# MAIN ANALYTICS ENGINE
# =============================================================================

def run_analytics(nl_query: str, lang: str = "en") -> Dict[str, Any]:
    """
    Main entrypoint for fraud analytics.

    Steps:
      1) Domain guardrail (must be fraud-related numeric question)
      2) Classify analytics intent (timeseries / ranking / generic)
      3) NL → SQL 
      4) Execute SQL
      5) Validate shape
      6) Build summary + chart
      7) Optionally refine summary with LLM
    """

    try:
        q_lower = nl_query.lower()

        # 1) Domain guardrail
        fraud_keywords = [
            "fraud", "fraud rate", "penipuan", "nilai fraud",
            "isfraud", "transaksi", "harian", "bulanan",
            "daily", "monthly", "over time", "trend", "fluctuate", "fluctuation"
        ]
        if not any(k in q_lower for k in fraud_keywords):
            msg = (
                "Sorry, I can only run analytics on fraud-related transaction data."
                if lang == "en" else
                "Maaf, saya hanya dapat menjalankan analitik pada data transaksi terkait fraud."
            )
            return AnalyticsResponse(answer=msg, data_points=None, chart_data=None, confidence=0.0).model_dump()

        # 2) Intent classification within analytics
        intent = classify_analytics_intent(nl_query)
        logger.info(f"[analytics] intent={intent} | query={nl_query!r}")

        # 3) NL → SQL
        df = pd.DataFrame()
        try:
            sql = nl_to_sql(nl_query, intent=intent)
            df = execute_sql(sql)
        except Exception as e:
            logger.warning(f"[analytics] NL->SQL generation failed: {e}")

        # 4) Fallback templates if LLM SQL failed or returned empty
        if df.empty:
            logger.info("[analytics] Trying fallback SQL templates due to empty result.")
            if intent == "timeseries":
                # daily vs monthly already handled in fallback_time_series_sql
                sql = fallback_time_series_sql(nl_query)
                df = execute_sql(sql)
            elif intent == "category_rank":
                sql = fallback_count_by_category_sql()
                df = execute_sql(sql)
            elif intent == "merchant_rank":
                sql = fallback_merchant_fraud_sql()
                df = execute_sql(sql)

        # 5) No data → graceful message
        if df.empty:
            msg = "Insufficient data to answer the question." if lang == "en" else \
                  "Data tidak mencukupi untuk menjawab pertanyaan tersebut."
            return AnalyticsResponse(answer=msg, data_points=None, chart_data=None, confidence=0.0).model_dump()

        # 6) Validate shape vs intent & build summary
        if intent == "timeseries":
            summary, conf = summarize_timeseries(df, lang)
        elif intent in {"merchant_rank", "category_rank"}:
            summary, conf = summarize_ranking(df, lang)
        else:
            summary, conf = summarize_generic(df, lang)

        # 7) Chart data
        chart = to_chart_data(df)

        # 8) Optional LLM refinement of summary
        summary_refined = refine_summary_with_llm(summary, df, lang)

        # 9) Build final response
        resp = AnalyticsResponse(
            answer=strip_html(summary_refined),
            data_points=df.to_dict(orient="records"),
            chart_data=chart,
            confidence=conf,
        ).model_dump()

        logger.info(f"[analytics] Final analytics payload constructed with confidence={conf:.3f}")
        return resp

    except Exception as e:
        logger.error(f"[analytics] Analytics pipeline failed: {e}", exc_info=True)
        return ErrorResponse(
            error="Analytics failed.",
            details=str(e)
        ).model_dump()
