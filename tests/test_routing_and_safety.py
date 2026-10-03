"""
Deterministic (no LLM / DB) tests for routing, guardrails, SQL safety and
client-IP resolution.
"""

import pytest

from src.orchestrator import detect_intent_heuristic
from src.safety.guardrails import is_domain_related
from src.analytics.fraud_analytics import is_safe_select, format_sql, nl_to_sql


@pytest.mark.parametrize("query,expected", [
    ("Which merchants or merchant categories exhibit the highest incidence of fraudulent transactions?", "analytics"),
    ("Top 10 merchants by fraud count", "analytics"),
    ("Merchant dengan penipuan tertinggi?", "analytics"),
    ("How does the daily fraud rate fluctuate over time?", "analytics"),
    ("What are the primary methods by which credit card fraud is committed?", "rag"),
    ("How much higher are fraud rates when the counterpart is outside the EEA?", "rag"),
    ("What share of card fraud value was due to cross-border transactions according to the EBA report?", "rag"),
])
def test_heuristic_routes_to_best_path(query, expected):
    intent, conf = detect_intent_heuristic(query)
    assert intent == expected
    assert conf >= 0.80


def test_bare_merchant_mention_is_ambiguous_and_defers_to_llm():
    intent, conf = detect_intent_heuristic("Which merchant behaviors are linked to fraud exposure?")
    assert conf < 0.80


def test_merchant_ranking_uses_sql_template_not_llm():
    sql = nl_to_sql("which merchants have the highest fraud", "merchant_rank")
    assert "fraud_transactions" in sql and "GROUP BY merchant" in sql
    assert format_sql(sql).startswith("SELECT")


@pytest.mark.parametrize("sql,ok", [
    ("SELECT category, COUNT(*) FROM fraud_transactions GROUP BY category", True),
    ("SELECT created_at FROM fraud_transactions", True),
    ("SELECT 1; DROP TABLE fraud_transactions", False),
    ("SELECT pg_sleep(10)", False),
    ("SELECT * FROM information_schema.tables", False),
    ("DELETE FROM fraud_transactions", False),
    ("WITH x AS (SELECT 1) SELECT * FROM x", False),
])
def test_is_safe_select(sql, ok):
    assert is_safe_select(sql) is ok


@pytest.mark.parametrize("query,ok", [
    ("monthly trend of transactions", True),
    ("top merchants", True),
    ("What is card fraud?", True),
    ("Apa itu penipuan kartu?", True),
    ("remember to member the number", False),   # 'eba'-like substrings must not match
    ("what is the weather today", False),
])
def test_domain_keywords_use_word_boundaries(query, ok):
    assert is_domain_related(query) is ok


def test_rate_limit_uses_trusted_forwarded_for(monkeypatch):
    from starlette.requests import Request
    import src.safety.rate_limit as rl

    def req(xff):
        return Request({
            "type": "http", "headers": [(b"x-forwarded-for", xff.encode())],
            "client": ("10.0.0.1", 1),
        })

    monkeypatch.setattr(rl, "TRUST_FORWARDED_FOR", False)
    assert rl.get_client_ip(req("1.1.1.1")) == "10.0.0.1"

    monkeypatch.setattr(rl, "TRUST_FORWARDED_FOR", True)
    monkeypatch.setattr(rl, "TRUSTED_PROXY_HOPS", 1)
    # leftmost is spoofable; the proxy-appended rightmost entry wins
    assert rl.get_client_ip(req("6.6.6.6, 2.2.2.2")) == "2.2.2.2"
