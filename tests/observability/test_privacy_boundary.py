# tests/observability/test_privacy_boundary.py
"""
The contract's §6 states, as a KNOWN GAP, which modules write raw query text to
application INFO logs. The statement is only honest while it matches the code,
so tie the two together: if a log site is added, removed or fixed, this fails
until the contract is updated (and, if the gap is closed, the finding removed).
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]

# an f-string placeholder rendering a query variable, e.g. query={query!r},
# rewritten_query={rewritten_query!r}, query={nl_query!r}
RAW_QUERY_IN_LOG = re.compile(r"query\w*=\{\w+!r\}")


def _modules_logging_raw_query():
    modules = set()
    for base in ("src", "api"):
        for path in (ROOT / base).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            # Deliberately independent of the logger's name (logger / log /
            # logging): an f-string rendering a query variable with !r is the
            # pattern the existing log sites use.
            if RAW_QUERY_IN_LOG.search(text):
                modules.add(".".join(path.relative_to(ROOT).with_suffix("").parts))
    return modules


def _boundary_section():
    doc = (ROOT / "docs" / "observability-contract.md").read_text(encoding="utf-8")
    start = doc.index("### Boundary of this contract: application logs")
    end = doc.index("\n---\n", start)
    return doc[start:end]


def test_contract_lists_exactly_the_modules_that_log_raw_query_text():
    section = _boundary_section()
    documented = set(re.findall(r"`(src\.[a-z_.]+)`", section))

    assert _modules_logging_raw_query() == documented, (
        "the §6 'application logs' finding no longer matches the code: update the "
        "contract (and remove the finding if the gap has been closed)"
    )


def test_boundary_section_scopes_its_claims():
    section = _boundary_section()
    assert "uvicorn" in section and "access log" in section
    assert "has **not been verified**" in section          # no claim about Render's proxy
    assert "**not** fixed by M7 or M8" in section
    assert "do not contain query text" in section


def test_telemetry_surfaces_still_carry_no_query_text():
    """The other half of the statement: events and signals are query-text free."""
    from src.observability import dimensions as dim

    for (event, key), rule in dim.POLICY.items():
        assert "query" not in key or key in {"query_length", "query_hash"}, (event, key)
    assert dim.classify("guardrails.completed", "query_hash").kind == dim.CORRELATION
    assert dim.classify("guardrails.completed", "query_length").kind == dim.MEASURE
