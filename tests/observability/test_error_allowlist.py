# tests/observability/test_error_allowlist.py
"""
Error-class allowlist (M7.2, expanded after the error-type audit).

`error_type` / `primary_error_type` are dimension-safe only after
normalization: a known exception class passes through, anything else becomes
`other`. These tests pin the exact set, the intentional exclusions, and that
each allowlisted name is a real class in the library it is attributed to (so a
typo cannot silently turn a real failure into `other`).
"""

import pytest

from src.observability import dimensions as dim

PRE_EXISTING = {
    "ConnectionError", "TimeoutError", "ValueError", "KeyError", "TypeError", "RuntimeError",
    "OSError", "AttributeError", "IndexError", "LLMExhaustedRetriesError", "APIConnectionError",
    "APITimeoutError", "RateLimitError", "OperationalError",
}
ADDED = {
    "QueryCanceled", "AuthenticationError", "InternalServerError",
    "ConnectError", "ReadTimeout", "ConnectTimeout", "APIError",
}
DELIBERATELY_EXCLUDED = {
    "BadRequestError", "ProgrammingError", "NotFoundError", "SupabaseException",
    "JSONDecodeError", "ValidationError", "PermissionDeniedError", "APIStatusError",
    # psycopg2 SQLSTATE subclasses: a separate database-taxonomy question
    "UndefinedTable", "UndefinedColumn", "SyntaxError", "InsufficientPrivilege",
    "ReadOnlySqlTransaction", "ConnectionFailure",
}

ERROR_EVENTS = [
    ("llm.failed", "error_type"),
    ("retrieval.failed", "error_type"),
    ("ranking.failed", "error_type"),
    ("intent.failed", "error_type"),
    ("analytics.sql.failed", "error_type"),
    ("analytics.sql.failed", "primary_error_type"),
    ("analytics.completed", "error_type"),
    ("request.completed", "error_type"),
]


def test_exact_allowlist_membership_and_count():
    assert dim.ERROR_TYPES == PRE_EXISTING | ADDED
    assert len(dim.ERROR_TYPES) == 21
    assert not (PRE_EXISTING & ADDED)


def test_existing_classes_are_preserved():
    assert PRE_EXISTING <= dim.ERROR_TYPES


@pytest.mark.parametrize("event,key", ERROR_EVENTS)
@pytest.mark.parametrize("name", sorted(ADDED))
def test_each_new_class_passes_through_every_error_dimension(event, key, name):
    assert dim.dimension_value(event, key, name) == name


@pytest.mark.parametrize("name", sorted(DELIBERATELY_EXCLUDED))
def test_deliberately_excluded_classes_collapse_to_other(name):
    assert name not in dim.ERROR_TYPES
    assert dim.dimension_value("llm.failed", "error_type", name) == dim.OTHER
    assert dim.dimension_value("analytics.sql.failed", "primary_error_type", name) == dim.OTHER


def test_unknown_and_absent_values_are_unchanged():
    assert dim.dimension_value("retrieval.failed", "error_type", "SomeBespokeError") == dim.OTHER
    assert dim.dimension_value("analytics.sql.completed", "primary_error_type", None) == "none"
    assert dim.dimension_value("retrieval.failed", "error_type", dim.OTHER) == dim.OTHER


def test_bounds_after_the_expansion_stay_under_the_cap_with_real_headroom():
    assert dim.series_upper_bound("llm.failed") == 1 * 10 * 5 * (21 + 2) == 1150
    errored = {ev for (ev, _k), r in dim.POLICY.items() if r.domain == "error_types"}
    for event in errored:
        assert dim.series_upper_bound(event) <= dim.CARDINALITY_CAP, event


# --- every allowlisted name is a REAL class where we say it comes from ------

LIBRARY_CLASSES = [
    ("openai", "APIConnectionError"), ("openai", "APITimeoutError"), ("openai", "RateLimitError"),
    ("openai", "AuthenticationError"), ("openai", "InternalServerError"),
    ("httpx", "ConnectError"), ("httpx", "ReadTimeout"), ("httpx", "ConnectTimeout"),
    ("psycopg2", "OperationalError"), ("postgrest.exceptions", "APIError"),
]


@pytest.mark.parametrize("module,name", LIBRARY_CLASSES)
def test_allowlisted_names_exist_in_their_libraries(module, name):
    mod = pytest.importorskip(module)
    cls = getattr(mod, name)
    assert issubclass(cls, BaseException) and cls.__name__ == name
    assert name in dim.ERROR_TYPES


def test_query_canceled_is_the_name_psycopg2_raises_for_a_statement_timeout():
    errors = pytest.importorskip("psycopg2.errors")
    cls = errors.lookup("57014")          # SQLSTATE query_canceled: what statement_timeout raises
    assert cls.__name__ == "QueryCanceled"
    assert cls.__name__ in dim.ERROR_TYPES
    # its base class name does NOT match it, which is why it needs its own entry
    assert "OperationalError" in {b.__name__ for b in cls.__mro__}
    assert cls.__name__ != "OperationalError"


def test_httpx_transport_errors_are_not_the_builtin_classes_already_allowlisted():
    httpx = pytest.importorskip("httpx")
    assert not issubclass(httpx.ConnectError, ConnectionError)
    assert not issubclass(httpx.ReadTimeout, TimeoutError)
