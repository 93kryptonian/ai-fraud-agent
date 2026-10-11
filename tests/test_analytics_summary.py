"""
The generic analytics summary must state the values it computed.

Found by the evaluation set: "How many fraudulent transactions are there in total?"
ran correct SQL and answered "Analysis processed 1 rows and 1 columns." with no number.
"""

import numpy as np
import pandas as pd

from src.analytics.fraud_analytics import summarize_generic


def test_a_count_is_stated_with_its_label():
    df = pd.DataFrame([{"total_fraudulent_transactions": 7506}])
    answer, _ = summarize_generic(df, "en")
    assert "7,506" in answer and "total fraudulent transactions" in answer
    assert "rows and" not in answer          # the old shape-only wording


def test_numpy_integers_and_floats_are_formatted():
    df = pd.DataFrame({"n": np.array([1234567], dtype=np.int64), "average_amount": [531.320091926459]})
    answer, _ = summarize_generic(df, "en")
    assert "1,234,567" in answer and "531.32" in answer


def test_a_small_fraction_keeps_its_precision():
    answer, _ = summarize_generic(pd.DataFrame([{"fraud_rate": 0.0197}]), "en")
    assert "0.0197" in answer


def test_indonesian_answers_are_indonesian_framed_and_still_carry_the_value():
    answer, _ = summarize_generic(pd.DataFrame([{"jumlah_fraud": 7506}]), "id")
    assert answer.startswith("Hasil analisis") and "7,506" in answer


def test_a_multi_row_result_reports_its_size_and_its_first_row():
    df = pd.DataFrame([{"channel": "online", "n": 10}, {"channel": "pos", "n": 4}])
    answer, _ = summarize_generic(df, "en")
    assert "2 rows" in answer and "channel: online" in answer and "n: 10" in answer
    assert "pos" not in answer               # only the first row, not a dump of the result


def test_a_wide_row_is_bounded():
    df = pd.DataFrame([{f"c{i}": i for i in range(20)}])
    answer, _ = summarize_generic(df, "en")
    assert "c7: 7" in answer and "c8" not in answer


def test_missing_values_are_shown_as_missing_not_as_zero_or_nan():
    answer, _ = summarize_generic(pd.DataFrame([{"average": np.nan, "label": None}]), "en")
    assert "n/a" in answer and "nan" not in answer.lower() and "None" not in answer


def test_an_empty_result_still_says_there_is_no_data():
    answer, confidence = summarize_generic(pd.DataFrame(), "en")
    assert "No data" in answer and confidence == 0.0


def test_the_generic_summary_has_modest_confidence():
    assert summarize_generic(pd.DataFrame([{"n": 1}]), "en")[1] == 0.5
