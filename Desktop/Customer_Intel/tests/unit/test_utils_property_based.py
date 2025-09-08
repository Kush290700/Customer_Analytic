from __future__ import annotations

import math

import pandas as pd
import pytest
from hypothesis import given, strategies as st

from ci_app.services.analytics import _safe_div, _complete_period_mask
from ci_app.services.utils import fmt_date_or_none


@given(st.floats(allow_nan=False, allow_infinity=False), st.floats(allow_nan=False, allow_infinity=False))
def test_safe_div_never_raises(a: float, b: float):
    out = _safe_div(a, b)
    assert isinstance(out, float)
    if b != 0:
        assert math.isfinite(out)
    else:
        assert out == 0.0


@given(st.dates(min_value=pd.Timestamp(2018, 1, 1).date(), max_value=pd.Timestamp(2026, 12, 31).date()))
def test_fmt_date_or_none_round_trip(d):
    s = fmt_date_or_none(d, "%Y-%m-%d")
    assert s is not None and len(s) == 10


@pytest.mark.parametrize("freq", ["D", "W", "MS", "QS", "YS"]) 
def test_complete_period_mask_monotonic(freq: str):
    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    mask = _complete_period_mask(pd.Series(dates), freq=freq, now=pd.Timestamp("2024-05-15"))
    assert mask.dtype == bool
    # Once False begins, it should remain False (only drop the tail)
    seen_false = False
    for m in mask.tolist():
        if seen_false:
            assert m is False
        if m is False:
            seen_false = True

