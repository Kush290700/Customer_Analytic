from __future__ import annotations

import os

import pandas as pd
import pytest

from ci_app import create_app
from tests.fixtures.data_factories import make_orders_df


@pytest.mark.contract
def test_analytics_with_seeded_orders(monkeypatch):
    os.environ.setdefault("TESTING", "1")
    os.environ.setdefault("APP_ENABLE_SCHEDULER", "false")
    os.environ.setdefault("ENABLE_METRICS", "false")

    df = make_orders_df(days=30, seed=123)

    # Patch data access to return synthetic DF filtered by dates
    import ci_app.services.data_access as dao

    def _get_dataframe(start: str, end: str) -> pd.DataFrame:
        s = pd.to_datetime(start)
        e = pd.to_datetime(end)
        mask = (df["Date"] >= s) & (df["Date"] <= e)
        return df.loc[mask].copy()

    monkeypatch.setattr(dao, "get_dataframe", _get_dataframe)

    app = create_app()
    c = app.test_client()
    r = c.get("/api/analytics")
    assert r.status_code == 200
    data = r.get_json()
    assert isinstance(data, dict)
    # Realistic synthetic payloads include keys even if small
    assert "timeline" in data and isinstance(data["timeline"], dict)

