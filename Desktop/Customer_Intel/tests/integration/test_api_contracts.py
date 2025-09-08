from __future__ import annotations

import os
import time

import pytest

from ci_app import create_app


def _mk_app_env() -> None:
    os.environ.setdefault("TESTING", "1")
    os.environ.setdefault("APP_ENABLE_SCHEDULER", "false")
    os.environ.setdefault("ENABLE_METRICS", "false")
    os.environ.setdefault("PARQUET_PATH", os.path.join(os.getcwd(), "cached_data.parquet"))


@pytest.mark.contract
def test_caching_etag_and_cache_control_present():
    _mk_app_env()
    app = create_app()
    client = app.test_client()

    r1 = client.get("/api/analytics")
    assert r1.status_code == 200
    etag1 = r1.headers.get("ETag")
    cc1 = r1.headers.get("Cache-Control", "")
    assert etag1, "ETag must be present"
    assert "max-age" in cc1

    # Subsequent call should be fast and return same ETag for identical inputs
    t0 = time.perf_counter()
    r2 = client.get("/api/analytics")
    dt = time.perf_counter() - t0
    assert r2.status_code == 200
    assert r2.headers.get("ETag") == etag1
    # Budget sanity: should be well under 0.4s in tests
    assert dt < 0.4

