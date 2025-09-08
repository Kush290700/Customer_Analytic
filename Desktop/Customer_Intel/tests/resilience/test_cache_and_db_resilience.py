from __future__ import annotations

import concurrent.futures as futures
import os
from typing import Callable

import pytest

from ci_app import create_app


def _mk_app_env() -> None:
    os.environ.setdefault("TESTING", "1")
    os.environ.setdefault("APP_ENABLE_SCHEDULER", "false")
    os.environ.setdefault("ENABLE_METRICS", "false")
    os.environ.setdefault("PARQUET_PATH", os.path.join(os.getcwd(), "cached_data.parquet"))


@pytest.mark.resilience
def test_cache_unavailable_falls_back_to_simplecache(monkeypatch):
    _mk_app_env()
    # Point REDIS_URL to an unreachable endpoint so RedisCache would fail if enforced
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:6399/0")
    # Ensure limiter also tolerates missing Redis
    monkeypatch.delenv("RATELIMIT_STORAGE_URL", raising=False)
    app = create_app()
    client = app.test_client()
    # App should serve even if Redis is not reachable (SimpleCache fallback in config)
    rv = client.get("/api/analytics")
    assert rv.status_code == 200


@pytest.mark.resilience
def test_api_json_error_envelope_on_internal_error(monkeypatch):
    _mk_app_env()
    # Force an exception inside analytics route by patching apply_filters
    import ci_app.blueprints.api.routes_basic as rb

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(rb, "apply_filters", boom)
    app = create_app()
    client = app.test_client()
    rv = client.get("/api/analytics")
    # The app.errorhandler(500) should return JSON envelope for /api/* paths
    assert rv.status_code == 500
    data = rv.get_json()
    assert isinstance(data, dict)
    assert data.get("error") in {"Internal Server Error", "Internal server error", "Internal Server Error"}


@pytest.mark.resilience
def test_concurrent_requests_do_not_race_or_error():
    _mk_app_env()
    app = create_app()
    client = app.test_client()

    def hit() -> int:
        resp = client.get("/api/analytics")
        assert resp.status_code == 200
        return resp.status_code

    N = 8
    with futures.ThreadPoolExecutor(max_workers=N) as ex:
        codes = list(ex.map(lambda _: hit(), range(N)))
    assert all(c == 200 for c in codes)

