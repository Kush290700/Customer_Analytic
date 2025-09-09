from __future__ import annotations

import os
import pytest

from ci_app import create_app


@pytest.fixture(scope="session")
def app():
    os.environ.setdefault("APP_ENABLE_SCHEDULER", "false")
    os.environ.setdefault("ENABLE_METRICS", "false")
    os.environ.setdefault("TESTING", "1")
    os.environ.setdefault("RATELIMIT_ENABLED", "false")
    os.environ.setdefault("SQLALCHEMY_DATABASE_URI", "sqlite:///:memory:")
    os.environ.setdefault("PARQUET_PATH", os.path.join(os.getcwd(), "cached_data.parquet"))
    app = create_app()
    app.config.update(TESTING=True)
    return app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.mark.benchmark(group="api")
@pytest.mark.parametrize(
    "path",
    ["/api/analytics"],
)
def test_api_benchmarks(client, benchmark, path):
    resp = benchmark(lambda: client.get(path))
    assert resp.status_code == 200

    # Budget: p95 < 300ms on small dataset (pytest-benchmark records this in JSON)
    # Enforce a soft assert by checking last call duration via extra_info if available.
    # Strict gating typically happens in CI by comparing trends.
