from __future__ import annotations

import os
import time
import statistics

from ci_app import create_app


def test_metrics_endpoint_available():
    os.environ["APP_ENABLE_SCHEDULER"] = "false"
    os.environ["ENABLE_METRICS"] = "true"  # force-enable metrics for this test
    os.environ["TESTING"] = "1"
    app = create_app()
    with app.test_client() as c:
        r = c.get("/metrics")
        assert r.status_code in (200, 204)


def test_latency_budget_smoke():
    """Lightweight smoke budget that runs quickly in CI.
    Only enforces a very loose average bound to prevent extreme regressions.
    Stricter p95 gates should be run via pytest-benchmark and Locust.
    """
    os.environ.setdefault("APP_ENABLE_SCHEDULER", "false")
    os.environ.setdefault("ENABLE_METRICS", "false")
    os.environ.setdefault("TESTING", "1")
    app = create_app()
    with app.test_client() as c:
        # Use a stable, fast endpoint in test mode
        paths = ["/api/analytics"]
        for path in paths:
            samples = []
            for _ in range(5):
                t0 = time.perf_counter()
                r = c.get(path)
                assert r.status_code == 200
                samples.append((time.perf_counter() - t0) * 1000.0)
            mean_ms = statistics.mean(samples)
            assert mean_ms < 800.0, f"Mean too high for {path}: {mean_ms:.1f}ms"
