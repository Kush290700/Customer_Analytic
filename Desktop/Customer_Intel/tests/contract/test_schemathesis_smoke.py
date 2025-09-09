from __future__ import annotations

import os
from typing import Any, Dict

import pytest

from ci_app import create_app


def _load_spec_dict() -> Dict[str, Any]:
    yaml = pytest.importorskip("yaml")
    with open(os.path.join(os.getcwd(), "api", "openapi.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.mark.contract
def test_schemathesis_smoke_short_run():
    # Configure app in testing mode, no schedulers
    os.environ.setdefault("TESTING", "1")
    os.environ.setdefault("APP_ENABLE_SCHEDULER", "false")
    os.environ.setdefault("ENABLE_METRICS", "false")
    os.environ.setdefault("PARQUET_PATH", os.path.join(os.getcwd(), "cached_data.parquet"))
    app = create_app()

    schemathesis = pytest.importorskip("schemathesis")
    spec = _load_spec_dict()
    schema = schemathesis.from_wsgi(spec, app)

    # Limit to a few critical endpoints to keep it fast locally
    # and limit examples to avoid long fuzzing sessions
    for endpoint in schema:
        if endpoint.path in {"/api/analytics", "/api/kpis", "/api/filters_meta"}:
            # Run a small number of generated tests
            endpoint.add_examples(2)
            for result in endpoint.get_strategies().execute():
                # Any error or schema violation will raise inside execute()
                assert result.has_failures is False
