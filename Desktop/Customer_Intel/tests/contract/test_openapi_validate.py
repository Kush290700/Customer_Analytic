from __future__ import annotations

import json
import os
from typing import Any, Dict

import pytest
import yaml
from jsonschema import validate

from ci_app import create_app


def _load_spec() -> Dict[str, Any]:
    with open(os.path.join(os.getcwd(), "api", "openapi.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _schema_for(spec: Dict[str, Any], path: str, method: str = "get", status: str = "200") -> Dict[str, Any] | None:
    try:
        return (
            spec["paths"][path][method]["responses"][status]["content"]["application/json"]["schema"]
        )
    except Exception:
        return None


@pytest.mark.contract
def test_openapi_exists_and_parses():
    spec = _load_spec()
    assert spec["openapi"].startswith("3."), "OpenAPI 3.x required"
    assert "/api/analytics" in spec["paths"], "/api/analytics missing from spec"


@pytest.mark.contract
def test_responses_conform_to_openapi_schemas():
    os.environ.setdefault("TESTING", "1")
    os.environ.setdefault("APP_ENABLE_SCHEDULER", "false")
    os.environ.setdefault("ENABLE_METRICS", "false")
    os.environ.setdefault("PARQUET_PATH", os.path.join(os.getcwd(), "cached_data.parquet"))
    app = create_app()
    client = app.test_client()

    spec = _load_spec()

    endpoints = [
        "/api/filters_meta",
        "/api/kpis?freq=MS",
        "/api/analytics",
        "/api/clv_dashboard?h=3&method=heuristic&topn=10",
        "/api/product_insights?days=60&topn=10&target_margin=18",
        "/api/forecast",
    ]
    for ep in endpoints:
        rv = client.get(ep)
        assert rv.status_code == 200, f"GET {ep} -> {rv.status_code}"
        path = ep.split("?")[0]
        schema = _schema_for(spec, path)
        # Only validate where a schema is defined in our spec
        if schema:
            payload = rv.get_json()
            assert isinstance(payload, dict), f"Response for {ep} must be JSON object"
            # Perform a best-effort JSON schema validation
            validate(instance=payload, schema=schema)  # raises on failure

