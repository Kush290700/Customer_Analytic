from __future__ import annotations

import os
import pytest

from ci_app import create_app


@pytest.mark.contract
def test_api_not_found_error_envelope_shape():
    os.environ.setdefault("TESTING", "1")
    os.environ.setdefault("APP_ENABLE_SCHEDULER", "false")
    os.environ.setdefault("ENABLE_METRICS", "false")
    app = create_app()
    c = app.test_client()
    r = c.get("/api/does_not_exist")
    assert r.status_code == 404
    data = r.get_json()
    assert isinstance(data, dict)
    assert set(["code", "message"]).issubset(data.keys())
    assert data["code"] == "not_found"
