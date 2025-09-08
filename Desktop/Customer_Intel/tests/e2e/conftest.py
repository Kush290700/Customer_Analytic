from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from typing import Iterator

import pytest
from werkzeug.serving import make_server
from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

from ci_app import create_app
from ci_app.extensions import db
from ci_app.models import Role, User


PORT = int(os.getenv("E2E_PORT", "5060"))
BASE_URL = f"http://127.0.0.1:{PORT}"


@contextmanager
def run_flask_app() -> Iterator[None]:
    os.environ.setdefault("TESTING", "1")
    os.environ.setdefault("APP_ENABLE_SCHEDULER", "false")
    os.environ.setdefault("PREWARM_ON_START", "false")
    os.environ.setdefault("ENABLE_METRICS", "false")
    # File-based SQLite so server thread and test thread share DB
    inst = os.path.join(os.getcwd(), "instance")
    os.makedirs(inst, exist_ok=True)
    os.environ.setdefault("SQLALCHEMY_DATABASE_URI", f"sqlite:///{os.path.join(inst, 'e2e_tests.db')}")
    # Ensure parquet path points to the repo’s local cached file
    os.environ.setdefault("PARQUET_PATH", os.path.join(os.getcwd(), "cached_data.parquet"))

    app = create_app()
    # Seed roles and a user
    with app.app_context():
        db.create_all()
        viewer = Role.query.filter_by(name="viewer").first()
        if not viewer:
            viewer = Role(name="viewer")
            db.session.add(viewer)
        user = User.query.filter_by(email="admin@example.com").first()
        if not user:
            user = User(email="admin@example.com", name="Admin", active=True)
            user.set_password("Admin@123")
            user.roles.append(viewer)
            db.session.add(user)
        db.session.commit()

    httpd = make_server("127.0.0.1", PORT, app)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    # Give server a moment
    time.sleep(0.5)
    try:
        yield
    finally:
        httpd.shutdown()
        th.join(timeout=2)


@pytest.fixture(scope="session")
def base_url() -> str:
    return BASE_URL


@pytest.fixture(scope="session")
def browser() -> Iterator[Browser]:
    with run_flask_app():
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            yield browser
            browser.close()


@pytest.fixture()
def context(browser: Browser) -> Iterator[BrowserContext]:
    ctx = browser.new_context()
    # Start tracing for each test
    ctx.tracing.start(screenshots=True, snapshots=True, sources=True)
    yield ctx
    # Save trace for debugging if a test fails; file always saved per test for simplicity
    out_dir = os.path.join(os.getcwd(), "perf", "artifacts", "e2e")
    os.makedirs(out_dir, exist_ok=True)
    ts = str(int(time.time()))
    ctx.tracing.stop(path=os.path.join(out_dir, f"trace-{ts}.zip"))
    ctx.close()


@pytest.fixture()
def page(context: BrowserContext) -> Iterator[Page]:
    p = context.new_page()
    yield p
    p.close()
