from __future__ import annotations

import os
import unittest
from contextlib import contextmanager

from ci_app import create_app
from ci_app.extensions import db
from ci_app.models import User, Role


@contextmanager
def app_client():
    # Keep schedulers/metrics off for tests; use sqlite memory DB
    os.environ["APP_ENABLE_SCHEDULER"] = "false"
    os.environ["ENABLE_METRICS"] = "false"
    os.environ["TESTING"] = "1"
    # Force in-memory DB to isolate tests even if .env provides a file DB
    os.environ["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    # Ensure the app points to the local parquet cache
    os.environ.setdefault("PARQUET_PATH", os.path.join(os.getcwd(), "cached_data.parquet"))

    app = create_app()
    app.config.update(TESTING=True, WTF_CSRF_ENABLED=False)
    with app.app_context():
        db.create_all()
        # Seed roles and a user
        viewer = Role(name="viewer")
        db.session.add(viewer)
        u = User(email="tester@example.com", name="Tester", active=True)
        u.set_password("Passw0rd!")
        u.roles.append(viewer)
        db.session.add(u)
        db.session.commit()
        with app.test_client() as client:
            yield app, client


class TestAppFeatures(unittest.TestCase):
    def _login(self, client):
        resp = client.post(
            "/login",
            data={"email": "tester@example.com", "password": "Passw0rd!"},
            follow_redirects=True,
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Dashboard", resp.data)

    def test_pages_render(self):
        with app_client() as (app, client):
            self._login(client)
            # Main pages should render
            for path in [
                "/",
                "/kpis",
                "/rfm",
                "/cohort",
                "/clv",
                "/analytics",
                "/recommendations",
                "/drilldown",
                "/product_drilldown",
                "/download",
            ]:
                r = client.get(path)
                self.assertIn(r.status_code, (200, 204), msg=f"GET {path} => {r.status_code}")

    def test_api_endpoints(self):
        with app_client() as (app, client):
            self._login(client)
            # Core analytics APIs should respond with JSON (may be empty if no data)
            for path in [
                "/api/kpis",
                "/api/rfm",
                "/api/cohort",
                "/api/clv",
                "/api/analytics",
                "/api/recommendations",
            ]:
                r = client.get(path)
                self.assertEqual(r.status_code, 200, msg=f"GET {path} => {r.status_code}")
                # All are JSON responses
                self.assertIn("application/json", r.headers.get("Content-Type", ""))

    def test_filters_apply_and_clear(self):
        with app_client() as (app, client):
            self._login(client)
            # Apply a JSON filter payload (empty selections)
            r = client.post(
                "/set_filters",
                json={
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31",
                    "regions": [],
                    "methods": [],
                    "customers": [],
                },
                headers={"X-Requested-With": "fetch"},
            )
            self.assertEqual(r.status_code, 200)
            self.assertTrue(r.is_json)
            # Subsequent API calls should still work after filter change
            r2 = client.get("/api/kpis")
            self.assertEqual(r2.status_code, 200)

            # Clear via form POST on index
            r3 = client.post("/", data={"action": "clear_filters"}, follow_redirects=True)
            self.assertEqual(r3.status_code, 200)
            # Verify session reflects cleared lists
            with client.session_transaction() as sess:
                self.assertEqual(sess.get("regions"), [])
                self.assertEqual(sess.get("methods"), [])
                self.assertEqual(sess.get("customers"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
