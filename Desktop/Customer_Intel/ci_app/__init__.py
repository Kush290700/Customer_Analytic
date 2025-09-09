# ci_app/__init__.py
import logging
import os
import time
import uuid
from typing import Tuple

from flask import Flask, g, request, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix

from .config import load_config
from sqlalchemy.pool import NullPool
from .extensions import (
    db,
    migrate,
    login_manager,
    limiter,
    cache,
    scheduler,
    metrics_app,
    register_metrics,
    compress,
)
from .admin import init_admin
from .blueprints.pages import pages_bp
from .blueprints.api import api_bp
from .blueprints.auth import auth_bp
from .security import add_security_headers, security_before_request, enforce_https

# Import lazily in create_app to avoid side effects at import time
# from data_loader import start_auto_refresh as start_data_refresh


def create_app() -> Flask:
    """
    Application factory — safe for multi-worker servers.
    Initializes extensions, blueprints, logging, metrics, schedulers,
    security headers, and health endpoints.
    """
    # Load .env early so config/env are in sync regardless of entrypoint
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    # Suppress noisy third-party warnings in local/dev (e.g., pkg_resources deprecation from flask_admin)
    try:
        import warnings
        warnings.filterwarnings(
            "ignore",
            message=r"pkg_resources is deprecated as an API",
            category=UserWarning,
        )
    except Exception:
        pass

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
        instance_relative_config=True,
    )
    app.config.from_mapping(load_config())
    # Test-mode hardening: isolate engine and disable background jobs
    try:
        if str(app.config.get("TESTING", False)).lower() in {"1","true","yes"}:
            # Disable scheduler entirely to avoid external DB calls and thread noise
            app.config["APP_ENABLE_SCHEDULER"] = False
            uri = str(app.config.get("SQLALCHEMY_DATABASE_URI") or "")
            if uri.startswith("sqlite:///:memory:") or uri.strip() == "sqlite://":
                opts = dict(app.config.get("SQLALCHEMY_ENGINE_OPTIONS") or {})
                opts.setdefault("poolclass", NullPool)
                app.config["SQLALCHEMY_ENGINE_OPTIONS"] = opts
    except Exception:
        pass
    # Improve static asset caching in production
    try:
        app.send_file_max_age_default = app.config.get("SEND_FILE_MAX_AGE_DEFAULT", 60 * 60 * 24 * 30)
    except Exception:
        pass

    # Normalize SQLite URI pointing at instance/ to an absolute path.
    # This prevents "unable to open database file" when CWD != project root.
    try:
        uri = str(app.config.get("SQLALCHEMY_DATABASE_URI") or "")
        if uri.startswith("sqlite:///instance/"):
            import os as _os
            filename = uri.split("sqlite:///instance/", 1)[1]
            abs_path = _os.path.join(app.instance_path, filename)
            app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{abs_path}"
    except Exception:
        pass

    # Trust proxy headers in production behind a load balancer
    if app.config.get("USE_PROXY_FIX", True):
        app.wsgi_app = ProxyFix(
            app.wsgi_app,
            x_for=1,
            x_proto=1,
            x_host=1,
            x_prefix=1,
        )

    # Configure logging ASAP (before extensions may log)
    _configure_logging(app)

    # Security: per-request nonce + optional HTTPS enforcement
    try:
        security_before_request(app)
        if app.config.get("SECURE_SSL_REDIRECT"):
            enforce_https(app)
    except Exception as e:
        app.logger.warning("Security hooks not attached: %s", e)

    # Core extensions
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    limiter.init_app(app)
    cache.init_app(app)
    # Enable gzip compression where available (optional dependency)
    try:
        if compress is not None:
            compress.init_app(app)  # type: ignore
            app.logger.info("HTTP compression enabled.")
    except Exception as e:
        app.logger.warning("Compression not enabled: %s", e)

    # Ensure instance folder exists if using instance-relative SQLite
    _ensure_sqlite_dirs(app)

    # Auto-create tables on first run for SQLite/local dev so the app works out-of-the-box.
    # In production with real DBs, prefer migrations.
    _maybe_create_tables(app)

    # Background scheduler (for emails, digest jobs, etc.)
    # Only start in a single process to avoid duplicate jobs.
    if _should_start_jobs(app):
        try:
            scheduler.init_app(app)
            # Register cron jobs before starting the scheduler
            try:
                from .tasks import register_scheduled_jobs  # local import
                register_scheduled_jobs(app)
            except Exception as e:
                app.logger.exception("Failed to register scheduled jobs: %s", e)
            scheduler.start()
            app.logger.info("APScheduler started in this process.")
        except Exception as e:
            app.logger.exception("Failed to start APScheduler: %s", e)

        # Start the data refresh scheduler (parquet auto-refresh)
        try:
            from ci_app.data_loader import start_auto_refresh as start_data_refresh  # local import
            start_data_refresh()  # idempotent in our loader; guarded internally
            app.logger.info("Data-loader auto-refresh enabled.")
        except Exception as e:
            app.logger.exception("Failed to start data-loader auto-refresh: %s", e)
    else:
        app.logger.info("Skipping schedulers in this process (worker/setting guard).")

    # Blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(pages_bp)
    # Register Region page blueprint (separate from pages_bp)
    try:
        from .blueprints.pages.routes_region import bp as region_bp
        app.register_blueprint(region_bp)
    except Exception as e:
        app.logger.warning("Region blueprint not registered: %s", e)
    app.register_blueprint(api_bp, url_prefix="/api")

    # Admin (RBAC management, etc.)
    init_admin(app)

    # Observability: Prometheus exporter preferred, fallback to built-in wrappers
    try:
        from .observability import setup_prometheus_exporter
        setup_prometheus_exporter(app)
    except Exception as e:
        app.logger.info("Prometheus exporter not attached: %s", e)
        try:
            register_metrics(app)
            app.wsgi_app = metrics_app(app.wsgi_app)
        except Exception as e2:
            app.logger.warning("Metrics middleware not attached: %s", e2)
        # Ensure a simple /metrics route exists for local/testing when exporter is unavailable
        def _has_metrics_route() -> bool:
            try:
                for r in app.url_map.iter_rules():
                    if str(getattr(r, "rule", "")) == "/metrics":
                        return True
            except Exception:
                pass
            return False

        if not _has_metrics_route():
            @app.get("/metrics")
            def _metrics_fallback():  # type: ignore
                try:
                    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest  # type: ignore
                    data = generate_latest()
                    mimetype = CONTENT_TYPE_LATEST
                    status = 200
                except Exception:
                    data = b""
                    mimetype = "text/plain"
                    status = 204
                return app.response_class(response=data, status=status, mimetype=mimetype)

    # Ensure minimal /metrics route exists in local/testing when exporter not attached
    try:
        if app.config.get("ENABLE_METRICS", True):
            def _has_metrics_route() -> bool:
                try:
                    for r in app.url_map.iter_rules():
                        if str(getattr(r, "rule", "")) == "/metrics":
                            return True
                except Exception:
                    pass
                return False
            if not _has_metrics_route():
                @app.get("/metrics")
                def _metrics_min():  # type: ignore
                    try:
                        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest  # type: ignore
                        data = generate_latest()
                        return app.response_class(response=data, status=200, mimetype=CONTENT_TYPE_LATEST)
                    except Exception:
                        return ("", 204)
    except Exception:
        pass

    # Jinja/JSON niceties
    app.jinja_env.trim_blocks = True
    app.jinja_env.lstrip_blocks = True
    app.json.sort_keys = False

    # Safe date formatting filter for templates (handles pandas NaT/None/strings)
    try:
        from .services.utils import (
            fmt_date_or_none as _fmt_date_or_none,
            fmt_datetime_or_none as _fmt_datetime_or_none,
        )

        def _fmt_date(value, fmt="%Y-%m-%d"):
            try:
                s = _fmt_date_or_none(value, fmt)
                return s if s is not None else "N/A"
            except Exception:
                return "N/A"

        app.jinja_env.filters["fmt_date"] = _fmt_date
        
        def _fmt_datetime(value, fmt="%Y-%m-%d %H:%M"):
            try:
                s = _fmt_datetime_or_none(value, fmt)
                return s if s is not None else "N/A"
            except Exception:
                return "N/A"

        app.jinja_env.filters["fmt_datetime"] = _fmt_datetime
    except Exception as _e:
        app.logger.warning("Failed to register fmt_date filter: %s", _e)

    # Per-request context: request id + timer
    @app.before_request
    def _inject_request_id():
        g.request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
        g._start_ts = time.perf_counter()

        # Security headers + request id echo + basic access log
    @app.after_request
    def _secure_and_log(resp):
        # Security headers
        resp = add_security_headers(resp)
        # Correlation headers
        if getattr(g, "request_id", None):
            resp.headers["X-Request-Id"] = g.request_id
        # Caching strategy: aggressive for static, conservative elsewhere
        try:
            if request.endpoint == 'static' or (request.path or '').startswith('/static/'):
                resp.headers["Cache-Control"] = resp.headers.get("Cache-Control") or "public, max-age=31536000, immutable"
            else:
                # Default for dynamic pages/APIs unless explicitly overridden elsewhere
                resp.headers.setdefault("Cache-Control", "no-store")
        except Exception:
            resp.headers.setdefault("Cache-Control", "no-store")
        # Basic access timing
        try:
            elapsed_ms = int((time.perf_counter() - getattr(g, "_start_ts", time.perf_counter())) * 1000)
            request.environ["ci.elapsed_ms"] = elapsed_ms  # for downstream if needed
            thr = int(app.config.get("SLOW_REQUEST_MS", 800))
            if elapsed_ms >= thr:
                app.logger.warning("Slow request: %s %s %dms", request.method, request.path, elapsed_ms)
        except Exception:
            pass
        return resp

    # Health / readiness / liveness
    @app.get("/healthz")
    def healthz():
        return {"ok": True}, 200

    @app.get("/livez")
    def livez():
        return {"alive": True}, 200

    @app.get("/readyz")
    def readyz():
        try:
            with db.engine.connect() as conn:
                conn.execute(db.text("SELECT 1"))
            cache_ok = True
        except Exception:
            cache_ok = False
        # If prewarm is required for readiness, ensure snapshots exist
        prewarm_required = bool(app.config.get("PREWARM_REQUIRED", False))
        prewarm_ok = True
        if prewarm_required:
            try:
                from .snapshot_cache import has_snapshot as _has_snap
                prewarm_ok = _has_snap("analytics") and _has_snap("clv_dashboard") and _has_snap("product_insights")
            except Exception:
                prewarm_ok = False
        ok = cache_ok and (prewarm_ok or not prewarm_required)
        return jsonify({"db": True, "cache": cache_ok, "prewarm": prewarm_ok}), 200 if ok else 503

    # Optional synchronous prewarm on cold start
    try:
        if False and app.config.get("PREWARM_ON_START"):
            app.logger.info("Prewarming caches and snapshots on startup…")
            from .warmers import warm_all as _warm_all
            _warm_all(app, horizons=app.config.get("PREWARM_HORIZONS", [1,3,5]))
            app.logger.info("Prewarm complete.")
    except Exception as e:
        app.logger.exception("Prewarm failed: %s", e)

    # Error handlers (JSON for API prefix; HTML otherwise)
    @app.errorhandler(404)
    def _not_found(e):
        if request.path.startswith("/api/"):
            return (
                jsonify(
                    {
                        "code": "not_found",
                        "message": "Not Found",
                        "details": {
                            "request_id": getattr(g, "request_id", ""),
                            "path": request.path,
                        },
                    }
                ),
                404,
            )
        try:
            from flask import render_template
            return render_template("error.html", message="Page not found."), 404
        except Exception:
            return ("Not Found", 404)

    @app.errorhandler(500)
    def _server_error(e):
        app.logger.exception("Unhandled error: %s", e, extra={"request_id": getattr(g, "request_id", "-")})
        if request.path.startswith("/api/"):
            return (
                jsonify(
                    {
                        "code": "internal_error",
                        "message": "Internal Server Error",
                        "details": {"request_id": getattr(g, "request_id", "")},
                    }
                ),
                500,
            )
        # Fallback to a template if you have one
        try:
            from flask import render_template
            return render_template("error.html", message="An unexpected error occurred."), 500
        except Exception:
            return ("Internal Server Error", 500)

    # Serve a favicon to avoid noisy 404s in logs (define before any prewarm/test-client calls)
    @app.get("/favicon.ico")
    def _favicon():
        try:
            from flask import redirect, url_for
            # Prefer SVG (small, crisp). You can drop a favicon.ico into /static if desired.
            return redirect(url_for("static", filename="favicon.svg"), code=302)
        except Exception:
            return ("", 204)

    # Optional synchronous prewarm on cold start (skip during tests)
    try:
        is_testing = str(app.config.get("TESTING", False)).lower() in {"1","true","yes"}
        if app.config.get("PREWARM_ON_START") and not is_testing:
            app.logger.info("Prewarming caches and snapshots on startup�?�")
            from .warmers import warm_all as _warm_all
            _warm_all(app, horizons=app.config.get("PREWARM_HORIZONS", [1,3,5]))
            app.logger.info("Prewarm complete.")
    except Exception as e:
        app.logger.exception("Prewarm failed: %s", e)

    # Optional Sentry + OpenTelemetry
    try:
        from .observability import setup_sentry, setup_otel
        setup_sentry(app)
        setup_otel(app)
    except Exception as e:
        app.logger.warning("Observability (Sentry/OTEL) not fully enabled: %s", e)

    # Developer CLI: local warmup of key endpoints
    @app.cli.command("warmup")
    def warmup_cmd():
        """Pre-warm common analytics bundles for snappy local UX."""
        try:
            from .warmers import warm_all as _warm_all
            _warm_all(app)
            print("Warmup complete.")
        except Exception as e:  # pragma: no cover
            print(f"Warmup failed: {e}")

    return app


# ---------------------------- helpers --------------------------------- #

def _configure_logging(app: Flask) -> None:
    """Structured logging. Avoid Unicode issues on Windows terminals by stripping non-ASCII."""
    level_name = app.config.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    # If under Gunicorn, reuse its handlers; else add our own stream handler.
    gunicorn_error = logging.getLogger("gunicorn.error")
    root = logging.getLogger()

    class AsciiFormatter(logging.Formatter):
        def format(self, record):
            s = super().format(record)
            try:
                return s.encode("ascii", errors="ignore").decode("ascii", errors="ignore")
            except Exception:
                return s

    fmt = AsciiFormatter("%(asctime)s %(levelname)s %(name)s req_id=%(request_id)s %(message)s")

    class RequestIdFilter(logging.Filter):
        def filter(self, record):
            # attach request_id if available
            try:
                from flask import g  # local import to avoid app context issues at import time
                record.request_id = getattr(g, "request_id", "-")
            except Exception:
                record.request_id = "-"
            return True

    # Clean existing handlers only if we're not under Gunicorn
    if gunicorn_error.handlers:
        for h in gunicorn_error.handlers:
            h.addFilter(RequestIdFilter())
        root.handlers = []
        root.setLevel(level)
        app.logger.handlers = []
        app.logger.setLevel(level)
        app.logger.propagate = True
    else:
        if not app.logger.handlers:
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            sh.addFilter(RequestIdFilter())
            app.logger.addHandler(sh)
        app.logger.setLevel(level)
        app.logger.propagate = False


def _should_start_jobs(app: Flask) -> bool:
    """
    Decide if this process should run background jobs.
    Rules:
      - If APP_ENABLE_SCHEDULER=false → never.
      - In Flask dev reloader: only when WERKZEUG_RUN_MAIN == "true".
      - In Gunicorn: only when GUNICORN_WORKER_ID == "1" (or unset) AND RUN_JOBS truthy.
    """
    if str(app.config.get("APP_ENABLE_SCHEDULER", True)).lower() not in {"1", "true", "yes"}:
        return False

    run_jobs_env = os.getenv("RUN_JOBS", "1").lower() in {"1", "true", "yes"}

    is_debug = app.config.get("DEBUG", False)
    reloader_ok = os.environ.get("WERKZEUG_RUN_MAIN") in ("true", "True", "1", None)
    gunicorn_worker_id = os.getenv("GUNICORN_WORKER_ID")

    if "gunicorn" in os.getenv("SERVER_SOFTWARE", "").lower():
        return run_jobs_env and (gunicorn_worker_id in (None, "1"))

    # Werkzeug dev server: start once (only in the main reloader process)
    if is_debug:
        return run_jobs_env and (os.environ.get("WERKZEUG_RUN_MAIN") == "true")

    # Generic fallback (single-process servers)
    return run_jobs_env


def _ensure_sqlite_dirs(app: Flask) -> None:
    """Create instance directory if SQLALCHEMY_DATABASE_URI points into it."""
    try:
        uri = str(app.config.get("SQLALCHEMY_DATABASE_URI") or "")
        if uri.startswith("sqlite:///") and "/instance/" in uri.replace("\\", "/"):
            # Ensure the instance folder exists
            import os as _os
            _os.makedirs(app.instance_path, exist_ok=True)
    except Exception:
        pass


def _maybe_create_tables(app: Flask) -> None:
    """Create DB tables automatically for SQLite/local dev if empty/missing.

    This avoids first-run failures when no migrations were executed.
    Set AUTO_CREATE_DB=false to disable.
    """
    try:
        if str(app.config.get("AUTO_CREATE_DB", True)).lower() not in {"1","true","yes","on"}:
            return

        uri = str(app.config.get("SQLALCHEMY_DATABASE_URI") or "")
        is_sqlite = uri.startswith("sqlite:")
        if not is_sqlite:
            return

        with app.app_context():
            # Create all tables if none exist
            from sqlalchemy import inspect as _inspect
            insp = _inspect(db.engine)
            tables = insp.get_table_names()
            if not tables:
                app.logger.info("Initializing SQLite schema via create_all() …")
                db.create_all()
    except Exception as e:
        app.logger.warning("Auto-create tables skipped/failed: %s", e)
