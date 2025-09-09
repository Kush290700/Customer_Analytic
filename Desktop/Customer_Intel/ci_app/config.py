# ci_app/config.py
from __future__ import annotations

import os
from datetime import timedelta
from typing import Dict, Any


def _bool(val: str | None, default: bool = False) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _int(val: str | None, default: int) -> int:
    try:
        return int(str(val).strip())
    except Exception:
        return default


def _list(val: str | None, sep: str = ",") -> list[str]:
    if not val:
        return []
    return [x.strip() for x in val.split(sep) if x.strip()]


def load_config() -> Dict[str, Any]:
    """
    Build a single dict for app.config with safe, production-ready defaults.
    NOTE: data_loader.py reads environment variables directly. We mirror the most
    important ones here for visibility and also set sensible env defaults so both
    modules agree at runtime.
    """
    env = os.getenv("FLASK_ENV", "production")
    debug = _bool(os.getenv("DEBUG"), env == "development")

    # ---------- Application DB (Flask models: users/roles/etc.) ----------
    # Priority: explicit SQLALCHEMY_DATABASE_URI -> DATABASE_URL -> fallback SQLite file
    database_url = os.getenv("SQLALCHEMY_DATABASE_URI") or os.getenv("DATABASE_URL")
    if not database_url:
        # Default to a file sqlite if no external DB is provided
        db_path = os.getenv("SQLITE_PATH", os.path.join(os.getcwd(), "ci_app.db"))
        database_url = f"sqlite:///{db_path}"

    # ---------- Redis (optional) ----------
    redis_url = os.getenv("REDIS_URL") or os.getenv("CACHE_REDIS_URL") or ""

    # Flask-Caching config
    if redis_url:
        # If Redis URL is set but redis client is not installed, gracefully fall back
        try:
            import redis as _redis  # type: ignore  # noqa: F401
            cache_cfg = {
                "CACHE_TYPE": "RedisCache",
                "CACHE_REDIS_URL": redis_url,
                "CACHE_DEFAULT_TIMEOUT": _int(os.getenv("CACHE_DEFAULT_TIMEOUT"), 300),
            }
        except Exception:
            cache_cfg = {
                "CACHE_TYPE": "SimpleCache",
                "CACHE_DEFAULT_TIMEOUT": _int(os.getenv("CACHE_DEFAULT_TIMEOUT"), 300),
            }
    else:
        cache_cfg = {
            "CACHE_TYPE": "SimpleCache",
            "CACHE_DEFAULT_TIMEOUT": _int(os.getenv("CACHE_DEFAULT_TIMEOUT"), 300),
        }

    # Flask-Limiter storage (prefer Redis, fall back to memory for single-instance)
    # Accept both *URL and *URI env names
    limiter_storage = (
        os.getenv("RATELIMIT_STORAGE_URL")
        or os.getenv("RATELIMIT_STORAGE_URI")
        or (redis_url or "memory://")
    )

    # ---------- Build config ----------
    cfg: Dict[str, Any] = dict(
        # Core
        ENV=env,
        DEBUG=debug,
        TESTING=_bool(os.getenv("TESTING"), False),
        SECRET_KEY=os.getenv("FLASK_SECRET_KEY") or os.getenv("SECRET_KEY", "PleaseChangeMe"),
        TEMPLATES_AUTO_RELOAD=bool(debug),

        # SQLAlchemy
        SQLALCHEMY_DATABASE_URI=database_url,
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        SQLALCHEMY_ENGINE_OPTIONS={
            "pool_pre_ping": True,
            "pool_recycle": _int(os.getenv("SQL_POOL_RECYCLE"), 300),
        },

        # Sessions / Cookies
        SESSION_COOKIE_SECURE=_bool(os.getenv("SESSION_COOKIE_SECURE"), True),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE=os.getenv("SESSION_COOKIE_SAMESITE", "Lax"),
        REMEMBER_COOKIE_SECURE=True,
        REMEMBER_COOKIE_SAMESITE="Lax",
        PERMANENT_SESSION_LIFETIME=timedelta(days=_int(os.getenv("SESSION_DAYS"), 7)),

        # Rate limits
        # Accept RATE_LIMITS_DEFAULT for compatibility
        RATELIMIT_DEFAULT=(
            os.getenv("RATELIMIT_DEFAULT")
            or os.getenv("RATE_LIMITS_DEFAULT")
            or "200/day;50/hour"
        ),
        RATELIMIT_STORAGE_URL=limiter_storage,
        RATELIMIT_ENABLED=_bool(os.getenv("RATELIMIT_ENABLED"), True),

        # Cache
        **cache_cfg,

        # Scheduler (APScheduler)
        APP_ENABLE_SCHEDULER=_bool(os.getenv("APP_ENABLE_SCHEDULER"), True),
        SCHEDULER_API_ENABLED=False,
        SCHEDULER_TIMEZONE=os.getenv("TZ", "UTC"),

        # Email (SMTP)
        SMTP_SERVER=os.getenv("SMTP_SERVER", "tworiversmeats-com.mail.protection.outlook.com"),
        SMTP_PORT=_int(os.getenv("SMTP_PORT", "25"), 25),
        SMTP_USER=os.getenv("SMTP_USER", ""),
        SMTP_PASS=os.getenv("SMTP_PASS", ""),
        MAIL_FROM=os.getenv("MAIL_FROM", "Customer Intel <no-reply@tworiversmeats.com>"),
        MAIL_USE_TLS=_bool(os.getenv("MAIL_USE_TLS"), True),

        # Distribution lists
        KPI_RECIPIENTS=_list(os.getenv("KPI_RECIPIENTS")),
        CHURN_ALERT_RECIPIENTS=_list(os.getenv("CHURN_ALERT_RECIPIENTS")),
        REGION_REPORT_RECIPIENTS=_list(os.getenv("REGION_REPORT_RECIPIENTS")),

        # Feature flags
        ENABLE_ADMIN=_bool(os.getenv("ENABLE_ADMIN"), True),
        ENABLE_METRICS=_bool(os.getenv("ENABLE_METRICS"), True),
        ENABLE_AI_SQL=_bool(os.getenv("ENABLE_AI_SQL"), False),

        # Security headers toggles (used by ci_app.security.add_security_headers)
        CSP_ENABLE=_bool(os.getenv("CSP_ENABLE"), True),
        HSTS_ENABLE=_bool(os.getenv("HSTS_ENABLE"), True),
        # CSP defaults: include CDNs used by templates
        # Allow opting into 'unsafe-eval' for libraries (e.g., CDN Tailwind / Alpine non-CSP) when needed.
        # Default True to keep the UI functional with CDN assets; set to false if you fully remove CDN eval usage.
        CSP_ALLOW_UNSAFE_EVAL=_bool(os.getenv("CSP_ALLOW_UNSAFE_EVAL"), True),
        CSP_ALLOW_UNSAFE_INLINE=_bool(os.getenv("CSP_ALLOW_UNSAFE_INLINE"), True),
        # Upgrade insecure requests only outside of debug by default
        CSP_UPGRADE_INSECURE_REQUESTS=_bool(os.getenv("CSP_UPGRADE_INSECURE_REQUESTS"), not debug),
        CSP_SCRIPT_SRC=os.getenv("CSP_SCRIPT_SRC") or [
            "'self'",
            "https://cdn.tailwindcss.com",
            "https://cdn.jsdelivr.net",
            "https://unpkg.com",
            "https://fonts.googleapis.com",
            "https://fonts.gstatic.com",
        ],
        CSP_STYLE_SRC=os.getenv("CSP_STYLE_SRC") or [
            "'self'",
            "https://cdn.jsdelivr.net",
            "https://unpkg.com",
            "https://fonts.googleapis.com",
        ],
        CSP_FONT_SRC=os.getenv("CSP_FONT_SRC") or [
            "'self'",
            "https://fonts.gstatic.com",
            "data:",
        ],
        CSP_IMG_SRC=os.getenv("CSP_IMG_SRC") or [
            "'self'",
            "data:",
        ],
        CSP_CONNECT_SRC=os.getenv("CSP_CONNECT_SRC") or [
            "'self'",
        ],

        # Logging
        LOG_LEVEL=os.getenv("LOG_LEVEL", "INFO"),
        LOG_JSON=_bool(os.getenv("LOG_JSON"), False),

        # Parquet/cache for analytics (the app also reads this path)
        # Default to a common local filename if none provided
        PARQUET_PATH=os.getenv("PARQUET_PATH", os.path.join(os.getcwd(), "cached_data.parquet")),

        # Static/Template
        SEND_FILE_MAX_AGE_DEFAULT=(0 if debug else 60 * 60 * 24 * 365),

        # UI defaults
        DASHBOARD_PER_PAGE=_int(os.getenv("DASHBOARD_PER_PAGE", "50"), 50),
        # Performance toggles
        FAST_PAGES_SKIP_BASEDF=_bool(os.getenv("FAST_PAGES_SKIP_BASEDF"), True),
        FAST_INLINE_BUNDLES=_bool(os.getenv("FAST_INLINE_BUNDLES"), False),
        # Prewarm controls
        # Default to prewarm on start so graphs feel instant after boot
        PREWARM_ON_START=_bool(os.getenv("PREWARM_ON_START"), True),
        PREWARM_HORIZONS=[int(x) for x in _list(os.getenv("PREWARM_HORIZONS", "1,3,5")) or [1,3,5]],
        PREWARM_REQUIRED=_bool(os.getenv("PREWARM_REQUIRED"), False),
        # Snapshot cache backend: prefer memory to save disk space unless explicitly enabled
        SNAPSHOT_TO_DISK=_bool(os.getenv("SNAPSHOT_TO_DISK"), False),
        # Slow request logging threshold (ms)
        SLOW_REQUEST_MS=_int(os.getenv("SLOW_REQUEST_MS", "800"), 800),

        # CLV defaults
        CLV_HORIZON_YEARS=float(os.getenv("CLV_HORIZON_YEARS", "3")),
        # Lower default for CLV bubble cap to reduce JSON and compute
        MAX_PLOT_POINTS=_int(os.getenv("MAX_PLOT_POINTS", "1200"), 1200),
        ANALYTICS_CRM_MAX_ROWS=_int(os.getenv("ANALYTICS_CRM_MAX_ROWS", "250000"), 250000),
        WARMER_OFFSET_SECONDS=_int(os.getenv("WARMER_OFFSET_SECONDS", "120"), 120),
        SEGMENT_MAX_CUSTOMERS=_int(os.getenv("SEGMENT_MAX_CUSTOMERS", "4000"), 4000),
        AFFINITY_MAX_ORDERS=_int(os.getenv("AFFINITY_MAX_ORDERS", "3000"), 3000),
        PRICE_TARGET_MARGIN_PCT=float(os.getenv("PRICE_TARGET_MARGIN_PCT", "18")),
        FORECAST_CACHE_TTL=_int(os.getenv("FORECAST_CACHE_TTL", "180"), 180),
        ANALYTICS_CACHE_TTL=_int(os.getenv("ANALYTICS_CACHE_TTL", "120"), 120),

        # Data loader passthrough (the loader reads env directly; we mirror here)
        DATA_START_DATE=os.getenv("DATA_START_DATE", "2020-01-01"),
        DATA_END_DATE=os.getenv("DATA_END_DATE", ""),
        ORDER_STATUSES=os.getenv("ORDER_STATUSES", "packed,invoiced,shipped,delivered"),
        AUTO_REFRESH=_bool(os.getenv("AUTO_REFRESH"), True),
        REFRESH_EVERY_MIN=_int(os.getenv("REFRESH_EVERY_MIN", "60"), 60),

        # MSSQL env hints (data_loader.py consumes these environment variables)
        MSSQL_ODBC_DRIVER=os.getenv("MSSQL_ODBC_DRIVER", "ODBC Driver 18 for SQL Server"),
        MSSQL_SERVER=os.getenv("MSSQL_SERVER") or os.getenv("DB_SERVER"),
        MSSQL_DB=os.getenv("MSSQL_DB") or os.getenv("DB_NAME"),
        MSSQL_USER=os.getenv("MSSQL_USER") or os.getenv("DB_USER"),
        MSSQL_PASSWORD=os.getenv("MSSQL_PASS") or os.getenv("MSSQL_PASSWORD") or os.getenv("DB_PASS"),
        MSSQL_TRUSTED=_bool(os.getenv("MSSQL_TRUSTED"), False),
        MSSQL_PORT=os.getenv("MSSQL_PORT", ""),
    )

    # ---------- Keep data_loader and app aligned on shared paths ----------
    # Ensures the loader sees the same path when imported inside the app.
    os.environ.setdefault("PARQUET_PATH", cfg["PARQUET_PATH"])

    # Optional: helpful warnings raised as config flags the app can log at boot
    cfg["__WARNINGS__"] = []
    if cfg["SECRET_KEY"] == "PleaseChangeMe" and env == "production":
        cfg["__WARNINGS__"].append("Using default SECRET_KEY in production. Set FLASK_SECRET_KEY.")
    if database_url.startswith("sqlite") and env == "production":
        cfg["__WARNINGS__"].append("Using SQLite in production. Consider Postgres or MySQL for concurrency.")
    if limiter_storage.startswith("memory") and env == "production":
        cfg["__WARNINGS__"].append("Rate limit storage is in-memory. Set REDIS_URL or RATELIMIT_STORAGE_URL.")

    return cfg
