# data_loader.py — Production-ready data loader with auto-refresh + optional AI querying
# -------------------------------------------------------------------------------
# Features
# - Env-driven MSSQL connection (DATABASE_URL or split vars), with Windows auth support
# - Resilient fetch with retries/backoff + pool_pre_ping
# - Corrected joins (ShippingMethodRequested -> ShippingMethods -> Shippers)
# - IDs returned as strings; monetary values as float
# - Revenue/Cost per your business rule (UnitOfBillingId == 3 => WeightLb, else ItemCount)
# - Adds SKU and SkuName = "SKU - ProductName"
# - Writes parquet cache; loads fast in your Flask app
# - APScheduler auto-refresh (interval configurable)
# - Optional natural-language AI query -> SQL (plug your LLM provider)
# -------------------------------------------------------------------------------

import os
import sys
import time
import logging
import datetime as dt
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError, OperationalError, InterfaceError
from sqlalchemy.engine import URL

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE  = os.getenv("LOG_FILE",  "etl_ml_app.log")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("data_loader")

# ─────────────────────────────────────────────────────────────
# Parquet engine detection
# ─────────────────────────────────────────────────────────────
try:
    import pyarrow  # noqa: F401
    PARQUET_ENGINE = "pyarrow"
except Exception:
    try:
        import fastparquet  # noqa: F401
        PARQUET_ENGINE = "fastparquet"
    except Exception:
        PARQUET_ENGINE = "pyarrow"

# ─────────────────────────────────────────────────────────────
# Env & defaults
# ─────────────────────────────────────────────────────────────
# Date window defaults
DEFAULT_START = os.getenv("DATA_START_DATE", "2020-01-01")
DEFAULT_END   = os.getenv("DATA_END_DATE")  # None => today

ORDER_STATUSES = tuple(
    s.strip() for s in os.getenv(
        "ORDER_STATUSES", "packed,invoiced,shipped,delivered"
    ).split(",") if s.strip()
) or ("packed",)

# Cache file
PARQUET_PATH = os.getenv("PARQUET_PATH", "cache/fact_analytics.parquet")

# Auto-refresh
AUTO_REFRESH = os.getenv("AUTO_REFRESH", "true").lower() in {"1", "true", "yes"}
REFRESH_EVERY_MIN = int(os.getenv("REFRESH_EVERY_MIN", "60"))

# SQL pool
POOL_SIZE      = int(os.getenv("DB_POOL_SIZE", "8"))
MAX_OVERFLOW   = int(os.getenv("DB_MAX_OVERFLOW", "8"))
POOL_TIMEOUT   = int(os.getenv("DB_POOL_TIMEOUT", "30"))
CONNECT_TO     = int(os.getenv("DB_LOGIN_TIMEOUT", "15"))

# LLM (optional)
LLM_PROVIDER   = os.getenv("LLM_PROVIDER", "").lower()  # "openai", "azure", "openrouter", "" (disabled)
LLM_MODEL      = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ─────────────────────────────────────────────────────────────
# DB URL builder (supports URL or split vars; legacy env names too)
# ─────────────────────────────────────────────────────────────
def _build_url() -> URL:
    database_url = os.getenv("DATABASE_URL", "").strip()
    if database_url:
        logger.info("Using DATABASE_URL for connection.")
        return database_url  # SQLAlchemy will parse string URL too

    # Accept both new and legacy env vars
    server = (os.getenv("MSSQL_SERVER") or os.getenv("DB_SERVER") or "").strip()
    db     = (os.getenv("MSSQL_DB")     or os.getenv("DB_NAME")   or "").strip()
    user   = (os.getenv("MSSQL_USER")   or os.getenv("DB_USER")   or "").strip()
    pwd    = (os.getenv("MSSQL_PASSWORD") or os.getenv("DB_PASS") or "").strip()
    driver = os.getenv("MSSQL_ODBC_DRIVER", "ODBC Driver 18 for SQL Server").strip()
    port   = os.getenv("MSSQL_PORT", "").strip()
    trusted = os.getenv("MSSQL_TRUSTED", "false").lower() in {"1", "true", "yes"}

    if not (server and db and (trusted or (user and pwd))):
        raise RuntimeError("Set DATABASE_URL or MSSQL_SERVER/MSSQL_DB/(MSSQL_USER+MSSQL_PASSWORD or MSSQL_TRUSTED)")

    if port and ("\\" not in server) and ("," not in server):
        server = f"{server},{port}"

    # Build ODBC string (use Login Timeout, not Connection Timeout)
    if trusted:
        odbc = (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};DATABASE={db};"
            "Trusted_Connection=yes;Encrypt=yes;TrustServerCertificate=yes;"
            f"Login Timeout={CONNECT_TO};"
        )
    else:
        odbc = (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};DATABASE={db};"
            f"UID={user};PWD={pwd};"
            "Encrypt=yes;TrustServerCertificate=yes;"
            f"Login Timeout={CONNECT_TO};"
        )

    safe_user = (user[:2] + "***") if user else ""
    logger.info("Using SQL ODBC: DRIVER=%s; SERVER=%s; DB=%s; USER=%s; Trusted=%s",
                driver, server, db, safe_user, trusted)

    return URL.create("mssql+pyodbc", query={"odbc_connect": odbc})

@lru_cache(maxsize=1)
def get_engine():
    url = _build_url()
    eng = create_engine(
        url,
        pool_pre_ping=True,
        pool_size=POOL_SIZE,
        max_overflow=MAX_OVERFLOW,
        pool_timeout=POOL_TIMEOUT,
        future=True,
    )
    # health check
    with eng.connect() as c:
        c.exec_driver_sql("SELECT 1")
    logger.info("✅ DB engine ready (pool=%s/%s)", POOL_SIZE, MAX_OVERFLOW)
    return eng

# ─────────────────────────────────────────────────────────────
# SQL builder (statuses parameterization, half-open window)
# ─────────────────────────────────────────────────────────────
def build_fact_sql(statuses):
    # produce placeholders :s0,:s1,... for IN clause
    s_params = ", ".join(f":s{i}" for i in range(len(statuses)))
    # DEV row limit (optional)
    dev_limit = os.getenv("DEV_ROW_LIMIT")
    dev_limit_clause = f"TOP ({int(dev_limit)}) " if dev_limit and dev_limit.isdigit() else ""

    sql = f"""
    /* Fact builder inline (uses Orders/OrderLines/Packs/Products/Customers/Regions/ShippingMethods/Shippers/Suppliers) */
    WITH PackAgg AS (
      SELECT
        p.PickedForOrderLine AS OrderLineId,
        SUM(COALESCE(p.WeightLb,0))  AS TotalWeightLb,
        SUM(COALESCE(p.ItemCount,0)) AS TotalItemCount
      FROM dbo.Packs p
      GROUP BY p.PickedForOrderLine
    ),
    Base AS (
      SELECT
        -- IDs as NVARCHAR to be GUID/int safe downstream
        CAST(o.CustomerId AS NVARCHAR(50)) AS CustomerId,
        c.Name AS CustomerName, r.Name AS RegionName,
        c.Address1, c.Address2, c.City, c.Province, c.PostalCode,

        CAST(o.OrderId    AS NVARCHAR(50)) AS OrderId,
        CAST(o.DateExpected AS datetime)   AS [Date],

        CAST(COALESCE(pa.TotalWeightLb,0)  AS float) AS WeightLb,
        CAST(COALESCE(pa.TotalItemCount,0) AS float) AS ItemCount,

        CAST(ol.Price     AS float) AS Price,
        CAST(ol.CostPrice AS float) AS CostPrice,

        pr.UnitOfBillingId,
        CAST(ol.ProductId  AS NVARCHAR(50)) AS ProductId,
        pr.Name AS ProductName,
        pr.SKU  AS SKU,
        -- SkuName for display
        LTRIM(RTRIM(
            CASE WHEN pr.SKU IS NULL OR pr.SKU = '' THEN pr.Name
                 ELSE CONCAT(pr.SKU, ' - ', pr.Name)
            END
        )) AS SkuName,

        CAST(pr.SupplierId AS NVARCHAR(50)) AS SupplierId,
        sup.Name AS SupplierName,

        CAST(o.SalesRepId  AS NVARCHAR(50)) AS SalesRepId,

        -- Corrected carrier join: Orders -> ShippingMethods -> Shippers
        sm.Name AS ShippingMethodName,
        sh.Name AS ShipperName
      FROM dbo.Orders o
      JOIN dbo.OrderLines ol ON o.OrderId = ol.OrderId
      LEFT JOIN PackAgg pa   ON pa.OrderLineId = ol.OrderLineId
      JOIN dbo.Products pr   ON pr.ProductId = ol.ProductId
      JOIN dbo.Customers c   ON c.CustomerId = o.CustomerId
      LEFT JOIN dbo.Regions r   ON r.RegionId = c.RegionId
      LEFT JOIN dbo.ShippingMethods sm ON sm.ShippingMethodId = o.ShippingMethodRequested
      LEFT JOIN dbo.Shippers        sh ON sh.ShipperId        = sm.ShipperId
      LEFT JOIN dbo.Suppliers sup     ON sup.SupplierId       = pr.SupplierId
      WHERE o.DateExpected IS NOT NULL
        AND o.OrderStatus IN ({s_params})
        AND o.DateExpected >= :start
        AND o.DateExpected <  :end_plus_1
    )
    SELECT {dev_limit_clause}
      b.CustomerId, b.CustomerName, b.RegionName,
      b.Address1, b.Address2, b.City, b.Province, b.PostalCode,
      b.OrderId, b.[Date],
      b.WeightLb, b.ItemCount,
      b.Price, b.CostPrice,
      b.UnitOfBillingId,
      b.ProductId, b.ProductName, b.SKU, b.SkuName,
      b.SupplierId, b.SupplierName,
      b.SalesRepId,
      b.ShippingMethodName, b.ShipperName,
      -- Business revenue/cost rule
      CAST(CASE WHEN b.UnitOfBillingId = 3
                THEN COALESCE(b.WeightLb,0) * COALESCE(b.Price,0)
                ELSE COALESCE(b.ItemCount,0) * COALESCE(b.Price,0)
           END AS float) AS Revenue,
      CAST(CASE WHEN b.UnitOfBillingId = 3
                THEN COALESCE(b.WeightLb,0) * COALESCE(b.CostPrice,0)
                ELSE COALESCE(b.ItemCount,0) * COALESCE(b.CostPrice,0)
           END AS float) AS Cost
    FROM Base b
    ORDER BY b.[Date] DESC, b.OrderId DESC
    """
    return text(sql)

# ─────────────────────────────────────────────────────────────
# Fetch with retries/backoff
# ─────────────────────────────────────────────────────────────
def _half_open_window(start: str|dt.date|None, end: str|dt.date|None):
    s = pd.to_datetime(start or DEFAULT_START).date()
    if end:
        e = pd.to_datetime(end).date()
    else:
        e = dt.date.today()
    e_plus_1 = e + dt.timedelta(days=1)
    return s.isoformat(), e.isoformat(), e_plus_1.isoformat()

def fetch_fact(start=None, end=None, statuses=ORDER_STATUSES, max_retries=3):
    start_iso, end_iso, end_plus_1 = _half_open_window(start, end)
    logger.info("Fetching fact window %s → %s (statuses=%s)", start_iso, end_iso, statuses)

    eng = get_engine()
    sql = build_fact_sql(statuses)

    params = {"start": start_iso, "end_plus_1": end_plus_1}
    for i, s in enumerate(statuses):
        params[f"s{i}"] = s

    last_err = None
    for attempt in range(1, max_retries+1):
        try:
            with eng.connect() as conn:
                df = pd.read_sql(sql, conn, params=params)
            logger.info("Fetched %s rows.", f"{len(df):,}")
            return df
        except (OperationalError, InterfaceError, SQLAlchemyError) as e:
            last_err = e
            logger.warning("DB fetch failed (attempt %d/%d): %s", attempt, max_retries, e)
            time.sleep(min(2**attempt, 8))

    raise RuntimeError(f"Data fetch failed after retries: {last_err}")

# ─────────────────────────────────────────────────────────────
# Post-processing & cache IO
# ─────────────────────────────────────────────────────────────
def _coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "CustomerId": "string",
        "CustomerName": "string",
        "RegionName": "string",
        "Address1": "string", "Address2": "string", "City": "string",
        "Province": "string", "PostalCode": "string",
        "OrderId": "string",
        "Date": "datetime64[ns]",
        "WeightLb": "float64",
        "ItemCount": "float64",
        "Price": "float64",
        "CostPrice": "float64",
        "UnitOfBillingId": "Int64",
        "ProductId": "string",
        "ProductName": "string",
        "SKU": "string",
        "SkuName": "string",
        "SupplierId": "string",
        "SupplierName": "string",
        "SalesRepId": "string",
        "ShippingMethodName": "string",
        "ShipperName": "string",
        "Revenue": "float64",
        "Cost": "float64",
    }
    for col, typ in required.items():
        if col not in df.columns:
            df[col] = pd.Series(pd.NA, dtype="string" if typ.endswith("string") else object)
        try:
            if typ.startswith("datetime"):
                df[col] = pd.to_datetime(df[col], errors="coerce")
            else:
                df[col] = df[col].astype(typ)
        except Exception:
            # fallback: try coercion for numerics
            if typ in ("float64","Int64"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = df[col].astype("string")

    # Computed fields
    df["Profit"] = (df["Revenue"] - df["Cost"]).astype("float64")
    # Normalize SkuName if SKU missing
    df["SkuName"] = df["SkuName"].fillna(df["ProductName"])
    return df

def write_parquet(df: pd.DataFrame, path: str | Path = PARQUET_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Always write with our detected engine so read/write stay consistent
    df.to_parquet(path, engine=PARQUET_ENGINE, index=False)
    logger.info("Wrote %s rows to '%s'", f"{len(df):,}", path)

def write_parquet_atomic(df: pd.DataFrame, path: str | Path = PARQUET_PATH):
    """Atomic parquet write to avoid partial/corrupt files during refresh."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, engine=PARQUET_ENGINE, index=False)
    try:
        tmp.replace(path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        except Exception:
            pass
    logger.info("Wrote %s rows to '%s'", f"{len(df):,}", path)

def load_parquet(path: str | Path = PARQUET_PATH) -> pd.DataFrame:
    """
    Robust parquet reader that tries the selected engine first and then
    falls back to the alternative to mitigate snappy/cramjam issues.
    """
    path = Path(path)
    last_err: Exception | None = None

    # Candidate engines: prefer the one we used to write
    engines = [PARQUET_ENGINE]
    alt = "pyarrow" if PARQUET_ENGINE == "fastparquet" else "fastparquet"
    # Only try alt if importable to avoid noisy ImportError
    try:
        __import__(alt)
        engines.append(alt)
    except Exception:
        pass

    for eng in engines:
        try:
            df = pd.read_parquet(path, engine=eng)
            logger.info("📥 Loaded %s rows from '%s' (engine=%s)", f"{len(df):,}", path, eng)
            return df
        except Exception as e:
            last_err = e
            # Log concise hint for common snappy/coramjam issues and try next engine
            msg = str(e).lower()
            if "snappy" in msg or "cramjam" in msg or "decompress" in msg:
                logger.warning("Parquet read failed with %s on engine=%s; trying fallback engine…", e.__class__.__name__, eng)
            else:
                logger.warning("Parquet read failed on engine=%s: %s", eng, e)

    # Last resort: try duckdb parquet scan and rewrite cache with primary engine
    try:
        import duckdb
        con = duckdb.connect(database=":memory:")
        df = con.execute("SELECT * FROM read_parquet(?)", [str(path)]).fetchdf()
        logger.warning("Loaded parquet via duckdb fallback; rewriting cache with primary engine.")
        try:
            write_parquet_atomic(df, path)
        except Exception as wex:
            logger.warning("Failed to rewrite parquet after duckdb fallback: %s", wex)
        return df
    except Exception:
        pass

    # If all attempts failed, raise the last error so callers can rebuild cache
    raise last_err if last_err is not None else RuntimeError(f"Failed to read parquet: {path}")

def get_dataframe(start=None, end=None, statuses=ORDER_STATUSES, write_cache=True):
    df = fetch_fact(start, end, statuses)
    df = _coerce_schema(df)
    if write_cache:
        try:
            write_parquet_atomic(df, PARQUET_PATH)
        except Exception:
            # Fallback to direct write if atomic replace is not supported
            write_parquet(df, PARQUET_PATH)
    return df

# ─────────────────────────────────────────────────────────────
# Optional: Natural-language AI query → SQL over the cached parquet
# ─────────────────────────────────────────────────────────────
def _llm_generate_sql(question: str, table_name: str = "fact"):
    """
    Plug-in LLM text-to-SQL. Requires setting LLM_PROVIDER + OPENAI_API_KEY (or your provider),
    and installing its SDK. This function returns a proposed SQL string against the provided
    schema (duckdb/sqlite-ish declarative). You can replace this with your own orchestrator.
    """
    schema = """
    Table fact columns:
      CustomerId TEXT, CustomerName TEXT, RegionName TEXT, Address1 TEXT, Address2 TEXT, City TEXT,
      Province TEXT, PostalCode TEXT, OrderId TEXT, Date TIMESTAMP,
      WeightLb DOUBLE, ItemCount DOUBLE, Price DOUBLE, CostPrice DOUBLE, UnitOfBillingId INTEGER,
      ProductId TEXT, ProductName TEXT, SKU TEXT, SkuName TEXT,
      SupplierId TEXT, SupplierName TEXT, SalesRepId TEXT,
      ShippingMethodName TEXT, ShipperName TEXT,
      Revenue DOUBLE, Cost DOUBLE, Profit DOUBLE
    """
    prompt = (
        "You are a SQL generator. Produce a single SELECT statement for DuckDB that answers the question.\n"
        "Use only the provided table and columns, avoid JOINs, and prefer GROUP BY for aggregations.\n"
        f"{schema}\n\nQuestion: {question}\nSQL:"
    )

    try:
        if LLM_PROVIDER == "openai":
            try:
                # OpenAI Python SDK v1 style
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_API_KEY)
                resp = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                sql = resp.choices[0].message.content.strip()
                return sql
            except Exception as e:
                logger.warning("OpenAI call failed: %s", e)
        # Add other providers here if you like.
    except Exception as e:
        logger.warning("LLM provider not configured: %s", e)

    raise RuntimeError("AI querying not configured. Set LLM_PROVIDER/OPENAI_API_KEY, or pass raw SQL to sql_query().")

def ai_query(question: str, df: pd.DataFrame | None = None):
    """
    Natural-language question over the cached data.
    Uses DuckDB (in-memory) to run LLM-generated SQL safely (read-only).
    """
    import duckdb  # pip install duckdb
    if df is None:
        df = load_parquet(PARQUET_PATH)
    duckdb_con = duckdb.connect(database=":memory:")
    duckdb_con.register("fact", df)
    sql = _llm_generate_sql(question, table_name="fact")
    logger.info("AI-SQL:\n%s", sql)
    return duckdb_con.execute(sql).fetchdf()

def sql_query(sql: str, df: pd.DataFrame | None = None):
    """
    Direct SQL over the cached data (no LLM). Uses DuckDB.
    """
    import duckdb
    if df is None:
        df = load_parquet(PARQUET_PATH)
    duckdb_con = duckdb.connect(database=":memory:")
    duckdb_con.register("fact", df)
    return duckdb_con.execute(sql).fetchdf()

# ─────────────────────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────────────────────
_scheduler = None

def _auto_refresh_job():
    try:
        start = DEFAULT_START
        end   = DEFAULT_END  # None => today
        df = get_dataframe(start, end, ORDER_STATUSES, write_cache=True)
        logger.info("Auto-refresh complete. Rows: %s", f"{len(df):,}")
    except Exception as e:
        logger.exception("Auto-refresh job failed: %s", e)

def start_auto_refresh(every_minutes: int = REFRESH_EVERY_MIN, delay_first: bool = True):
    global _scheduler
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except Exception as e:
        logger.warning("APScheduler not installed; auto-refresh disabled. (%s)", e)
        return

    if _scheduler:
        return _scheduler

    _scheduler = BackgroundScheduler()
    first_run = dt.datetime.now() + dt.timedelta(minutes=every_minutes) if delay_first else dt.datetime.now()
    _scheduler.add_job(_auto_refresh_job, 'interval', minutes=every_minutes,
                       next_run_time=first_run)
    _scheduler.start()
    logger.info("Auto-refresh scheduler started (every %d minutes).", every_minutes)
    return _scheduler

# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def _parse_cli():
    import argparse
    p = argparse.ArgumentParser(description="Load analytics fact and cache to parquet.")
    p.add_argument("--start", type=str, default=DEFAULT_START, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end",   type=str, default=DEFAULT_END, help="End date inclusive (YYYY-MM-DD). Omit for today.")
    p.add_argument("--statuses", type=str, default=",".join(ORDER_STATUSES),
                   help="Comma-separated statuses (default: packed,invoiced,shipped,delivered)")
    p.add_argument("--no-cache", action="store_true", help="Do not write parquet cache")
    p.add_argument("--once", action="store_true", help="Run once and exit (do not start scheduler)")
    return p.parse_args()

if __name__ == "__main__":
    # Smoke test
    s = os.getenv("SMOKE_START", "2020-01-01")
    e = os.getenv("SMOKE_END") or dt.date.today().isoformat()
    logger.info("Smoke test: %s → %s (statuses=%s, env=%s)", s, e, ORDER_STATUSES, os.getenv("FLASK_ENV","production"))
    try:
        args = _parse_cli()
        statuses = tuple(x.strip() for x in args.statuses.split(",") if x.strip())
        df = get_dataframe(args.start or s, args.end or e, statuses, write_cache=(not args.no_cache))
        logger.info("Done. Cached file: %s", PARQUET_PATH)
        if not args.once and AUTO_REFRESH:
            start_auto_refresh()
            # Keep the process alive for scheduler (CTRL+C to exit)
            while True:
                time.sleep(3600)
    except Exception as ex:
        logger.exception("Loader failed: %s", ex)
        sys.exit(1)
else:
    # If imported (e.g., by Flask), optionally start scheduler (honor TESTING and APP_ENABLE_SCHEDULER)
    is_testing = os.getenv("TESTING", "").lower() in {"1","true","yes"}
    enable_sched = os.getenv("APP_ENABLE_SCHEDULER", "true").lower() in {"1","true","yes"}
    if AUTO_REFRESH and enable_sched and not is_testing:
        start_auto_refresh()


