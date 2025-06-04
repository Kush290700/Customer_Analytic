# File: data_loader.py

import os
import datetime
import logging
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from dotenv import load_dotenv

# ─── Load .env (if present) ──────────────────────────────────────────────────
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# ─── Logging setup ───────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


# ─── (A) DATABASE CONNECTION + RAW FETCH  ────────────────────────────────────
@lru_cache(maxsize=1)
def get_engine():
    server = os.getenv("DB_SERVER", "")
    database = os.getenv("DB_NAME", "")
    user = os.getenv("DB_USER", "")
    pwd = os.getenv("DB_PASS", "")

    if not all([server, database, user, pwd]):
        raise RuntimeError("DB_SERVER/DB_NAME/DB_USER/DB_PASS must be set in .env")

    url = f"mssql+pymssql://{user}:{pwd}@{server}/{database}"
    try:
        engine = create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅ Connected to database")
        return engine
    except OperationalError:
        logger.exception("❌ Cannot connect to SQL Server")
        raise


@lru_cache(maxsize=32)
def fetch_raw_tables(start_date: str = "2020-01-01", end_date: str = None) -> dict:
    if end_date is None:
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")

    engine = get_engine()
    params = {"start": start_date, "end": end_date}

    queries = {
        "orders": text("""
            SELECT 
                OrderId,
                CustomerId,
                SalesRepId,
                CreatedAt      AS CreatedAt_order,
                DateOrdered,
                DateExpected,
                DateShipped    AS ShipDate,
                ShippingMethodRequested
            FROM dbo.Orders
            WHERE OrderStatus = 'packed'
              AND CreatedAt BETWEEN :start AND :end
        """),
        "order_lines": text("""
            SELECT 
                OrderLineId,
                OrderId,
                ProductId,
                ShipperId,
                QuantityShipped,
                Price        AS SalePrice,
                CostPrice    AS UnitCost,
                DateShipped
            FROM dbo.OrderLines
            WHERE CreatedAt BETWEEN :start AND :end
        """),
        "customers": text("""
            SELECT
                CustomerId,
                Name          AS CustomerName,
                RegionId,
                IsRetail,
                Address1,
                Address2,
                City,
                Province,
                PostalCode,
                Phone,
                Email
            FROM dbo.Customers
        """),
        "products": text("""
            SELECT
                ProductId,
                UnitOfBillingId,
                SupplierId,
                SKU,
                Description   AS ProductName,
                ListPrice     AS ProductListPrice,
                CostPrice
            FROM dbo.Products
        """),
        "regions": text("SELECT RegionId, Name AS RegionName FROM dbo.Regions"),
        "shippers": text("SELECT ShipperId, Name AS Carrier FROM dbo.Shippers"),
        "shipping_methods": text("""
            SELECT
                ShippingMethodId AS SMId,
                Name              AS ShippingMethodName
            FROM dbo.ShippingMethods
        """),
        "suppliers": text("SELECT SupplierId, Name AS SupplierName FROM dbo.Suppliers"),
        "packs": text("""
            WITH ol AS (
                SELECT OrderLineId
                  FROM dbo.OrderLines
                 WHERE CreatedAt BETWEEN :start AND :end
            )
            SELECT 
                p.PickedForOrderLine,
                p.WeightLb,
                p.ItemCount,
                p.ShippedAt   AS DeliveryDate
            FROM dbo.Packs p
            JOIN ol ON p.PickedForOrderLine = ol.OrderLineId
        """)
    }

    raw = {}
    for name, qry in queries.items():
        try:
            df = pd.read_sql(qry, engine, params=params)
            logger.debug(f"Fetched '{name}': {len(df):,} rows")
            raw[name] = df
        except SQLAlchemyError:
            logger.exception(f"Error fetching '{name}' – returning empty DataFrame")
            raw[name] = pd.DataFrame()

    return raw


def prepare_full_data(raw: dict) -> pd.DataFrame:
    orders = raw.get("orders", pd.DataFrame())
    lines = raw.get("order_lines", pd.DataFrame())

    if lines.empty:
        raise RuntimeError("No 'order_lines' returned – cannot prepare data")

    # 1) Cast join‐key columns to string
    for df_, cols in [
        (orders, ["OrderId", "CustomerId", "SalesRepId", "ShippingMethodRequested"]),
        (lines, ["OrderLineId", "OrderId", "ProductId", "ShipperId"])
    ]:
        for c in cols:
            if c not in df_.columns:
                raise KeyError(f"Expected column '{c}' in {df_.columns.tolist()}")
            df_[c] = df_[c].astype(str).str.strip()

    # 2) Merge orders + order_lines
    df = lines.merge(orders, on="OrderId", how="inner")
    logger.info(f"After join orders ↔ order_lines: {len(df):,} rows")

    # 3) Lookups (customers, products, regions, shippers, suppliers, shipping_methods)
    lookups = {
        "customers": (
            "CustomerId",
            ["CustomerId", "CustomerName", "RegionId", "IsRetail",
             "Address1", "Address2", "City", "Province", "PostalCode", "Phone", "Email"],
            raw.get("customers")
        ),
        "products": (
            "ProductId",
            ["ProductId", "SupplierId", "UnitOfBillingId", "ProductName", "ProductListPrice", "CostPrice"],
            raw.get("products")
        ),
        "regions": ("RegionId", ["RegionId", "RegionName"], raw.get("regions")),
        "shippers": ("ShipperId", ["ShipperId", "Carrier"], raw.get("shippers")),
        # For shipping_methods, rename SMId → ShippingMethodRequested before merging:
        "smethods": (
            "ShippingMethodRequested",
            ["ShippingMethodRequested", "ShippingMethodName"],
            raw.get("shipping_methods")
        ),
        "suppliers": ("SupplierId", ["SupplierId", "SupplierName"], raw.get("suppliers")),
    }

    for name, (keycol, cols, tbl) in lookups.items():
        if tbl is None or tbl.empty:
            logger.warning(f"Lookup '{name}' missing or empty – skipping")
            continue

        if name == "smethods":
            # rename SMId → ShippingMethodRequested so it matches orders.ShippingMethodRequested
            tbl = tbl.rename(columns={"SMId": "ShippingMethodRequested"})

        for c in cols:
            tbl[c] = tbl[c].astype(str).str.strip()

        df = df.merge(tbl[cols], on=keycol, how="left")
        logger.info(f"After merging '{name}': {len(df):,} rows")

    # 4) Packs aggregation
    packs = raw.get("packs", pd.DataFrame())
    if not packs.empty:
        packs["PickedForOrderLine"] = packs["PickedForOrderLine"].astype(str).str.strip()
        packs = packs.rename(columns={"PickedForOrderLine": "OrderLineId"})
        psum = (
            packs.groupby("OrderLineId", as_index=False)
            .agg(
                WeightLb=("WeightLb", "sum"),
                ItemCount=("ItemCount", "sum"),
                DeliveryDate=("DeliveryDate", "max")
            )
        )
        psum["OrderLineId"] = psum["OrderLineId"].astype(str).str.strip()
        df = df.merge(psum, on="OrderLineId", how="left")
        df[["WeightLb", "ItemCount"]] = df[["WeightLb", "ItemCount"]].fillna(0)
        logger.info(f"After merging 'packs': {len(df):,} rows")
    else:
        df["WeightLb"] = 0.0
        df["ItemCount"] = 0.0
        df["DeliveryDate"] = pd.NaT

    # 5) Numeric safety for key numeric fields
    for col in ["QuantityShipped", "SalePrice", "UnitCost", "WeightLb", "ItemCount"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0.0)

    # 6) Compute Revenue, Cost, Profit
    df["UnitOfBillingId"] = df.get("UnitOfBillingId", "").astype(str)

    df["Revenue"] = np.where(
        df["UnitOfBillingId"] == "3",
        df["WeightLb"] * df["SalePrice"],
        df["ItemCount"] * df["SalePrice"]
    )
    df["Cost"] = np.where(
        df["UnitOfBillingId"] == "3",
        df["WeightLb"] * df["UnitCost"],
        df["ItemCount"] * df["UnitCost"]
    )
    df["Profit"] = df["Revenue"] - df["Cost"]

    # 7) Date & delivery metrics
    df["Date"] = pd.to_datetime(df["CreatedAt_order"], errors="coerce").dt.normalize()
    df["ShipDate"] = pd.to_datetime(df.get("ShipDate"), errors="coerce")
    df["DeliveryDate"] = pd.to_datetime(df.get("DeliveryDate"), errors="coerce")
    df["DateExpected"] = pd.to_datetime(df.get("DateExpected"), errors="coerce")
    df["TransitDays"] = (df["DeliveryDate"] - df["ShipDate"]).dt.days.clip(lower=0)
    df["DeliveryStatus"] = df["DeliveryDate"].le(df["DateExpected"]).map({True: "On Time", False: "Late"})

    logger.info(f"Prepared full data: {len(df):,} rows")
    return df


def fetch_and_store_data(start: str = "2020-01-01",
                         end: str = None,
                         path: str = "cached_data.parquet") -> pd.DataFrame:
    raw = fetch_raw_tables(start, end)
    df = prepare_full_data(raw)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info(f"✅ Data written to '{path}' ({len(df):,} rows).")
    return df


def load_data(path: str = "cached_data.parquet") -> pd.DataFrame:
    cache_file = Path(path)
    if not cache_file.exists():
        raise FileNotFoundError(
            f"No cached file found at '{path}'.\n"
            "Please run `data_loader.fetch_and_store_data(start_date, end_date, path)`\n"
            "on a machine that can access the VPN‐protected database, so as to create the Parquet."
        )
    df = pd.read_parquet(cache_file)
    logger.info(f"📥 Loaded {len(df):,} rows from '{path}'.")
    return df
