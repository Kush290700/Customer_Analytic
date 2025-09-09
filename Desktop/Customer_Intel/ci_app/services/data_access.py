from __future__ import annotations

import os
import logging
from functools import lru_cache
from typing import Optional, Sequence, Tuple, Dict

import numpy as np
import pandas as pd
from flask import current_app, session, has_request_context

from ..data_loader import (
    get_dataframe as _loader_get_dataframe,
    load_parquet as _loader_load_parquet,
    PARQUET_PATH as _DEFAULT_PARQUET_PATH,
)


def _log() -> logging.Logger:
    try:
        return current_app.logger
    except Exception:
        return logging.getLogger(__name__)


def _parquet_path() -> str:
    try:
        return os.path.expanduser(current_app.config.get("PARQUET_PATH") or _DEFAULT_PARQUET_PATH)
    except Exception:
        return _DEFAULT_PARQUET_PATH


def _file_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except Exception:
        return -1.0


@lru_cache(maxsize=64)  # keyed by (path, mtime)
def _load_parquet_memoized(path: str, mtime: float) -> pd.DataFrame:
    _log().info("Loading parquet: %s (mtime=%s)", path, mtime)
    return _loader_load_parquet(path)


_NUMERIC_COLS = ("WeightLb", "ItemCount", "Revenue", "Cost")
_DATETIME_COLS = ("Date", "ShipDate")
_REQUIRED_COLS = {
    "CustomerId", "CustomerName", "RegionName",
    "Address1", "Address2", "City", "Province", "PostalCode",
    "OrderId", "Date",
    "Revenue", "Cost", "WeightLb", "ItemCount",
    "ShippingMethodName", "ShipperName",
    "SKU", "ProductName", "SkuName",
    "ShipDate", "SupplierName", "SalesRepId",
}
_ALIASES: Dict[str, str] = {
    "Carrier": "ShipperName",
    "ShippingMethod": "ShippingMethodName",
    "Sku": "SKU",
    "ShipTimestamp": "ShipDate",
    "Addr1": "Address1",
    "Addr2": "Address2",
}


def _normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=sorted(_REQUIRED_COLS | {"Address", "DeliveryNote"}))

    # Map aliases without overwriting existing columns
    for src, dst in _ALIASES.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    # Ensure required columns
    missing = _REQUIRED_COLS.difference(df.columns)
    for col in missing:
        df[col] = pd.NA

    # SkuName convenience
    if "SkuName" not in df.columns or df["SkuName"].isna().all():
        sku = df.get("SKU")
        pname = df.get("ProductName")
        if sku is not None and pname is not None:
            df["SkuName"] = np.where(
                sku.fillna("").astype(str).str.strip() == "",
                pname.astype(str),
                sku.astype(str).str.strip() + " - " + pname.astype(str),
            )
        else:
            df["SkuName"] = df.get("ProductName", pd.Series(dtype="string"))

    # Address + DeliveryNote
    try:
        addr = df[["Address1", "Address2", "City", "Province", "PostalCode"]].astype(str).fillna("")
        df["Address"] = addr.apply(
            lambda r: ", ".join([x for x in [r.Address1, r.Address2, f"{r.City} {r.Province}".strip(), r.PostalCode] if str(x).strip()]),
            axis=1,
        )
    except Exception:
        df["Address"] = pd.Series(dtype="string")
    df["DeliveryNote"] = df.get("Address2", pd.Series(dtype="string"))

    # Dates -> tz-naive
    for col in _DATETIME_COLS:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.tz_localize(None)
            except Exception:
                df[col] = pd.to_datetime(df[col], errors="coerce")

    # Numerics
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)

    # Profit convenience
    if "Profit" not in df.columns:
        df["Profit"] = (pd.to_numeric(df.get("Revenue", 0), errors="coerce").fillna(0.0)
                         - pd.to_numeric(df.get("Cost", 0), errors="coerce").fillna(0.0)).astype(float)

    # Deterministic order
    if "Date" in df.columns and "OrderId" in df.columns:
        df = df.sort_values(["Date", "OrderId"], ascending=[False, False], kind="stable")

    # String-safe IDs
    for id_col in ("CustomerId", "OrderId", "ProductId", "SupplierId", "SalesRepId"):
        if id_col in df.columns:
            df[id_col] = df[id_col].astype("string")

    return df


def _load_or_build(path: str) -> pd.DataFrame:
    # Try cached parquet first
    if os.path.exists(path):
        mtime = _file_mtime(path)
        try:
            df = _load_parquet_memoized(path, mtime)
            return _normalize_schema(df.copy())
        except Exception as e:
            _log().warning("Parquet read failed (%s). Attempting fallback read before rebuild...", e)
            # Tolerant fallback via DuckDB to avoid rebuild
            try:
                import duckdb
                con = duckdb.connect(database=":memory:")
                df_fb = con.execute("SELECT * FROM read_parquet(?)", [str(path)]).fetchdf()
                _log().warning("Loaded parquet via duckdb fallback. Proceeding with normalized data.")
                return _normalize_schema(df_fb.copy())
            except Exception:
                pass

    # Build cache via loader (writes parquet), then read back (memoized)
    _log().info("Building parquet cache via data_loader.get_dataframe() ...")
    _loader_get_dataframe(write_cache=True)
    mtime = _file_mtime(path)
    df2 = _load_parquet_memoized(path, mtime)
    return _normalize_schema(df2.copy())


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def fetch_and_store_data(start: Optional[str] = None,
                         end: Optional[str] = None,
                         path: Optional[str] = None) -> pd.DataFrame:
    return _loader_get_dataframe(start, end, write_cache=True)


def load_data(path: Optional[str] = None) -> pd.DataFrame:
    return _loader_load_parquet(path or _DEFAULT_PARQUET_PATH)


def get_dataframe(start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    path = _parquet_path()
    return _load_or_build(path)


def apply_filters(df: pd.DataFrame,
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None,
                  regions: Optional[Sequence[str]] = None,
                  methods: Optional[Sequence[str]] = None,
                  customers: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()

    def _sess(key: str):
        try:
            return (session.get(key) if has_request_context() else None)
        except Exception:
            return None

    s_start = (start_date or _sess("start_date") or "2020-01-01")
    s_end   = (end_date   or _sess("end_date"))
    if not s_end:
        max_dt = pd.to_datetime(df["Date"], errors="coerce").max()
        s_end = (max_dt or pd.Timestamp.today()).strftime("%Y-%m-%d")

    start = pd.to_datetime(s_start, errors="coerce")
    end   = pd.to_datetime(s_end, errors="coerce")
    if pd.isna(start):
        start = pd.Timestamp("2020-01-01")
    if pd.isna(end):
        end = pd.Timestamp.today()
    end_plus_1 = end + pd.Timedelta(days=1)

    dt_date = pd.to_datetime(df.get("Date"), errors="coerce")
    dt_ship = pd.to_datetime(df.get("ShipDate"), errors="coerce")
    eff_date = dt_date.fillna(dt_ship)
    out = df[(eff_date >= start) & (eff_date < end_plus_1)].copy()

    def _norm(vals: Optional[Sequence[str]]) -> Sequence[str]:
        if vals is None:
            return []
        if isinstance(vals, str):
            s = vals.strip()
            if not s:
                return []
            try:
                import json as _json
                parsed = _json.loads(s)
                if isinstance(parsed, (list, tuple, set)):
                    vals = list(parsed)
                else:
                    vals = [p.strip() for p in s.split(',') if p.strip()]
            except Exception:
                vals = [p.strip() for p in s.split(',') if p.strip()]
        elif isinstance(vals, (set, tuple)):
            vals = list(vals)
        elif not isinstance(vals, list):
            vals = [str(vals)]
        # If the list contains dicts (e.g., {"name": "North", "count": 123}), extract the name
        normed: list[str] = []
        for v in vals:
            try:
                if isinstance(v, dict):
                    name = v.get("name") or v.get("label") or v.get("value")
                    if name is not None:
                        s = str(name).strip()
                        if s:
                            normed.append(s)
                        continue
            except Exception:
                pass
            s = str(v).strip()
            if s:
                normed.append(s)
        # Treat common "all" sentinels as no filter
        lowered = {s.lower() for s in normed}
        if {"__all__", "all", "*"} & lowered:
            return []
        return normed

    regions   = _norm(regions or _sess("regions"))
    methods   = _norm(methods or _sess("methods"))
    customers = _norm(customers or _sess("customers"))

    if regions:
        out = out[out["RegionName"].astype(str).isin(regions)]
    if methods and "ShippingMethodName" in out.columns:
        out = out[out["ShippingMethodName"].astype(str).isin(methods)]
    if customers:
        out = out[out["CustomerName"].astype(str).isin(customers)]

    return out


def distinct_values(df: pd.DataFrame) -> Dict[str, list]:
    if df is None or df.empty:
        return {"regions": [], "methods": [], "customers": []}
    return {
        "regions":   sorted(out for out in df["RegionName"].dropna().astype(str).unique()),
        "methods":   sorted(out for out in df.get("ShippingMethodName", pd.Series(dtype=str)).dropna().astype(str).unique()),
        "customers": sorted(out for out in df["CustomerName"].dropna().astype(str).unique()),
    }


def parquet_info() -> Tuple[str, int, float]:
    path = _parquet_path()
    try:
        df = get_dataframe()
        return path, int(len(df)), _file_mtime(path)
    except Exception:
        return path, 0, _file_mtime(path)


def clear_caches() -> None:
    _load_parquet_memoized.cache_clear()
