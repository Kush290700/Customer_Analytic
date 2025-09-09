from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np
import pandas as pd


def make_orders_df(days: int = 60, seed: int = 42, end: str = "2024-05-15") -> pd.DataFrame:
    """Deterministic synthetic orders dataset matching app expectations.

    Columns: Date, CustomerId, CustomerName, RegionName, OrderId, Revenue, Cost,
    WeightLb, ItemCount, ProductName, SKU, SkuName, ShippingMethodName
    """
    rng = np.random.default_rng(seed)
    regions = ["North", "South", "East", "West"]
    methods = ["Ground", "Air", "Courier"]
    customers = [f"C{i:04d}" for i in range(1, 51)]

    records = []
    end_ts = pd.Timestamp(end).normalize()
    dates = pd.date_range(end=end_ts, periods=days, freq="D")
    for d in dates:
        for cid in rng.choice(customers, size=10, replace=False):
            if rng.random() < 0.2:
                continue
            revenue = float(max(rng.lognormal(mean=5.0, sigma=0.6) - 50, 0))
            cost = float(revenue * rng.uniform(0.5, 0.9))
            region = rng.choice(regions)
            method = rng.choice(methods)
            oid = f"O{rng.integers(1, 500000)}"
            records.append(
                {
                    "Date": pd.Timestamp(d),
                    "CustomerId": cid,
                    "CustomerName": f"Cust {cid}",
                    "RegionName": region,
                    "OrderId": oid,
                    "Revenue": revenue,
                    "Cost": cost,
                    "WeightLb": float(rng.uniform(1.0, 20.0)),
                    "ItemCount": int(rng.integers(1, 12)),
                    "ProductName": "SynthProduct",
                    "SKU": "SYN-1",
                    "SkuName": "SYN-1 - SynthProduct",
                    "ShippingMethodName": method,
                }
            )
    df = pd.DataFrame.from_records(records)
    # Ensure schema types are clean
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df
