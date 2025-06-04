# File: app.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import BytesIO
from pathlib import Path

# ─── Import your data-loading functions ────────────────────────────────────
from data_loader import fetch_and_store_data, load_data

# ─── Page Config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Advanced Meat Warehouse Customer Analytics",
    page_icon="🥩",
    layout="wide",
)

# ─── Constants ───────────────────────────────────────────────────────────
PARQUET_PATH = "cached_data.parquet"


# ─── Utility: Fetch & Cache Data ─────────────────────────────────────────
@st.cache_data(show_spinner=True)
def load_and_prepare(path: str, start: str, end: str) -> pd.DataFrame:
    """
    Load from Parquet if it exists; otherwise fetch from DB and store.
    Returns DataFrame with derived fields.
    """
    cache_file = Path(path)

    if cache_file.exists():
        try:
            df_fetched = load_data(path)
        except Exception:
            df_fetched = fetch_and_store_data(start, end, path)
    else:
        df_fetched = fetch_and_store_data(start, end, path)

    df = df_fetched.copy()

    # Ensure essential columns exist (fill with NaN if missing)
    required = [
        "CustomerId",
        "CustomerName",
        "RegionName",
        "Address1",
        "Address2",
        "City",
        "Province",
        "PostalCode",
        "OrderId",
        "Revenue",
        "Cost",
        "Date",
        "WeightLb",
        "ItemCount",
        "ShippingMethodName",  # ← from shipping_methods lookup
        "Carrier",             # ← from shippers lookup
        "ProductName",
        "ShipDate"
    ]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    # ─── Address Logic Fix ─────────────────────────────────────────────────
    # Only keep Address1 in a new "Address" column
    df["Address1"] = df["Address1"].fillna("").astype(str).str.strip()
    df["Address2"] = df["Address2"].fillna("").astype(str).str.strip()
    df["Address"] = df["Address1"]

    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["ShipDate"] = pd.to_datetime(df["ShipDate"], errors="coerce")

    # Numeric safety
    df["WeightLb"] = pd.to_numeric(df["WeightLb"], errors="coerce").fillna(0.0)
    df["ItemCount"] = pd.to_numeric(df["ItemCount"], errors="coerce").fillna(0.0)
    df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce").fillna(0.0)
    df["Cost"] = pd.to_numeric(df["Cost"], errors="coerce").fillna(0.0)

    return df


# ─── Utility: Customer Excel Export ───────────────────────────────────────
def customer_excel_export(cust_agg: pd.DataFrame) -> BytesIO:
    """
    Create an Excel file with:
      - Sheet 'Instructions' (describing content)
      - Sheet 'Customer Details' (one row per customer with address/contact)
    """
    df_copy = cust_agg.copy()

    # Ensure GrossProfit exists:
    if {"TotalRevenue", "TotalCost"}.issubset(df_copy.columns):
        df_copy["GrossProfit"] = df_copy["TotalRevenue"] - df_copy["TotalCost"]
    else:
        df_copy["GrossProfit"] = 0.0

    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # Instructions sheet
        instr_df = pd.DataFrame({
            "A": [
                "Customer Details Export",
                "",
                "Columns included:",
                "- CustomerId",
                "- CustomerName",
                "- RegionName",
                "- TotalRevenue",
                "- TotalCost",
                "- GrossProfit",
                "- TotalOrders",
                "- AvgOrderWt",
                "- FirstOrder",
                "- LastOrder",
                "- DeliveryTime",
                "- Address",
                "- City",
                "- Province",
                "- PostalCode"
            ]
        })
        instr_df.to_excel(writer, sheet_name="Instructions", index=False, header=False)

        # Customer Details sheet
        detail_cols = [
            "CustomerId",
            "CustomerName",
            "RegionName",
            "TotalRevenue",
            "TotalCost",
            "GrossProfit",
            "TotalOrders",
            "AvgOrderWt",
            "FirstOrder",
            "LastOrder",
            "DeliveryTime",
            "Address",
            "City",
            "Province",
            "PostalCode"
        ]
        detail_cols = [c for c in detail_cols if c in df_copy.columns]
        df_copy[detail_cols].to_excel(writer, sheet_name="Customer Details", index=False)

    output.seek(0)
    return output


# ─── Recommendation Logic ─────────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def get_top_n_customers_all_regions(data: pd.DataFrame, n=50) -> pd.DataFrame:
    """
    Return top N customers by total revenue per region.
    """
    grouped = (
        data.groupby(["RegionName", "CustomerName"], dropna=False)["Revenue"]
            .sum()
            .reset_index()
            .rename(columns={"Revenue": "TotalRevenue"})
            .sort_values(["RegionName", "TotalRevenue"], ascending=[True, False])
    )
    region_frames = []
    for region, gdf in grouped.groupby("RegionName", dropna=False):
        top_slice = gdf.head(n)
        region_frames.append(top_slice)
    if region_frames:
        return pd.concat(region_frames, ignore_index=True)
    return pd.DataFrame(columns=["RegionName", "CustomerName", "TotalRevenue"])


@st.cache_data(show_spinner=True)
def recommend_new_customers(
    data: pd.DataFrame,
    top_n_df: pd.DataFrame,
    min_lineitem_revenue=500
) -> pd.DataFrame:
    """
    For each region in top_n_df, find new customers not in top N,
    but with any line‐item revenue > min_lineitem_revenue.
    """
    recommended_frames = []

    for region, subtop in top_n_df.groupby("RegionName", dropna=False):
        rd = data[data["RegionName"] == region].copy()
        if rd.empty:
            continue
        exclude = subtop["CustomerName"].unique()
        rd = rd[~rd["CustomerName"].isin(exclude)]
        rd = rd[rd["Revenue"] > min_lineitem_revenue]
        if rd.empty:
            continue
        recs = (
            rd.groupby(["RegionName", "CustomerName"], dropna=False)
              .agg(
                  TotalRevenue=("Revenue", "sum"),
                  OrderFrequency=("OrderId", "nunique"),
                  TotalQuantity=("ItemCount", "sum")
              )
              .reset_index()
              .sort_values(["TotalRevenue", "OrderFrequency"], ascending=[False, False])
        )
        recommended_frames.append(recs)

    if recommended_frames:
        return pd.concat(recommended_frames, ignore_index=True)
    return pd.DataFrame(columns=[
        "RegionName", "CustomerName", "TotalRevenue", "OrderFrequency", "TotalQuantity"
    ])


@st.cache_data(show_spinner=True)
def compute_recommendations(
    data: pd.DataFrame,
    top_n: int = 50,
    min_lineitem_revenue: float = 500
):
    """
    Runs:
      1. Top N per region
      2. Recommend new customers
    Returns { "top50": DataFrame, "recommended": DataFrame }
    """
    top_n_df = get_top_n_customers_all_regions(data, n=top_n)
    recommended_df = recommend_new_customers(data, top_n_df, min_lineitem_revenue=min_lineitem_revenue)
    return {
        "top50": top_n_df,
        "recommended": recommended_df
    }


# ─── Sidebar: Date Range Inputs ────────────────────────────────────────────
st.sidebar.header("1. Date Range")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date   = st.sidebar.date_input("End Date",   value=pd.to_datetime("today"))

# ─── Load + prepare data once ─────────────────────────────────────────────
df_all = load_and_prepare(PARQUET_PATH, str(start_date), str(end_date))

# ─── VERY IMPORTANT: Apply date filter here ⟵ DATE FILTER APPLIED HERE ────
start_ts = pd.to_datetime(start_date)
end_ts   = pd.to_datetime(end_date)
df_all = df_all[
    (df_all["Date"] >= start_ts) &
    (df_all["Date"] <= end_ts)
].copy()


# ─── Sidebar: Filters ───────────────────────────────────────────────────────
st.sidebar.header("2. Region Filter")
all_regions = sorted(df_all["RegionName"].dropna().unique())
region_options = ["All"] + all_regions
selected_regions = st.sidebar.multiselect(
    "Select Region(s)",
    region_options,
    default=["All"]
)

st.sidebar.header("3. Shipping Method")
all_methods = sorted(df_all["ShippingMethodName"].dropna().unique())
method_options = ["All"] + all_methods
selected_methods = st.sidebar.multiselect(
    "Select Shipping Method(s)",
    method_options,
    default=["All"]
)

st.sidebar.header("4. Customer Filter")
all_customers = sorted(df_all["CustomerName"].dropna().unique())
customer_options = ["All"] + all_customers
selected_customers = st.sidebar.multiselect(
    "Select Customer(s)",
    customer_options,
    default=["All"]
)

# ─── Apply filters sequentially into df_filtered ────────────────────────────
df_filtered = df_all.copy()

# Region filter
if "All" not in selected_regions and selected_regions:
    df_filtered = df_filtered[df_filtered["RegionName"].isin(selected_regions)]

# Shipping Method filter (replaces previous “Carrier” filter)
if "All" not in selected_methods and selected_methods:
    df_filtered = df_filtered[df_filtered["ShippingMethodName"].isin(selected_methods)]

# CustomerName filter
if "All" not in selected_customers and selected_customers:
    df_filtered = df_filtered[df_filtered["CustomerName"].isin(selected_customers)]

no_data = df_filtered.empty


# ─── Compute per-customer aggregates ───────────────────────────────────────
@st.cache_data(show_spinner=True)
def compute_customer_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates per-customer metrics, including DeliveryTime from ShipDate.
    """
    named_agg = {
        "CustomerName": ("CustomerName", "first"),
        "RegionName":   ("RegionName", "first"),
        "TotalRevenue": ("Revenue", "sum"),
        "TotalCost":    ("Cost", "sum"),
        "TotalOrders":  ("OrderId", "nunique"),
        "TotalWeight":  ("WeightLb", "sum"),
        "FirstOrder":   ("Date", "min"),
        "LastOrder":    ("Date", "max"),
        "Address":      ("Address", "first"),
        "City":         ("City", "first"),
        "Province":     ("Province", "first"),
        "PostalCode":   ("PostalCode", "first")
    }

    cust_agg = (
        df.groupby("CustomerId", dropna=False)
          .agg(**named_agg)
          .reset_index()
    )

    # Derived columns
    cust_agg["AvgOrderWt"] = (
        cust_agg["TotalWeight"] / cust_agg["TotalOrders"].replace({0: np.nan})
    ).fillna(0).round(2)

    cust_agg["DaysSinceLastOrder"] = (
        (pd.Timestamp("today") - cust_agg["LastOrder"])
        .dt.days.clip(lower=0)
    )

    cust_agg["MonthsActive"] = (
        ((pd.Timestamp("today") - cust_agg["FirstOrder"]).dt.days // 30)
        .clip(lower=1)
    )

    cust_agg["RepeatRate"] = (
        cust_agg["TotalOrders"].fillna(0).astype(int)
        / cust_agg["MonthsActive"].replace({0: 1})
    ).round(2)

    # Compute last delivery time per customer (max ShipDate)
    last_ship = df.groupby("CustomerId")["ShipDate"].max().to_dict()
    cust_agg["DeliveryTime"] = cust_agg["CustomerId"].map(last_ship)

    cust_agg = cust_agg.sort_values("TotalRevenue", ascending=False)
    return cust_agg


if not no_data:
    cust_agg = compute_customer_aggregates(df_filtered)
else:
    cust_agg = pd.DataFrame(columns=[
        "CustomerId",
        "CustomerName",
        "RegionName",
        "TotalRevenue",
        "TotalCost",
        "TotalOrders",
        "TotalWeight",
        "AvgOrderWt",
        "FirstOrder",
        "LastOrder",
        "DeliveryTime",
        "Address",
        "City",
        "Province",
        "PostalCode",
        "DaysSinceLastOrder",
        "MonthsActive",
        "RepeatRate"
    ])


# ─── Precompute Recommendations ────────────────────────────────────────────
if not df_filtered.empty:
    rec_results = compute_recommendations(
        data=df_filtered,
        top_n=50,
        min_lineitem_revenue=500
    )
else:
    rec_results = {
        "top50": pd.DataFrame(),
        "recommended": pd.DataFrame()
    }


# ─── Tabs Navigation ───────────────────────────────────────────────────────
tabs = [
    "Instructions",
    "Customer KPIs",
    "Segmentation & RFM",
    "Cohort & Retention",
    "CLV & Profitability",
    "Advanced Analytics",
    "Recommendations",
    "Customer Drilldown",
    "Download Excel"
]
st.sidebar.markdown("---")
section = st.sidebar.radio("Go to", tabs)


# ─── Instructions Tab ─────────────────────────────────────────────────────
if section == "Instructions":
    st.title("📝 Instructions & Usage")
    st.markdown(
    """
        **Advanced Meat Warehouse Customer Analytics Dashboard**

        **How to use:**
        1. **Date Range** (Sidebar): filter orders by date.
        2. **Region Filter**: multi-select “All” or specific regions.
        3. **Shipping Method**: multi-select “All” or specific shipping methods.
        4. **Customer Filter**: multi-select “All” or specific customers.

        **Tabs:**
        - **Customer KPIs**: Top-line metrics and trends.
        - **Segmentation & RFM**: RFM scoring and segments.
        - **Cohort & Retention**: cohort heatmap and churn trends.
        - **CLV & Profitability**: lifetime value and margin charts.
        - **Advanced Analytics**: extra customer analytics graphs.
        - **Recommendations**: top-customers & new-customer recommendations.
        - **Customer Drilldown**: detailed view for a selected customer.
        - **Download Excel**: export full customer details (includes DeliveryTime).

        **Data Setup:**
        - On first run, the app will fetch from SQL Server for the chosen date range, writing `cached_data.parquet`.
        - Subsequent visits use `cached_data.parquet`.
        - Required columns in the Parquet:
          `CustomerId, CustomerName, RegionName, Address1, Address2, City, Province, PostalCode,`
          `OrderId, Revenue, Cost, Date, WeightLb, ItemCount, ShipperId (→ Carrier), ShippingMethodName (delivery method), ProductName, ShipDate`.
        - **Data currently starts from January 1, 2022 in the dataframe and will remain so until the next update.**

        Enjoy exploring your customer data!
    """
    )
    st.stop()


# ─── Customer KPIs Tab ────────────────────────────────────────────────────
elif section == "Customer KPIs":
    st.title("📊 Customer KPIs & Leaderboard")
    if no_data:
        st.warning("No data available for selected filters.")
        st.stop()

    total_cust = cust_agg["CustomerId"].nunique()
    total_rev = df_filtered["Revenue"].sum()
    total_orders = df_filtered["OrderId"].nunique()
    aov = (
        df_filtered.groupby("OrderId")["Revenue"].sum().mean()
        if total_orders > 0 else 0.0
    )
    overall_churn_pct = (
        cust_agg["DaysSinceLastOrder"].gt(90).sum() * 100.0 / total_cust
        if total_cust else 0.0
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers", f"{total_cust:,}")
    c2.metric("Total Revenue", f"${total_rev:,.0f}")
    c3.metric("Total Orders", f"{total_orders:,}")
    c4.metric("Avg Order Value", f"${aov:,.2f}")
    c5.metric("Churn Rate (90d)", f"{overall_churn_pct:.1f}%")

    st.markdown("---")
    st.subheader("🏆 Top 10 Customers by Revenue")
    top10 = (
        cust_agg.head(10)[[
            "CustomerName", "RegionName", "TotalRevenue", "TotalOrders", "RepeatRate"
        ]].rename(columns={
            "TotalRevenue": "Revenue",
            "TotalOrders": "Orders",
            "RepeatRate": "RetentionRate"
        })
    )
    st.dataframe(
        top10.style.format({
            "Revenue": "${:,.0f}",
            "RetentionRate": "{:.2f}"
        }),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("📈 Monthly Revenue & Orders Trend")
    by_month = (
        df_filtered
            .groupby(pd.Grouper(key="Date", freq="ME"), dropna=False)
            .agg(MonthRevenue=("Revenue", "sum"), MonthOrders=("OrderId", "nunique"))
            .reset_index()
    )
    by_month["month_ts"] = by_month["Date"].dt.to_period("M").dt.to_timestamp()
    trend_df = by_month.melt(
        id_vars="month_ts",
        value_vars=["MonthRevenue", "MonthOrders"],
        var_name="Metric",
        value_name="Value"
    )
    trend = (
        alt.Chart(trend_df)
           .mark_line(point=True)
           .encode(
               x=alt.X("month_ts:T", title="Month"),
               y=alt.Y("Value:Q"),
               color="Metric:N",
               tooltip=["month_ts", "Metric", alt.Tooltip("Value:Q", format=",")]
           )
           .properties(height=300)
    )
    st.altair_chart(trend, use_container_width=True)


# ─── Segmentation & RFM Tab ───────────────────────────────────────────────
elif section == "Segmentation & RFM":
    st.title("👥 Customer Segmentation & RFM Analysis")
    if no_data:
        st.warning("No data available for selected filters.")
        st.stop()

    rfm = cust_agg.copy()
    rfm["Recency"] = rfm["DaysSinceLastOrder"]
    rfm["Frequency"] = rfm["TotalOrders"]
    rfm["Monetary"] = rfm["TotalRevenue"]

    # Compute RFM Scores
    rfm["RecencyScore"] = pd.qcut(rfm["Recency"], 4, labels=[4, 3, 2, 1]).astype(int)
    rfm["FrequencyScore"] = pd.qcut(
        rfm["Frequency"].rank(method="first", ascending=False),
        4,
        labels=[1, 2, 3, 4]
    ).astype(int)
    rfm["MonetaryScore"] = pd.qcut(
        rfm["Monetary"].rank(method="first", ascending=False),
        4,
        labels=[1, 2, 3, 4]
    ).astype(int)
    rfm["RFM_Score"] = rfm["RecencyScore"] + rfm["FrequencyScore"] + rfm["MonetaryScore"]

    def label_rfm(row):
        if row["RFM_Score"] >= 10:
            return "Champion"
        elif row["RFM_Score"] >= 8:
            return "Loyal"
        elif row["RFM_Score"] >= 6:
            return "At Risk"
        return "Dormant"

    rfm["RFM_Segment"] = rfm.apply(label_rfm, axis=1)

    st.subheader("RFM Segment Distribution")
    seg_counts = (
        rfm.groupby("RFM_Segment", dropna=False)["CustomerId"]
           .count()
           .reset_index()
           .rename(columns={"CustomerId": "Count"})
           .sort_values("Count", ascending=False)
    )
    pie = (
        alt.Chart(seg_counts)
           .mark_arc(innerRadius=50)
           .encode(
               theta="Count:Q",
               color=alt.Color("RFM_Segment:N", title="Segment"),
               tooltip=["RFM_Segment", "Count"]
           )
           .properties(height=300)
    )
    st.altair_chart(pie, use_container_width=True)

    st.markdown("---")
    st.subheader("Top 10 by RFM Score")
    top_rfm = (
        rfm.sort_values("RFM_Score", ascending=False).head(10)[[
            "CustomerName", "RegionName", "RFM_Score", "Recency", "Frequency", "Monetary"
        ]].rename(columns={
            "RFM_Score": "RFM",
            "Recency": "Days Since Last",
            "Frequency": "Orders",
            "Monetary": "Revenue"
        })
    )
    st.dataframe(top_rfm.style.format({"Revenue": "${:,.0f}"}), use_container_width=True)

    st.markdown("---")
    st.subheader("RFM Scatter: Frequency vs. Monetary")
    scatter = (
        alt.Chart(rfm)
           .mark_circle(size=80)
           .encode(
               x=alt.X("Frequency:Q", title="Order Frequency"),
               y=alt.Y("Monetary:Q", title="Lifetime Revenue"),
               color=alt.Color("RFM_Score:Q", scale=alt.Scale(scheme="turbo"), title="RFM"),
               tooltip=[
                   "CustomerName",
                   alt.Tooltip("Monetary:Q", format="$,.0f"),
                   "Frequency",
                   "RFM_Score"
               ]
           )
           .interactive()
           .properties(height=400)
    )
    st.altair_chart(scatter, use_container_width=True)


# ─── Cohort & Retention Tab ───────────────────────────────────────────────
elif section == "Cohort & Retention":
    st.title("📆 Cohort Analysis & Retention")
    if no_data:
        st.warning("No data available for selected filters.")
        st.stop()

    # Build cohort DataFrame
    cust_cohort = cust_agg.copy()
    cust_cohort["CohortMonth"] = cust_cohort["FirstOrder"].dt.to_period("M")

    cohort_df = (
        cust_cohort
            .groupby("CohortMonth", dropna=False)
            .agg(
                TotalCustomers=("CustomerId", "count"),
                StillActive90=("DaysSinceLastOrder", lambda x: (x <= 90).sum())
            )
            .reset_index()
    )
    cohort_df["CohortMonthTS"] = cohort_df["CohortMonth"].dt.to_timestamp()
    cohort_df = cohort_df.sort_values("CohortMonthTS")
    cohort_df["RetentionRate"] = (
        cohort_df["StillActive90"] / cohort_df["TotalCustomers"]
    ).round(2)

    st.subheader("Retention by Cohort Month")
    cohort_chart = (
        alt.Chart(cohort_df)
           .mark_bar()
           .encode(
               x=alt.X(
                   "CohortMonthTS:T",
                   title="Cohort Month",
                   axis=alt.Axis(format="%Y-%m", tickCount="month")
               ),
               y=alt.Y("TotalCustomers:Q", title="New Customers"),
               tooltip=[
                   alt.Tooltip("CohortMonthTS:T", title="Cohort"),
                   alt.Tooltip("TotalCustomers:Q", title="Count"),
                   alt.Tooltip("StillActive90:Q", title="Still Active ≤ 90d"),
                   alt.Tooltip("RetentionRate:Q", format=".0%", title="Retention")
               ]
           )
           .properties(height=300)
    )
    st.altair_chart(cohort_chart, use_container_width=True)

    st.markdown("---")
    st.subheader("Churn Risk by Threshold")
    thresh = st.select_slider(
        "Select Churn Threshold (days since last order)",
        options=[90, 180, 365],
        value=90
    )
    churn_df = cust_agg[cust_agg["DaysSinceLastOrder"] > thresh]
    churn_by_region = (
        churn_df
            .groupby("RegionName", dropna=False)["CustomerId"]
            .nunique()
            .reset_index()
            .rename(columns={"CustomerId": "ChurnCount"})
            .sort_values("ChurnCount", ascending=False)
    )
    if churn_by_region.empty:
        st.write(f"No customers inactive for more than {thresh} days.")
    else:
        churn_chart = (
            alt.Chart(churn_by_region)
               .mark_bar()
               .encode(
                   x=alt.X("RegionName:N", sort="-y", title="Region"),
                   y=alt.Y("ChurnCount:Q", title=f"Customers > {thresh}d"),
                   tooltip=["RegionName", "ChurnCount"]
               )
               .properties(height=300)
        )
        st.altair_chart(churn_chart, use_container_width=True)

    st.markdown("---")
    st.subheader("Monthly Churn Trend (Threshold = 90d)")
    first_order_overall = df_filtered["Date"].min()
    last_order_overall = df_filtered["Date"].max()
    if pd.notnull(first_order_overall) and pd.notnull(last_order_overall):
        month_range = pd.period_range(
            start=first_order_overall.to_period("M"),
            end=last_order_overall.to_period("M"),
            freq="M"
        )
        churn_trend_list = []
        for per in month_range:
            month_end = per.to_timestamp("M")
            cutoff_date = month_end - pd.Timedelta(days=90)
            churned_count = cust_agg[cust_agg["LastOrder"] < cutoff_date]["CustomerId"].nunique()
            churn_trend_list.append({"Month": month_end, "ChurnedCustomers": churned_count})
        churn_trend_df = pd.DataFrame(churn_trend_list)

        churn_line = (
            alt.Chart(churn_trend_df)
               .mark_line(point=True)
               .encode(
                   x=alt.X("Month:T", title="Month"),
                   y=alt.Y("ChurnedCustomers:Q", title="Number of Churned Customers"),
                   tooltip=["Month", alt.Tooltip("ChurnedCustomers:Q", title="Churned Customers")]
               )
               .properties(height=300)
        )
        st.altair_chart(churn_line, use_container_width=True)
    else:
        st.write("Insufficient order date range to compute monthly churn trend.")

    st.markdown("---")
    st.subheader("New vs Returning Revenue (date range)")
    merged = df_filtered.copy()
    first_order_map = cust_agg.set_index("CustomerId")["FirstOrder"].to_dict()
    merged["IsNewOrder"] = merged.apply(
        lambda r: r["Date"] == first_order_map.get(r["CustomerId"]), axis=1
    )
    new_rev = merged[merged["IsNewOrder"]]["Revenue"].sum()
    ret_rev = merged[~merged["IsNewOrder"]]["Revenue"].sum()
    behavior_df = pd.DataFrame({
        "Type": ["New Customer Revenue", "Returning Customer Revenue"],
        "Revenue": [new_rev, ret_rev]
    })
    pie = (
        alt.Chart(behavior_df)
           .mark_arc(innerRadius=50)
           .encode(
               theta="Revenue:Q",
               color=alt.Color("Type:N", title="Segment"),
               tooltip=["Type", alt.Tooltip("Revenue:Q", format="$,.0f")]
           )
           .properties(height=300)
    )
    st.altair_chart(pie, use_container_width=True)

    st.markdown("---")
    st.subheader("Download Churned Customer Details")
    if churn_df.empty:
        st.write(f"No customers have been inactive for more than {thresh} days.")
    else:
        churn_detail_df = churn_df[[
            "CustomerId", "CustomerName", "RegionName",
            "LastOrder", "DaysSinceLastOrder",
            "Address", "City", "Province", "PostalCode"
        ]].drop_duplicates()

        revenue_cost = cust_agg[["CustomerId", "TotalRevenue", "TotalCost"]].drop_duplicates()
        churn_detail_df = churn_detail_df.merge(
            revenue_cost,
            on="CustomerId",
            how="left"
        )
        churn_detail_df["GrossProfit"] = (
            churn_detail_df["TotalRevenue"].fillna(0)
            - churn_detail_df["TotalCost"].fillna(0)
        )

        final_cols = [
            "CustomerId", "CustomerName", "RegionName",
            "LastOrder", "DaysSinceLastOrder",
            "TotalRevenue", "TotalCost", "GrossProfit",
            "Address", "City", "Province", "PostalCode"
        ]
        final_cols = [c for c in final_cols if c in churn_detail_df.columns]
        churn_export_df = churn_detail_df[final_cols]

        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            churn_export_df.to_excel(writer, sheet_name="ChurnedCustomers", index=False)
        output.seek(0)

        st.download_button(
            label=f"Download {len(churn_export_df)} Churned Customers (Excel)",
            data=output,
            file_name="churned_customers_details.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# ─── CLV & Profitability Tab ─────────────────────────────────────────────
elif section == "CLV & Profitability":
    st.title("💰 Customer Lifetime Value & Profitability")
    if no_data:
        st.warning("No data available for selected filters.")
        st.stop()

    cust_clv = cust_agg.copy()
    cust_clv["AvgOrderValue"] = (
        cust_clv["TotalRevenue"] / cust_clv["TotalOrders"].replace({0: np.nan})
    ).fillna(0)
    cust_clv["OrdersPerYear"] = (
        cust_clv["TotalOrders"] / cust_clv["MonthsActive"].replace({0: 1}) * 12
    ).fillna(0)
    life_years = 3
    cust_clv["CLV"] = (
        cust_clv["AvgOrderValue"] * cust_clv["OrdersPerYear"] * life_years
    ).round(2)

    st.subheader("Top 10 by CLV")
    top_clv = (
        cust_clv.sort_values("CLV", ascending=False).head(10)[[
            "CustomerName", "RegionName", "AvgOrderValue", "OrdersPerYear", "CLV"
        ]]
    )
    st.dataframe(
        top_clv.style.format({
            "AvgOrderValue": "${:,.2f}",
            "CLV": "${:,.0f}"
        }),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("Gross Margin % by Customer")
    cust_clv["GrossMarginPct"] = (
        (cust_clv["TotalRevenue"] - cust_clv["TotalCost"])
        / cust_clv["TotalRevenue"].replace({0: np.nan}) * 100
    ).round(2).fillna(0)
    margin_chart = (
        alt.Chart(cust_clv)
           .mark_bar()
           .encode(
               x=alt.X("GrossMarginPct:Q", title="Gross Margin (%)"),
               y=alt.Y("CustomerName:N", sort="-x", title=None),
               tooltip=[
                   "CustomerName",
                   alt.Tooltip("TotalRevenue:Q", format="$,.0f"),
                   alt.Tooltip("TotalCost:Q", format="$,.0f"),
                   alt.Tooltip("GrossMarginPct:Q", format=".1f"),
                   alt.Tooltip("CLV:Q", format="$,.0f")
               ]
           )
           .properties(height=400, title="Gross Margin % by Customer")
    )
    st.altair_chart(margin_chart, use_container_width=True)

    st.markdown("---")
    st.subheader("Profit vs Volume Scatter")
    profit_scatter = (
        alt.Chart(cust_clv)
           .mark_circle(size=80)
           .encode(
               x=alt.X("TotalRevenue:Q", title="Total Revenue"),
               y=alt.Y("TotalOrders:Q", title="Total Orders"),
               color=alt.Color("GrossMarginPct:Q", scale=alt.Scale(scheme="turbo"), title="Gross Margin %"),
               size=alt.Size("CLV:Q", title="Predicted CLV"),
               tooltip=[
                   "CustomerName",
                   alt.Tooltip("TotalRevenue:Q", format="$,.0f"),
                   "TotalOrders",
                   alt.Tooltip("GrossMarginPct:Q", format=".1f"),
                   alt.Tooltip("CLV:Q", format="$,.0f")
               ]
           )
           .interactive()
           .properties(height=400)
    )
    st.altair_chart(profit_scatter, use_container_width=True)


# ─── Advanced Analytics Tab ───────────────────────────────────────────────
elif section == "Advanced Analytics":
    st.title("📊 Advanced Customer Analytics")
    if no_data:
        st.warning("No data available for selected filters.")
        st.stop()

    # 1) Pareto Chart
    st.subheader("1. Pareto: Cumulative Revenue by Customer")
    pareto_df = cust_agg.copy().reset_index(drop=True)
    pareto_df = pareto_df.sort_values("TotalRevenue", ascending=False).reset_index(drop=True)
    pareto_df["Rank"] = pareto_df.index + 1
    pareto_df["CumulativeRevenue"] = pareto_df["TotalRevenue"].cumsum()
    pareto_df["CumulativePct"] = pareto_df["CumulativeRevenue"] / pareto_df["TotalRevenue"].sum()
    pareto_plot = (
        alt.Chart(pareto_df)
           .mark_line(point=True)
           .encode(
               x=alt.X("Rank:Q", title="Customer Rank"),
               y=alt.Y("CumulativePct:Q", title="Cumulative Revenue %", axis=alt.Axis(format=".0%")),
               tooltip=[
                   alt.Tooltip("CustomerName:N", title="Customer"),
                   alt.Tooltip("TotalRevenue:Q", format="$,.0f", title="Revenue"),
                   alt.Tooltip("CumulativePct:Q", format=".0%", title="Cum %")
               ]
           )
           .properties(height=300, title="Pareto: Top Customers Drive Revenue")
    )
    st.altair_chart(pareto_plot, use_container_width=True)

    st.markdown("---")
    # 2) Customer Tenure Distribution
    st.subheader("2. Customer Tenure Distribution")
    tenure_df = cust_agg.copy()
    tenure_df["TenureMonths"] = tenure_df["MonthsActive"]
    hist_tenure = (
        alt.Chart(tenure_df)
           .mark_bar()
           .encode(
               alt.X("TenureMonths:Q", bin=alt.Bin(maxbins=20), title="Months Active"),
               alt.Y("count():Q", title="Number of Customers"),
               tooltip=[alt.Tooltip("count():Q", title="Customers")]
           )
           .properties(height=300)
    )
    st.altair_chart(hist_tenure, use_container_width=True)

    st.markdown("---")
    # 3) Revenue by Day of Week
    st.subheader("3. Revenue by Day of Week")
    df_week = df_filtered.copy()
    df_week["Weekday"] = df_week["Date"].dt.day_name()
    weekday_rev = df_week.groupby("Weekday", dropna=False)["Revenue"].sum().reset_index()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_rev["Weekday"] = pd.Categorical(weekday_rev["Weekday"], categories=order, ordered=True)
    weekday_rev = weekday_rev.sort_values("Weekday")
    weekday_chart = (
        alt.Chart(weekday_rev)
           .mark_bar()
           .encode(
               x=alt.X("Weekday:N", title="Day of Week"),
               y=alt.Y("Revenue:Q", title="Total Revenue"),
               tooltip=[alt.Tooltip("Revenue:Q", format="$,.0f")]
           )
           .properties(height=300)
    )
    st.altair_chart(weekday_chart, use_container_width=True)

    st.markdown("---")
    # 4) Order Frequency Distribution
    st.subheader("4. Order Frequency Distribution")
    freq_df = cust_agg[["CustomerName", "TotalOrders"]].copy()
    freq_hist = (
        alt.Chart(freq_df)
           .mark_bar()
           .encode(
               alt.X("TotalOrders:Q", bin=alt.Bin(maxbins=20), title="Number of Orders"),
               alt.Y("count():Q", title="Number of Customers"),
               tooltip=[alt.Tooltip("count():Q", title="Customers")]
           )
           .properties(height=300)
    )
    st.altair_chart(freq_hist, use_container_width=True)


# ─── Recommendations Tab ─────────────────────────────────────────────────
elif section == "Recommendations":
    st.title("⭐ Customer Recommendations")
    if df_filtered.empty:
        st.warning("No data available for the selected filters.")
        st.stop()

    top50_df = rec_results["top50"]
    recommended_df = rec_results["recommended"]

    st.subheader("1. Top 50 Customers per Region")
    if top50_df.empty:
        st.write("No top‐50 data available.")
    else:
        st.dataframe(
            top50_df.rename(columns={"TotalRevenue": "Revenue"}).style.format({"Revenue": "${:,.0f}"}),
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("2. Recommended New Customers")
    if recommended_df.empty:
        st.write("No recommended customers (threshold may be too high).")
    else:
        st.dataframe(
            recommended_df.rename(columns={"TotalRevenue": "Revenue"}).style.format({"Revenue": "${:,.0f}"}),
            use_container_width=True
        )


# ─── Customer Drilldown Tab ───────────────────────────────────────────────
elif section == "Customer Drilldown":
    st.title("🔎 Customer Detail Drilldown")
    if no_data:
        st.warning("No data available for selected filters.")
        st.stop()

    cust_names = sorted(cust_agg["CustomerName"].dropna().unique())
    if not cust_names:
        st.warning("No customers to select.")
        st.stop()

    selected = st.selectbox("Choose a Customer", cust_names)

    if selected:
        cust_id = cust_agg[cust_agg["CustomerName"] == selected]["CustomerId"].iloc[0]
        df_c = df_filtered[df_filtered["CustomerId"] == cust_id].copy()

        info = cust_agg[cust_agg["CustomerId"] == cust_id].iloc[0]
        st.subheader(f"Profile: {info['CustomerName']} ({info['RegionName']})")
        st.markdown(f"""
        - **First Order:** {info['FirstOrder'].date()}
        - **Last Order:** {info['LastOrder'].date()}
        - **Last Delivery:** {info['DeliveryTime'].date() if pd.notnull(info['DeliveryTime']) else 'N/A'}
        - **Total Revenue:** ${info['TotalRevenue']:,.0f}
        - **Total Orders:** {info['TotalOrders']}
        - **Avg Order Weight:** {info['AvgOrderWt']:.2f} lbs
        - **Repeat Rate:** {info['RepeatRate']:.2f}
        """)

        st.markdown("---")
        st.subheader("Order History Timeline")
        orders_c = (
            df_c.groupby(pd.Grouper(key="Date", freq="M"), dropna=False)
                .agg(MonthlyRevenue=("Revenue", "sum"), MonthlyOrders=("OrderId", "nunique"))
                .reset_index()
        )
        orders_c["month_ts"] = orders_c["Date"]
        timeline = (
            alt.Chart(
                orders_c.melt(
                    id_vars="month_ts",
                    value_vars=["MonthlyRevenue", "MonthlyOrders"],
                    var_name="Metric",
                    value_name="Value"
                )
            )
               .mark_line(point=True)
               .encode(
                   x=alt.X("month_ts:T", title="Month"),
                   y=alt.Y("Value:Q"),
                   color="Metric:N",
                   tooltip=["month_ts", "Metric", alt.Tooltip("Value:Q", format=",")]
               )
               .properties(height=300)
        )
        st.altair_chart(timeline, use_container_width=True)

        st.markdown("---")
        if "ProductName" in df_c.columns:
            df_c["ProductCategory"] = df_c["ProductName"].astype(str).apply(
                lambda x: x.split(" – ")[0] if " – " in x else x
            )
            pm_c = (
                df_c.groupby("ProductCategory", dropna=False)
                    .agg(Spend=("Revenue", "sum"), Weight=("WeightLb", "sum"))
                    .reset_index()
                    .sort_values("Spend", ascending=False)
            )
            st.subheader("Product Category Mix")
            pm_chart = (
                alt.Chart(pm_c)
                   .mark_bar()
                   .encode(
                       x=alt.X("Spend:Q", title="Spend ($)"),
                       y=alt.Y("ProductCategory:N", sort="-x", title=None),
                       tooltip=[
                           "ProductCategory",
                           alt.Tooltip("Spend:Q", format="$,.0f"),
                           alt.Tooltip("Weight:Q", format=",.1f")
                       ]
                   )
                   .properties(height=300)
            )
            st.altair_chart(pm_chart, use_container_width=True)
        else:
            st.write("No product‐level data available.")


# ─── Download Excel Tab ───────────────────────────────────────────────────
elif section == "Download Excel":
    st.title("⬇️ Download Customer Details (Excel)")
    if no_data:
        st.warning("No data to export for selected filters.")
        st.stop()

    st.markdown("""
    The Excel file includes two sheets:
    1. **Instructions** – Explains the contents of the 'Customer Details' sheet.
    2. **Customer Details** – One row per customer with DeliveryTime, address, and key metrics.
    """)
    excel_bytes = customer_excel_export(cust_agg)
    st.download_button(
        label="Download Customer Details Excel",
        data=excel_bytes,
        file_name="Customer_Details.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    st.info("The export reflects your current filters (date, region, shipping method, etc.).")
