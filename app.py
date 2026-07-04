"""
Agentic AI E-commerce Demand Forecasting & Inventory Management System
------------------------------------------------------------------------
Streamlit front-end wrapping the multi-agent pipeline in agentic_ecommerce.py.

Run locally:
    pip install streamlit pandas numpy matplotlib scikit-learn xgboost vaderSentiment
    streamlit run app.py

Covers:
  UC-01 Login (simple role picker demo)      UC-07 Overstock detection
  UC-02 Upload dataset                        UC-08 Dynamic pricing
  UC-03 Data cleaning and validation          UC-09 Customer analytics
  UC-04 Sales analytics                       UC-10 Review sentiment analysis
  UC-05 Demand forecasting                    UC-11 AI business report generation
  UC-06 Inventory optimization                UC-12 Export reports (CSV)
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from agentic_ecommerce import AgenticOrchestrator, DATA_DIR

st.set_page_config(page_title="Agentic AI E-commerce Ops", layout="wide")

# --- UC-01: Login (lightweight role-based demo) -----------------------------
if "role" not in st.session_state:
    st.session_state.role = None

if st.session_state.role is None:
    st.title("🔐 Agentic AI E-commerce System — Login")
    role = st.selectbox("Select your role", ["Admin", "Store Manager", "Inventory Manager", "Business Analyst"])
    if st.button("Log in"):
        st.session_state.role = role
        st.rerun()
    st.stop()

st.sidebar.success(f"Logged in as: {st.session_state.role}")
if st.sidebar.button("Log out"):
    st.session_state.role = None
    st.rerun()

st.title("🤖 Agentic AI E-commerce Demand Forecasting & Inventory Management")
st.caption("Multi-agent system: Data → Forecast → Inventory → Pricing → Customer → Review → Business Report agents")

# --- UC-02: Data upload ------------------------------------------------------
st.sidebar.header("1. Data Source")
uploaded = st.sidebar.file_uploader("Upload sales CSV (InvoiceNo, StockCode, Description, Quantity, "
                                     "InvoiceDate, UnitPrice, CustomerID, Country)", type="csv")
use_sample = st.sidebar.checkbox("Use bundled real sample dataset (UCI Online Retail)", value=uploaded is None)

data_path = None
if uploaded is not None:
    tmp_path = Path("/tmp/uploaded_sales.csv")
    tmp_path.write_bytes(uploaded.getvalue())
    data_path = tmp_path
elif use_sample:
    data_path = DATA_DIR / "sample_data_online_retail.csv"

n_top = st.sidebar.slider("Top SKUs to forecast", 3, 15, 5)
horizon = st.sidebar.slider("Forecast horizon (days)", 7, 30, 14)

run_btn = st.sidebar.button("▶ Run Agentic Pipeline", type="primary")

if run_btn and data_path is not None:
    with st.spinner("Agents running: cleaning data, forecasting demand, optimizing inventory, "
                     "pricing, segmenting customers, scoring reviews..."):
        orch = AgenticOrchestrator()
        orch.forecast_agent.n_top = n_top
        orch.forecast_agent.horizon = horizon
        st.session_state.results = orch.run(data_path)
    st.success("Pipeline complete.")

if "results" not in st.session_state:
    st.info("Upload a dataset or use the bundled sample, then click **Run Agentic Pipeline** in the sidebar.")
    st.stop()

R = st.session_state.results

tabs = st.tabs(["📊 Sales Analytics", "📈 Forecasting", "📦 Inventory", "💲 Pricing",
                "👥 Customers", "⭐ Reviews", "📑 AI Business Report"])

# --- UC-04 Sales analytics ---------------------------------------------------
with tabs[0]:
    log = R["data_log"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue", f"£{log['total_revenue']:,.0f}")
    c2.metric("Customers", f"{log['n_customers']:,}")
    c3.metric("SKUs", f"{log['n_products']:,}")
    c4.metric("Countries", log["n_countries"])
    st.caption(f"Cleaned {log['rows_after_cleaning']:,} / {log['rows_in']:,} rows "
               f"({log['pct_removed']}% removed) · {log.get('bulk_order_anomalies_capped', 0)} "
               f"bulk-order anomalies capped · Range {log['date_range'][0]} → {log['date_range'][1]}")
    st.line_chart(R["daily_overall"])
    country_rev = R["df"].groupby("Country")["Revenue"].sum().sort_values(ascending=False).head(10)
    st.bar_chart(country_rev)

# --- UC-05 Forecasting --------------------------------------------------------
with tabs[1]:
    for sku, r in R["forecast_results"].items():
        st.subheader(f"{sku} — {r['description']}")
        fdf = pd.DataFrame({"date": pd.to_datetime(r["forecast_dates"]), "forecast_units": r["forecast_14d"]})
        colA, colB = st.columns([2, 1])
        with colA:
            hist = r["history"][-60:].rename("units").to_frame()
            st.line_chart(pd.concat([hist, fdf.set_index("date").rename(columns={"forecast_units": "units"})]))
        with colB:
            st.metric("Avg forecast demand/day", r["avg_daily_forecast"])
            st.metric("Model MAE", r["mae_model"])
            st.metric("Naive baseline MAE", r["mae_naive_baseline"])

# --- UC-06 / UC-07 Inventory ---------------------------------------------------
with tabs[2]:
    st.dataframe(R["inventory_df"], use_container_width=True)
    st.download_button("⬇ Export inventory recommendations (CSV)",
                        R["inventory_df"].to_csv(index=False), "inventory_recommendations.csv")

# --- UC-08 Pricing --------------------------------------------------------------
with tabs[3]:
    st.dataframe(R["pricing_df"], use_container_width=True)
    st.download_button("⬇ Export pricing recommendations (CSV)",
                        R["pricing_df"].to_csv(index=False), "pricing_recommendations.csv")

# --- UC-09 Customer analytics -----------------------------------------------
with tabs[4]:
    st.dataframe(R["seg_summary"], use_container_width=True)
    st.bar_chart(R["seg_summary"].set_index("persona")["avg_monetary"])
    st.download_button("⬇ Export customer segments (CSV)",
                        R["seg_summary"].to_csv(index=False), "customer_segments.csv")

# --- UC-10 Review sentiment ---------------------------------------------------
with tabs[5]:
    st.caption("Illustrative sample — plug in a live reviews feed to replace this with real review text.")
    st.dataframe(R["review_df"], use_container_width=True)

# --- UC-11 / UC-12 AI business report + export --------------------------------
with tabs[6]:
    for line in R["report"]["executive_summary"]:
        st.markdown(f"- {line}")
    st.download_button("⬇ Export full business report (JSON)",
                        pd.Series(R["report"]).to_json(), "business_report.json")
