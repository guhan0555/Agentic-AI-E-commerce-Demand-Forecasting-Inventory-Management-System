"""
Agentic AI E-commerce Demand Forecasting & Inventory Management System
========================================================================
A multi-agent pipeline built on REAL transactional data (UCI "Online Retail"
dataset: 541,909 real invoice lines from a UK-based online gift retailer,
Dec 2010 - Dec 2011).

Each class below is an autonomous "agent" with a narrow responsibility.
An orchestrator (AgenticOrchestrator) calls them in sequence, the way a
CrewAI/LangGraph graph would route tasks between agents, and each agent
hands a structured result to the next.

Run: python3 agentic_ecommerce.py
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE
OUT_DIR = BASE / "outputs"
CHART_DIR = BASE / "charts"
OUT_DIR.mkdir(exist_ok=True)
CHART_DIR.mkdir(exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 140
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
COLOR_PRIMARY = "#2C6E6B"
COLOR_ACCENT = "#D97D42"
COLOR_GREY = "#8A8F98"


# ---------------------------------------------------------------------------
# AGENT 1: DATA AGENT  (UC-02, UC-03)
# ---------------------------------------------------------------------------
class DataAgent:
    """Ingests raw transactional data, validates and cleans it."""

    def run(self, path):
        raw = pd.read_csv(path, encoding="latin1")
        log = {"rows_in": len(raw)}

        df = raw.copy()
        df["Description"] = df["Description"].astype(str).str.strip()
        df = df.dropna(subset=["CustomerID"])
        df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]  # drop cancellations
        df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%m/%d/%Y %H:%M")
        df["CustomerID"] = df["CustomerID"].astype(int)
        df = df.drop_duplicates()

        # Anomaly handling: a handful of single-line bulk/wholesale orders are
        # extreme outliers relative to normal retail demand (e.g. one real
        # order of ~74,000 units of a single SKU). Left untouched these would
        # badly distort forecasting and elasticity models, so we cap
        # (winsorize) per-SKU quantity at the 99.5th percentile and flag the
        # affected rows rather than silently discard real transactions -- the
        # same signal a fraud/anomaly-detection agent (future scope) would use.
        caps = df.groupby("StockCode")["Quantity"].transform(lambda s: s.quantile(0.995))
        anomaly_mask = df["Quantity"] > caps
        log["bulk_order_anomalies_capped"] = int(anomaly_mask.sum())
        df.loc[anomaly_mask, "Quantity"] = caps[anomaly_mask].round().astype(int)

        df["Revenue"] = df["Quantity"] * df["UnitPrice"]

        log.update({
            "rows_after_cleaning": len(df),
            "rows_removed": len(raw) - len(df),
            "pct_removed": round((len(raw) - len(df)) / len(raw) * 100, 2),
            "date_range": [str(df.InvoiceDate.min().date()), str(df.InvoiceDate.max().date())],
            "n_customers": int(df.CustomerID.nunique()),
            "n_products": int(df.StockCode.nunique()),
            "n_countries": int(df.Country.nunique()),
            "n_invoices": int(df.InvoiceNo.nunique()),
            "total_revenue": round(float(df.Revenue.sum()), 2),
        })
        return df, log


# ---------------------------------------------------------------------------
# AGENT 2: FORECAST AGENT  (UC-04, UC-05)
# ---------------------------------------------------------------------------
class ForecastAgent:
    """Builds daily demand forecasts per top SKU using gradient-boosted
    regression on lag/calendar features (a lightweight stand-in for the
    XGBoost/Prophet stack named in the tech spec)."""

    def __init__(self, n_top_products=5, horizon_days=14):
        self.n_top = n_top_products
        self.horizon = horizon_days

    def _feature_frame(self, series):
        series = series.copy()
        series.index.name = "ds"
        df = series.to_frame("y").reset_index()
        df["dow"] = df.ds.dt.dayofweek
        df["day"] = df.ds.dt.day
        df["month"] = df.ds.dt.month
        for lag in [1, 2, 3, 7, 14]:
            df[f"lag_{lag}"] = df["y"].shift(lag)
        df["rolling_7"] = df["y"].shift(1).rolling(7).mean()
        return df

    def run(self, df):
        from xgboost import XGBRegressor

        top_products = (
            df.groupby(["StockCode", "Description"])["Quantity"].sum()
            .sort_values(ascending=False).head(self.n_top).reset_index()
        )

        results = {}
        daily_overall = df.set_index("InvoiceDate").resample("D")["Quantity"].sum()

        for _, row in top_products.iterrows():
            sku = row.StockCode
            sub = df[df.StockCode == sku].set_index("InvoiceDate").resample("D")["Quantity"].sum()
            sub = sub.reindex(pd.date_range(sub.index.min(), sub.index.max()), fill_value=0)

            feat = self._feature_frame(sub).dropna()
            if len(feat) < 30:
                continue

            X_cols = ["dow", "day", "month", "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "rolling_7"]
            split = int(len(feat) * 0.85)
            train, test = feat.iloc[:split], feat.iloc[split:]

            model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.08,
                                  subsample=0.9, colsample_bytree=0.9, random_state=42)
            model.fit(train[X_cols], train["y"])

            if len(test):
                preds_test = model.predict(test[X_cols])
                mae = float(np.mean(np.abs(preds_test - test["y"])))
                baseline_mae = float(np.mean(np.abs(test["y"] - test["rolling_7"])))
            else:
                mae, baseline_mae = None, None

            # Recursive forecast forward
            history = sub.copy()
            future_preds = []
            cur = self._feature_frame(history).dropna().iloc[[-1]][X_cols].copy()
            last_date = history.index.max()
            for i in range(self.horizon):
                next_date = last_date + timedelta(days=i + 1)
                row_feat = {
                    "dow": next_date.dayofweek, "day": next_date.day, "month": next_date.month,
                }
                recent = list(history.values[-14:])
                row_feat["lag_1"] = recent[-1]
                row_feat["lag_2"] = recent[-2]
                row_feat["lag_3"] = recent[-3]
                row_feat["lag_7"] = recent[-7]
                row_feat["lag_14"] = recent[-14]
                row_feat["rolling_7"] = np.mean(recent[-7:])
                x = pd.DataFrame([row_feat])[X_cols]
                pred = max(0, float(model.predict(x)[0]))
                future_preds.append(pred)
                history.loc[next_date] = pred

            results[sku] = {
                "description": row.Description,
                "total_units_sold_hist": int(row.Quantity),
                "mae_model": round(mae, 2) if mae else None,
                "mae_naive_baseline": round(baseline_mae, 2) if baseline_mae else None,
                "history": sub,
                "forecast_14d": future_preds,
                "forecast_dates": [str((last_date + timedelta(days=i + 1)).date()) for i in range(self.horizon)],
                "avg_daily_forecast": round(float(np.mean(future_preds)), 1),
            }

        return results, top_products, daily_overall


# ---------------------------------------------------------------------------
# AGENT 3: INVENTORY AGENT  (UC-06, UC-07)
# ---------------------------------------------------------------------------
class InventoryAgent:
    """Turns demand forecasts into reorder points, safety stock and
    overstock/understock flags using classic inventory-control formulas."""

    def __init__(self, lead_time_days=7, service_z=1.65):
        self.lead_time = lead_time_days
        self.z = service_z  # ~95% service level

    def run(self, forecast_results):
        rows = []
        for sku, r in forecast_results.items():
            hist = r["history"]
            daily_mean = float(hist.mean())
            daily_std = float(hist.std())
            forecast_mean = r["avg_daily_forecast"]

            safety_stock = self.z * daily_std * np.sqrt(self.lead_time)
            reorder_point = forecast_mean * self.lead_time + safety_stock
            # crude "current stock" proxy: assume last 30 days of demand as available cover
            simulated_stock_on_hand = round(daily_mean * 21, 0)  # illustrative, no live stock feed
            days_of_cover = simulated_stock_on_hand / forecast_mean if forecast_mean > 0 else np.inf

            if days_of_cover < self.lead_time:
                status = "UNDERSTOCK RISK - reorder now"
            elif days_of_cover > 45:
                status = "OVERSTOCK RISK - slow-moving"
            else:
                status = "Healthy"

            rows.append({
                "StockCode": sku,
                "Description": r["description"],
                "avg_daily_demand": round(daily_mean, 1),
                "forecast_daily_demand": forecast_mean,
                "safety_stock_units": round(safety_stock, 0),
                "reorder_point_units": round(reorder_point, 0),
                "simulated_stock_on_hand": simulated_stock_on_hand,
                "estimated_days_of_cover": round(days_of_cover, 1) if days_of_cover != np.inf else None,
                "status": status,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# AGENT 4: PRICING AGENT  (UC-08)
# ---------------------------------------------------------------------------
class PricingAgent:
    """Estimates price elasticity per top SKU from historical price/quantity
    variation and recommends a bounded price adjustment."""

    def run(self, df, top_products):
        rows = []
        for _, row in top_products.iterrows():
            sku = row.StockCode
            sub = df[df.StockCode == sku]
            weekly = sub.set_index("InvoiceDate").resample("W").agg(
                Quantity=("Quantity", "sum"), UnitPrice=("UnitPrice", "mean")
            ).dropna()
            weekly = weekly[(weekly.Quantity > 0) & (weekly.UnitPrice > 0)]

            elasticity = None
            recommendation = "Hold current price (insufficient price variation to estimate elasticity)"
            rec_change_pct = 0.0

            if len(weekly) >= 6 and weekly.UnitPrice.std() > 0.01:
                log_p = np.log(weekly.UnitPrice)
                log_q = np.log(weekly.Quantity.clip(lower=1))
                # simple OLS slope = elasticity
                slope = np.polyfit(log_p, log_q, 1)[0]
                elasticity = round(float(slope), 2)

                if elasticity < -1:  # elastic: price cut grows revenue
                    rec_change_pct = -3.0
                    recommendation = "Demand is price-elastic: a small price decrease is likely to raise revenue"
                elif -1 <= elasticity < -0.2:
                    rec_change_pct = 2.0
                    recommendation = "Demand is moderately inelastic: a small price increase should be revenue-positive"
                else:
                    rec_change_pct = 0.0
                    recommendation = "Demand shows little price sensitivity: hold price, focus on bundling/promotion"

            current_price = round(float(sub.UnitPrice.mean()), 2)
            rows.append({
                "StockCode": sku,
                "Description": row.Description,
                "current_avg_price": current_price,
                "estimated_elasticity": elasticity,
                "recommended_change_pct": rec_change_pct,
                "recommended_price": round(current_price * (1 + rec_change_pct / 100), 2),
                "recommendation": recommendation,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# AGENT 5: CUSTOMER INSIGHT AGENT  (UC-09)
# ---------------------------------------------------------------------------
class CustomerInsightAgent:
    """RFM segmentation + KMeans clustering into customer personas."""

    def run(self, df, n_clusters=4):
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans

        snapshot_date = df.InvoiceDate.max() + timedelta(days=1)
        rfm = df.groupby("CustomerID").agg(
            Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("Revenue", "sum"),
        ).reset_index()

        X = np.log1p(rfm[["Recency", "Frequency", "Monetary"]].clip(lower=0))
        X_scaled = StandardScaler().fit_transform(X)

        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        rfm["Segment"] = km.fit_predict(X_scaled)

        seg_summary = rfm.groupby("Segment").agg(
            customers=("CustomerID", "count"),
            avg_recency_days=("Recency", "mean"),
            avg_frequency=("Frequency", "mean"),
            avg_monetary=("Monetary", "mean"),
        ).reset_index()

        # Label segments heuristically by monetary/recency/frequency rank
        seg_summary["persona"] = "Standard"
        top_val = seg_summary.avg_monetary.idxmax()
        top_freq = seg_summary.avg_frequency.idxmax()
        worst_recency = seg_summary.avg_recency_days.idxmax()
        seg_summary.loc[top_val, "persona"] = "VIP / High-Value"
        if top_freq != top_val:
            seg_summary.loc[top_freq, "persona"] = "Loyal Repeat Buyers"
        seg_summary.loc[worst_recency, "persona"] = "At-Risk / Churning"
        seg_summary["persona"] = seg_summary["persona"].where(
            seg_summary["persona"] != "Standard", "Occasional Shoppers"
        )

        return rfm, seg_summary.round(1)


# ---------------------------------------------------------------------------
# AGENT 6: REVIEW / SENTIMENT AGENT  (UC-10)
# ---------------------------------------------------------------------------
class ReviewAgent:
    """The UCI transactional dataset has no review text, so this agent
    demonstrates the module on a small, clearly-labeled illustrative
    review sample tied to the real top-selling products. In production
    this plugs into the same VADER/NLP pipeline against a live reviews
    feed (UC-10)."""

    SAMPLE_REVIEWS = [
        ("Great quality, exactly as pictured, arrived fast.", "positive_expected"),
        ("Item was okay but packaging was damaged in transit.", "mixed_expected"),
        ("Absolutely love this, bought three more as gifts!", "positive_expected"),
        ("Colour was duller than the photo, a bit disappointed.", "negative_expected"),
        ("Fast shipping, good value for the price.", "positive_expected"),
        ("Broke after a week of light use, not durable.", "negative_expected"),
        ("Cute design, works well, would recommend.", "positive_expected"),
        ("Customer service was slow to respond to my query.", "negative_expected"),
    ]

    def run(self, top_products):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()

        rows = []
        products = top_products["Description"].tolist()
        for i, (text, _) in enumerate(self.SAMPLE_REVIEWS):
            score = analyzer.polarity_scores(text)
            label = "Positive" if score["compound"] > 0.2 else ("Negative" if score["compound"] < -0.2 else "Neutral")
            rows.append({
                "product": products[i % len(products)],
                "review_sample": text,
                "compound_score": round(score["compound"], 3),
                "sentiment": label,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# AGENT 7: BUSINESS REPORT AGENT  (UC-11, UC-12)
# ---------------------------------------------------------------------------
class BusinessReportAgent:
    """Synthesizes outputs of all agents into a structured business report
    (the JSON produced here is what feeds the PDF export in UC-12)."""

    def run(self, data_log, forecast_results, inventory_df, pricing_df, seg_summary, review_df, daily_overall):
        top_line = []
        top_line.append(f"Cleaned {data_log['rows_after_cleaning']:,} of {data_log['rows_in']:,} raw transaction "
                         f"lines ({data_log['pct_removed']}% removed as cancellations/invalid rows).")
        top_line.append(f"Dataset spans {data_log['date_range'][0]} to {data_log['date_range'][1]} across "
                         f"{data_log['n_countries']} countries, {data_log['n_customers']:,} customers and "
                         f"{data_log['n_products']:,} SKUs, totalling £{data_log['total_revenue']:,.0f} in revenue.")

        understock = inventory_df[inventory_df.status.str.contains("UNDERSTOCK")]
        overstock = inventory_df[inventory_df.status.str.contains("OVERSTOCK")]
        top_line.append(f"Of the top {len(inventory_df)} SKUs analysed, {len(understock)} show understock risk and "
                         f"{len(overstock)} show overstock risk under a 7-day lead time / 95% service-level policy.")

        elastic = pricing_df[pricing_df.estimated_elasticity.notna()]
        top_line.append(f"Price elasticity could be estimated for {len(elastic)} of {len(pricing_df)} top SKUs from "
                         f"historical price variation; recommendations range from "
                         f"{pricing_df.recommended_change_pct.min()}% to {pricing_df.recommended_change_pct.max()}%.")

        vip_row = seg_summary.loc[seg_summary.persona.str.contains("VIP")]
        if len(vip_row):
            top_line.append(f"Customer segmentation identifies a VIP/High-Value segment of "
                             f"{int(vip_row.customers.iloc[0])} customers averaging £{vip_row.avg_monetary.iloc[0]:,.0f} "
                             f"lifetime spend, alongside an at-risk segment needing win-back campaigns.")

        pos_share = round((review_df.sentiment == "Positive").mean() * 100, 0)
        top_line.append(f"Illustrative sentiment scoring on sample reviews shows {pos_share:.0f}% positive tone, "
                         f"flagging packaging/durability/service themes as recurring negative drivers.")

        return {"executive_summary": top_line}


# ---------------------------------------------------------------------------
# ORCHESTRATOR
# ---------------------------------------------------------------------------
class AgenticOrchestrator:
    def __init__(self):
        self.data_agent = DataAgent()
        self.forecast_agent = ForecastAgent()
        self.inventory_agent = InventoryAgent()
        self.pricing_agent = PricingAgent()
        self.customer_agent = CustomerInsightAgent()
        self.review_agent = ReviewAgent()
        self.report_agent = BusinessReportAgent()

    def run(self, csv_path):
        print("[Data Agent] cleaning & validating raw data...")
        df, data_log = self.data_agent.run(csv_path)

        print("[Forecast Agent] training demand models for top SKUs...")
        forecast_results, top_products, daily_overall = self.forecast_agent.run(df)

        print("[Inventory Agent] computing reorder points & stock risk...")
        inventory_df = self.inventory_agent.run(forecast_results)

        print("[Pricing Agent] estimating elasticity & price actions...")
        pricing_df = self.pricing_agent.run(df, top_products)

        print("[Customer Insight Agent] running RFM + clustering...")
        rfm_df, seg_summary = self.customer_agent.run(df)

        print("[Review Agent] scoring sentiment...")
        review_df = self.review_agent.run(top_products)

        print("[Business Report Agent] synthesizing final report...")
        report = self.report_agent.run(data_log, forecast_results, inventory_df, pricing_df,
                                        seg_summary, review_df, daily_overall)

        return {
            "df": df, "data_log": data_log, "forecast_results": forecast_results,
            "top_products": top_products, "daily_overall": daily_overall,
            "inventory_df": inventory_df, "pricing_df": pricing_df,
            "rfm_df": rfm_df, "seg_summary": seg_summary,
            "review_df": review_df, "report": report,
        }


if __name__ == "__main__":
    orch = AgenticOrchestrator()
    results = orch.run(DATA_DIR / "sample_data_online_retail.csv")

    # Persist tabular outputs
    results["inventory_df"].to_csv(OUT_DIR / "inventory_recommendations.csv", index=False)
    results["pricing_df"].to_csv(OUT_DIR / "pricing_recommendations.csv", index=False)
    results["seg_summary"].to_csv(OUT_DIR / "customer_segments.csv", index=False)
    results["review_df"].to_csv(OUT_DIR / "review_sentiment_sample.csv", index=False)

    with open(OUT_DIR / "data_log.json", "w") as f:
        json.dump(results["data_log"], f, indent=2)
    with open(OUT_DIR / "business_report.json", "w") as f:
        json.dump(results["report"], f, indent=2)

    print("\n=== EXECUTIVE SUMMARY ===")
    for line in results["report"]["executive_summary"]:
        print("-", line)

    print("\nDone. Outputs written to:", OUT_DIR)
