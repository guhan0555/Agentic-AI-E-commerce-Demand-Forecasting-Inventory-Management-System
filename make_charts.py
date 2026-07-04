import sys
sys.path.insert(0, ".")
from agentic_ecommerce import AgenticOrchestrator, DATA_DIR, CHART_DIR, COLOR_PRIMARY, COLOR_ACCENT, COLOR_GREY
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

orch = AgenticOrchestrator()
R = orch.run(DATA_DIR / "online_retail.csv")

# 1. Overall daily demand trend
fig, ax = plt.subplots(figsize=(9, 3.6))
overall = R["daily_overall"]
ax.plot(overall.index, overall.values, color=COLOR_PRIMARY, linewidth=1.1)
ax.set_title("Overall Daily Units Sold (Dec 2010 - Dec 2011, real transaction data)", fontsize=11, weight="bold")
ax.set_ylabel("Units sold / day")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.tight_layout()
fig.savefig(CHART_DIR / "01_overall_demand_trend.png")
plt.close(fig)

# 2. Forecast chart per top SKU (grid)
fr = R["forecast_results"]
skus = list(fr.keys())
fig, axes = plt.subplots(len(skus), 1, figsize=(9, 2.6 * len(skus)))
if len(skus) == 1:
    axes = [axes]
for ax, sku in zip(axes, skus):
    r = fr[sku]
    hist = r["history"][-60:]
    ax.plot(hist.index, hist.values, color=COLOR_GREY, label="History (last 60d)", linewidth=1)
    fdates = pd.to_datetime(r["forecast_dates"])
    ax.plot(fdates, r["forecast_14d"], color=COLOR_ACCENT, marker="o", markersize=3,
            label="14-day forecast", linewidth=1.6)
    ax.set_title(f"{sku} - {r['description'][:40]}", fontsize=9.5, weight="bold")
    ax.legend(fontsize=7, loc="upper left")
    ax.tick_params(labelsize=7)
fig.suptitle("Demand Forecast Agent: Top-Selling SKUs (real history vs. 14-day forecast)", fontsize=11, weight="bold", y=1.0)
fig.tight_layout()
fig.savefig(CHART_DIR / "02_sku_forecasts.png", bbox_inches="tight")
plt.close(fig)

# 3. Inventory status
inv = R["inventory_df"]
fig, ax = plt.subplots(figsize=(8, 3.2))
colors = inv.status.map({"Healthy": COLOR_PRIMARY, "OVERSTOCK RISK - slow-moving": COLOR_ACCENT,
                          "UNDERSTOCK RISK - reorder now": "#B23A48"})
ax.barh(inv.Description.str[:28], inv.estimated_days_of_cover, color=colors)
ax.axvline(7, color=COLOR_GREY, linestyle="--", linewidth=1, label="Lead time (7d)")
ax.axvline(45, color=COLOR_GREY, linestyle=":", linewidth=1, label="Overstock threshold (45d)")
ax.set_xlabel("Estimated days of stock cover")
ax.set_title("Inventory Agent: Days of Cover by SKU", fontsize=11, weight="bold")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(CHART_DIR / "03_inventory_cover.png")
plt.close(fig)

# 4. Pricing elasticity
pr = R["pricing_df"].dropna(subset=["estimated_elasticity"])
fig, ax = plt.subplots(figsize=(8, 3))
bar_colors = [COLOR_ACCENT if e < -1 else (COLOR_PRIMARY if e < -0.2 else COLOR_GREY) for e in pr.estimated_elasticity]
ax.bar(pr.Description.str[:22], pr.estimated_elasticity, color=bar_colors)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Estimated price elasticity")
ax.set_title("Pricing Agent: Estimated Demand Elasticity by SKU", fontsize=11, weight="bold")
plt.xticks(rotation=20, ha="right", fontsize=8)
fig.tight_layout()
fig.savefig(CHART_DIR / "04_pricing_elasticity.png")
plt.close(fig)

# 5. Customer segments
seg = R["seg_summary"]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
axes[0].pie(seg.customers, labels=seg.persona, autopct="%1.0f%%",
            colors=[COLOR_PRIMARY, COLOR_ACCENT, COLOR_GREY, "#B23A48"][:len(seg)], textprops={"fontsize": 8})
axes[0].set_title("Customer Base by Segment", fontsize=10, weight="bold")
axes[1].bar(seg.persona, seg.avg_monetary, color=[COLOR_PRIMARY, COLOR_ACCENT, COLOR_GREY, "#B23A48"][:len(seg)])
axes[1].set_ylabel("Avg lifetime spend (£)")
axes[1].set_title("Avg Spend by Segment", fontsize=10, weight="bold")
plt.setp(axes[1].get_xticklabels(), rotation=20, ha="right", fontsize=7.5)
fig.suptitle("Customer Insight Agent: RFM Segmentation (KMeans, real order history)", fontsize=11, weight="bold")
fig.tight_layout()
fig.savefig(CHART_DIR / "05_customer_segments.png")
plt.close(fig)

# 6. Review sentiment
rev = R["review_df"]
fig, ax = plt.subplots(figsize=(7, 2.8))
counts = rev.sentiment.value_counts()
ax.bar(counts.index, counts.values, color=[COLOR_PRIMARY, COLOR_GREY, "#B23A48"][:len(counts)])
ax.set_title("Review Agent: Sentiment Distribution (illustrative sample)", fontsize=11, weight="bold")
ax.set_ylabel("Review count")
fig.tight_layout()
fig.savefig(CHART_DIR / "06_review_sentiment.png")
plt.close(fig)

# 7. Top countries by revenue (extra real-data insight)
df = R["df"]
country_rev = df.groupby("Country")["Revenue"].sum().sort_values(ascending=False).head(8)
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.barh(country_rev.index[::-1], country_rev.values[::-1], color=COLOR_PRIMARY)
ax.set_xlabel("Revenue (£)")
ax.set_title("Data Agent Output: Top 8 Countries by Revenue (real data)", fontsize=11, weight="bold")
fig.tight_layout()
fig.savefig(CHART_DIR / "07_top_countries.png")
plt.close(fig)

print("Charts written to", CHART_DIR)
import os
for f in sorted(os.listdir(CHART_DIR)):
    print(" -", f)
