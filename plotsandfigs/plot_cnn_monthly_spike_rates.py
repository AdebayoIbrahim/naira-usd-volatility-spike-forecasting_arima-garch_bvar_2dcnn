#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 04:47:59 2026

@author: setup
"""

#!/usr/bin/env python3
"""Plot actual and predicted monthly spike rates for the final 2D-CNN model.

Run from the project root. The plot aggregates daily spike classifications
by calendar month; it does not change the daily forecasting frequency.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
PROJECT_ROOT = Path.cwd()
INPUT_FILE = PROJECT_ROOT / "results" / "cnn_A_final_test_predictions.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = FIGURES_DIR / "figure_4_14_cnn_monthly_spike_rate_trends.png"


# ------------------------------------------------------------------
# Load and validate final daily forecasts
# ------------------------------------------------------------------
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"CNN prediction file not found:\n{INPUT_FILE}\n\n"
        "Check that cnn_A_final_test_predictions.csv is in the results folder."
    )

df = pd.read_csv(INPUT_FILE)
required_columns = {"target_date", "actual_spike", "predicted_spike"}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise ValueError(
        f"Prediction file is missing columns: {sorted(missing_columns)}\n"
        f"Columns found: {list(df.columns)}"
    )

df["target_date"] = pd.to_datetime(df["target_date"], errors="coerce")
for column in ["actual_spike", "predicted_spike"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=["target_date", "actual_spike", "predicted_spike"]).copy()
df = df.sort_values("target_date").reset_index(drop=True)

if df.empty:
    raise ValueError("No valid CNN forecast observations remain after cleaning.")

for column in ["actual_spike", "predicted_spike"]:
    if not df[column].isin([0, 1]).all():
        raise ValueError(f"Column '{column}' must contain only binary 0/1 labels.")

if df["target_date"].duplicated().any():
    raise ValueError("Duplicate target dates detected in the CNN prediction file.")


# ------------------------------------------------------------------
# Aggregate daily spike labels into calendar-month rates
# ------------------------------------------------------------------
df["month"] = df["target_date"].dt.to_period("M").dt.to_timestamp()
monthly = (
    df.groupby("month", as_index=False)
    .agg(
        actual_spike_rate=("actual_spike", "mean"),
        predicted_spike_rate=("predicted_spike", "mean"),
    )
)
monthly["actual_spike_rate"] *= 100
monthly["predicted_spike_rate"] *= 100


# ------------------------------------------------------------------
# Plot — keep styling consistent with ARIMA-GARCH and BVAR plots
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(
    monthly["month"],
    monthly["actual_spike_rate"],
    marker="o",
    linewidth=1.8,
    markersize=4,
    label="Actual monthly spike rate",
)
ax.plot(
    monthly["month"],
    monthly["predicted_spike_rate"],
    marker="s",
    linewidth=1.8,
    markersize=4,
    label="Predicted monthly spike rate",
)

ax.set_title(
    "2D-CNN: Actual vs Predicted Monthly Spike-Rate Trends",
    fontsize=14,
    fontweight="bold",
)
ax.set_xlabel("Month", fontsize=11)
ax.set_ylabel("Spike rate (% of daily forecast observations)", fontsize=11)
ax.set_ylim(0, 105)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
ax.xaxis.set_major_formatter(
    FuncFormatter(
        lambda x, position: (
            mdates.num2date(x).strftime("%Y")
            if mdates.num2date(x).month == 1
            else "Jul"
        )
    )
)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", frameon=True)

# Show the fixed raw-scale spike threshold when available, without adding
# a forecast-observation count to the figure.
threshold_column = (
    "raw_spike_threshold"
    if "raw_spike_threshold" in df.columns
    else "spike_threshold"
    if "spike_threshold" in df.columns
    else None
)
if threshold_column is not None:
    threshold_values = pd.to_numeric(df[threshold_column], errors="coerce").dropna()
    if not threshold_values.empty:
        ax.text(
            0.99,
            0.97,
            f"Spike threshold: {threshold_values.iloc[0]:.4f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
        )

fig.tight_layout()
fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
plt.show()

print("\n2D-CNN monthly spike-rate plot saved:")
print(OUTPUT_FILE)
print(f"Daily forecast rows used: {len(df):,}")
print(f"Months plotted: {len(monthly):,}")
