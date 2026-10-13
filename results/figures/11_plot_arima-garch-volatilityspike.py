#!/usr/bin/env python3
"""Plot monthly actual and predicted volatility-spike rates for ARIMA-GARCH.

Run this script from the project root. It reads the saved common-sample
ARIMA-GARCH forecasts and saves a PNG in the figures/ directory.
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
INPUT_FILE = PROJECT_ROOT / "results" / "arima_garch_common_sample_forecasts.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = FIGURES_DIR / "arima_garch_monthly_spike_rate_trends.png"


# ------------------------------------------------------------------
# Load and validate forecasts
# ------------------------------------------------------------------
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Forecast file not found:\n{INPUT_FILE}\n\n"
        "Run the ARIMA-GARCH common-sample forecasting script first, "
        "or update INPUT_FILE to the location of your saved forecast CSV."
    )

df = pd.read_csv(INPUT_FILE)

required_columns = {"target_date", "actual_spike", "predicted_spike"}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise ValueError(
        f"Forecast file is missing columns: {sorted(missing_columns)}\n"
        f"Columns found: {list(df.columns)}"
    )

df["target_date"] = pd.to_datetime(df["target_date"], errors="coerce")
df["actual_spike"] = pd.to_numeric(df["actual_spike"], errors="coerce")
df["predicted_spike"] = pd.to_numeric(df["predicted_spike"], errors="coerce")
df = df.dropna(subset=["target_date", "actual_spike", "predicted_spike"]).copy()

if df.empty:
    raise ValueError("No valid forecast observations remain after cleaning.")

for column in ["actual_spike", "predicted_spike"]:
    if not df[column].isin([0, 1]).all():
        raise ValueError(f"Column '{column}' must contain only binary 0/1 labels.")


# ------------------------------------------------------------------
# Calculate monthly spike rates (% of forecast observations in each month)
# ------------------------------------------------------------------
df["month"] = df["target_date"].dt.to_period("M").dt.to_timestamp()
monthly = (
    df.groupby("month", as_index=False)
    .agg(
        actual_spike_rate=("actual_spike", "mean"),
        predicted_spike_rate=("predicted_spike", "mean"),
        observations=("target_date", "size"),
    )
)

monthly["actual_spike_rate"] *= 100
monthly["predicted_spike_rate"] *= 100


# ------------------------------------------------------------------
# Plot — matching the BVAR monthly spike-rate figure style
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
    "ARIMA-GARCH: Actual vs Predicted Monthly Spike-Rate Trends",
    fontsize=14,
    fontweight="bold",
)
ax.set_xlabel("Month", fontsize=11)
ax.set_ylabel("Spike rate (% of forecast observations)", fontsize=11)
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

# Add the spike threshold note only, matching the volatility plot style.
threshold_values = (
    pd.to_numeric(df["spike_threshold"], errors="coerce")
    if "spike_threshold" in df.columns
    else pd.Series(dtype=float)
)
if not threshold_values.empty and threshold_values.notna().any():
    threshold_text = f"Spike threshold: {threshold_values.dropna().iloc[0]:.4f}"
else:
    threshold_text = ""

if threshold_text:
    ax.text(
        0.99,
        0.97,
        threshold_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )

fig.tight_layout()
fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
plt.show()

print("\nARIMA-GARCH monthly spike-rate plot saved:")
print(OUTPUT_FILE)
print(f"Forecast observations: {len(df):,}")
print(f"Months plotted: {len(monthly):,}")
