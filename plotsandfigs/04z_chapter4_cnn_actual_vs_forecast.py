#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 23:00:15 2026

@author: setup
"""

"""
FIGURE 4.13
Actual vs Forecast Volatility from the Selected Multivariate 2D-CNN

Source:
results/cnn_A_final_test_predictions.csv

Output:
figures/figure_4_13_cnn_actual_vs_forecast.png
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# 1. PATHS
# ============================================================

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_FILE = RESULTS_DIR / "cnn_A_final_test_predictions.csv"
OUTPUT_FILE = FIGURES_DIR / "figure_4_13_cnn_actual_vs_forecast.png"


# ============================================================
# 2. LOAD DATA
# ============================================================

if not SOURCE_FILE.exists():
    raise FileNotFoundError(
        f"Source file not found: {SOURCE_FILE.resolve()}"
    )

df = pd.read_csv(SOURCE_FILE)

if df.empty:
    raise ValueError("CNN prediction file is empty.")


# ============================================================
# 3. VERIFY REQUIRED COLUMNS
# ============================================================

required_columns = [
    "target_date",
    "actual_volatility",
    "predicted_volatility"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}\n"
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# 4. PREPARE DATA
# ============================================================

df["target_date"] = pd.to_datetime(df["target_date"], errors="coerce")

df["actual_volatility"] = pd.to_numeric(
    df["actual_volatility"],
    errors="coerce"
)

df["predicted_volatility"] = pd.to_numeric(
    df["predicted_volatility"],
    errors="coerce"
)

df = (
    df.dropna(
        subset=[
            "target_date",
            "actual_volatility",
            "predicted_volatility"
        ]
    )
    .sort_values("target_date")
    .reset_index(drop=True)
)


# ============================================================
# 5. PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(13, 6.5))

ax.plot(
    df["target_date"],
    df["actual_volatility"],
    linewidth=1.0,
    label="Observed volatility"
)

ax.plot(
    df["target_date"],
    df["predicted_volatility"],
    linewidth=1.2,
    label="CNN forecast"
)


# ============================================================
# 6. HANDLE EXTREME VOLATILITY VALUES
# ============================================================

# Log-like scaling keeps extreme spikes visible without
# allowing them to dominate the entire figure.
ax.set_yscale("symlog", linthresh=0.01)


# ============================================================
# 7. AXIS FORMATTING
# ============================================================

ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Daily Volatility", fontsize=11)

ax.set_title(
    "Observed and Forecast Volatility from the Multivariate 2D-CNN",
    fontsize=13,
    pad=12
)

ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

ax.grid(
    True,
    which="major",
    linestyle="--",
    linewidth=0.5,
    alpha=0.35
)

ax.legend(
    loc="upper left",
    frameon=True
)

plt.xticks(rotation=45)
plt.tight_layout()


# ============================================================
# 8. SAVE
# ============================================================

fig.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 9. REPORT
# ============================================================

print("\n" + "=" * 75)
print("FIGURE 4.13 CREATED")
print("=" * 75)

print(f"Observations : {len(df)}")
print(f"Start date   : {df['target_date'].min().date()}")
print(f"End date     : {df['target_date'].max().date()}")
print(f"Output       : {OUTPUT_FILE}")