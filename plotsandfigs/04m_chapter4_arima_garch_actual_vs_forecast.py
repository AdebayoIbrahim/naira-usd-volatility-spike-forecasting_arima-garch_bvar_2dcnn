#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 05:42:01 2026

@author: setup
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# CHAPTER 4 - ARIMA-GARCH RESULTS
# [FIG 4.8 HERE]
# Actual vs Forecast Volatility
# ============================================================


# ------------------------------------------------------------
# 1. Project paths
# ------------------------------------------------------------
PROJECT_ROOT = Path.cwd()

FORECAST_PATH = (
    PROJECT_ROOT
    / "results"
    / "arima_garch_common_sample_forecasts.csv"
)

FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Check file
# ------------------------------------------------------------
if not FORECAST_PATH.exists():
    raise FileNotFoundError(
        f"Forecast file not found:\n{FORECAST_PATH}"
    )


# ------------------------------------------------------------
# 3. Load data
# ------------------------------------------------------------
df = pd.read_csv(
    FORECAST_PATH
)

required_columns = [
    "target_date",
    "actual_volatility",
    "predicted_volatility"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}\n\n"
        f"Available columns:\n{df.columns.tolist()}"
    )


# ------------------------------------------------------------
# 4. Convert and prepare
# ------------------------------------------------------------
df["target_date"] = pd.to_datetime(
    df["target_date"],
    errors="coerce"
)

df["actual_volatility"] = pd.to_numeric(
    df["actual_volatility"],
    errors="coerce"
)

df["predicted_volatility"] = pd.to_numeric(
    df["predicted_volatility"],
    errors="coerce"
)

plot_df = (
    df[
        [
            "target_date",
            "actual_volatility",
            "predicted_volatility"
        ]
    ]
    .replace([float("inf"), float("-inf")], pd.NA)
    .dropna()
    .sort_values("target_date")
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# 5. Create figure
# ------------------------------------------------------------
fig, ax = plt.subplots(
    figsize=(13, 6),
    dpi=150
)


# ------------------------------------------------------------
# 6. Plot actual and predicted volatility
# ------------------------------------------------------------
ax.plot(
    plot_df["target_date"],
    plot_df["actual_volatility"],
    color="#163A5F",
    linewidth=1.25,
    label="Observed volatility"
)

ax.plot(
    plot_df["target_date"],
    plot_df["predicted_volatility"],
    color="#E67E22",
    linewidth=1.35,
    linestyle="--",
    label="ARIMA-GARCH forecast"
)


# ------------------------------------------------------------
# 7. Use symlog scale for better visibility
# ------------------------------------------------------------
ax.set_yscale(
    "symlog",
    linthresh=0.1
)


# ------------------------------------------------------------
# 8. Titles and labels
# ------------------------------------------------------------
ax.set_title(
    "ARIMA-GARCH Actual and One-Step-Ahead Forecast Volatility",
    fontsize=15,
    fontweight="bold",
    pad=14
)

ax.set_xlabel(
    "Target Date",
    fontsize=11
)

ax.set_ylabel(
    r"Exchange-Rate Volatility ($r_t^2$)",
    fontsize=11
)


# ------------------------------------------------------------
# 9. Date formatting
# ------------------------------------------------------------
ax.xaxis.set_major_locator(
    mdates.YearLocator(1)
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%Y")
)

plt.setp(
    ax.get_xticklabels(),
    rotation=45,
    ha="right"
)


# ------------------------------------------------------------
# 10. Grid and axes styling
# ------------------------------------------------------------
ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.30
)

ax.grid(
    axis="x",
    visible=False
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# ------------------------------------------------------------
# 11. Legend
# ------------------------------------------------------------
ax.legend(
    frameon=False,
    loc="upper left",
    fontsize=10
)


# ------------------------------------------------------------
# 12. Layout
# ------------------------------------------------------------
fig.tight_layout()


# ------------------------------------------------------------
# 13. Save Figure 4.8
# ------------------------------------------------------------
figure_path = (
    FIGURES_DIR
    / "figure_4_8_arima_garch_actual_vs_forecast.png"
)

fig.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ------------------------------------------------------------
# 14. Print verification
# ------------------------------------------------------------
print("\n" + "=" * 75)
print("[FIG 4.8 HERE]")
print("=" * 75)

print(
    "ARIMA-GARCH Actual and One-Step-Ahead "
    "Forecast Volatility"
)

print("\nSource:")
print(FORECAST_PATH)

print(
    f"\nObservations plotted: "
    f"{len(plot_df):,}"
)

print(
    f"Forecast period: "
    f"{plot_df['target_date'].min().date()} "
    f"to "
    f"{plot_df['target_date'].max().date()}"
)

print("\nSaved to:")
print(figure_path)

print("\nCompleted successfully.")