#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 05:46:37 2026

@author: setup
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score


# ============================================================
# CHAPTER 4 - ARIMA-GARCH RESULTS
# [FIG 4.9 HERE]
# ARIMA-GARCH ROC Curve
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
# 3. Load common-sample forecasts
# ------------------------------------------------------------
df = pd.read_csv(
    FORECAST_PATH
)


# ------------------------------------------------------------
# 4. Required columns
# ------------------------------------------------------------
required_columns = [
    "actual_spike",
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
# 5. Prepare data
# ------------------------------------------------------------
roc_df = (
    df[
        [
            "actual_spike",
            "predicted_volatility"
        ]
    ]
    .dropna()
    .copy()
)

roc_df["actual_spike"] = (
    roc_df["actual_spike"]
    .astype(int)
)

roc_df["predicted_volatility"] = (
    pd.to_numeric(
        roc_df["predicted_volatility"],
        errors="coerce"
    )
)

roc_df = roc_df.dropna()


y_true = roc_df["actual_spike"]
y_score = roc_df["predicted_volatility"]


# ------------------------------------------------------------
# 6. Calculate ROC curve and AUC
# ------------------------------------------------------------
fpr, tpr, thresholds = roc_curve(
    y_true,
    y_score
)

roc_auc = roc_auc_score(
    y_true,
    y_score
)


# ------------------------------------------------------------
# 7. Create figure
# ------------------------------------------------------------
fig, ax = plt.subplots(
    figsize=(8.5, 7),
    dpi=150
)

ax.plot(
    fpr,
    tpr,
    color="#176B87",
    linewidth=2.5,
    label=f"ARIMA-GARCH (AUC = {roc_auc:.3f})"
)

ax.plot(
    [0, 1],
    [0, 1],
    color="#A0A0A0",
    linewidth=1.2,
    linestyle="--",
    label="Random classification"
)


# ------------------------------------------------------------
# 8. Titles and labels
# ------------------------------------------------------------
ax.set_title(
    "ROC Curve for ARIMA-GARCH Volatility-Spike Detection",
    fontsize=15,
    fontweight="bold",
    pad=14
)

ax.set_xlabel(
    "False Positive Rate",
    fontsize=11
)

ax.set_ylabel(
    "True Positive Rate",
    fontsize=11
)


# ------------------------------------------------------------
# 9. Grid and axes
# ------------------------------------------------------------
ax.grid(
    linestyle="--",
    linewidth=0.7,
    alpha=0.30
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# ------------------------------------------------------------
# 10. Legend
# ------------------------------------------------------------
ax.legend(
    frameon=False,
    loc="lower right",
    fontsize=10
)


# ------------------------------------------------------------
# 11. Axis limits
# ------------------------------------------------------------
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)


# ------------------------------------------------------------
# 12. Layout
# ------------------------------------------------------------
fig.tight_layout()


# ------------------------------------------------------------
# 13. Save Figure 4.9
# ------------------------------------------------------------
figure_path = (
    FIGURES_DIR
    / "figure_4_9_arima_garch_roc_curve.png"
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
print("[FIG 4.9 HERE]")
print("=" * 75)

print(
    "ROC Curve for ARIMA-GARCH "
    "Volatility-Spike Detection"
)

print("\nSource:")
print(FORECAST_PATH)

print(
    f"\nObservations used: "
    f"{len(roc_df):,}"
)

print(
    f"ROC-AUC: "
    f"{roc_auc:.6f}"
)

print("\nSaved to:")
print(figure_path)

print("\nCompleted successfully.")