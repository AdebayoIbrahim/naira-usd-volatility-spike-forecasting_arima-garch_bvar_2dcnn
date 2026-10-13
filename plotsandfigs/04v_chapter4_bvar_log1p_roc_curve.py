#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 16:29:18 2026

@author: setup
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score


# ============================================================
# CHAPTER 4 - BVAR RESULTS
# [FIG 4.12 HERE]
# BVAR Log1p ROC Curve
# ============================================================


# ------------------------------------------------------------
# 1. Project paths
# ------------------------------------------------------------
PROJECT_ROOT = Path.cwd()

FORECAST_PATH = (
    PROJECT_ROOT
    / "results"
    / "bvar8_final_log1p_common_sample_forecasts.csv"
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
# 3. Load final BVAR-log1p forecasts
# ------------------------------------------------------------
df = pd.read_csv(FORECAST_PATH)


# ------------------------------------------------------------
# 4. Required columns
# ------------------------------------------------------------
required_columns = [
    "actual_spike",
    "predicted_spike_probability"
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
# 5. Prepare ROC data
# ------------------------------------------------------------
roc_df = (
    df[
        [
            "actual_spike",
            "predicted_spike_probability"
        ]
    ]
    .copy()
)

roc_df["actual_spike"] = pd.to_numeric(
    roc_df["actual_spike"],
    errors="coerce"
)

roc_df["predicted_spike_probability"] = pd.to_numeric(
    roc_df["predicted_spike_probability"],
    errors="coerce"
)

roc_df = (
    roc_df
    .replace([float("inf"), float("-inf")], pd.NA)
    .dropna()
)


y_true = roc_df["actual_spike"].astype(int)
y_score = roc_df["predicted_spike_probability"].astype(float)


# ------------------------------------------------------------
# 6. ROC curve and AUC
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


# ------------------------------------------------------------
# 8. ROC curve
# ------------------------------------------------------------
ax.plot(
    fpr,
    tpr,
    color="#7A3E9D",
    linewidth=2.5,
    label=f"BVAR Log1p (AUC = {roc_auc:.3f})"
)


# ------------------------------------------------------------
# 9. Random classifier reference
# ------------------------------------------------------------
ax.plot(
    [0, 1],
    [0, 1],
    color="#9E9E9E",
    linewidth=1.2,
    linestyle="--",
    label="Random classification"
)


# ------------------------------------------------------------
# 10. Titles and labels
# ------------------------------------------------------------
ax.set_title(
    "ROC Curve for BVAR Log1p Volatility-Spike Detection",
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
# 11. Axis limits
# ------------------------------------------------------------
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)


# ------------------------------------------------------------
# 12. Grid
# ------------------------------------------------------------
ax.grid(
    linestyle="--",
    linewidth=0.7,
    alpha=0.30
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# ------------------------------------------------------------
# 13. Legend
# ------------------------------------------------------------
ax.legend(
    frameon=False,
    loc="lower right",
    fontsize=10
)


# ------------------------------------------------------------
# 14. Layout
# ------------------------------------------------------------
fig.tight_layout()


# ------------------------------------------------------------
# 15. Save Figure 4.12
# ------------------------------------------------------------
figure_path = (
    FIGURES_DIR
    / "figure_4_12_bvar_log1p_roc_curve.png"
)

fig.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ------------------------------------------------------------
# 16. Print verification
# ------------------------------------------------------------
print("\n" + "=" * 75)
print("[FIG 4.12 HERE]")
print("=" * 75)

print(
    "ROC Curve for BVAR Log1p "
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