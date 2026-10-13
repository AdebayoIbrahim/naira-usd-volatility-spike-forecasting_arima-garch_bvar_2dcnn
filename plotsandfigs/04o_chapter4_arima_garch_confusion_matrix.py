#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 05:49:10 2026

@author: setup
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


# ============================================================
# CHAPTER 4 - ARIMA-GARCH RESULTS
# [FIG 4.10 HERE]
# ARIMA-GARCH Confusion Matrix
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
# 4. Check required columns
# ------------------------------------------------------------
required_columns = [
    "actual_spike",
    "predicted_spike"
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
plot_df = (
    df[
        [
            "actual_spike",
            "predicted_spike"
        ]
    ]
    .dropna()
    .copy()
)

plot_df["actual_spike"] = (
    plot_df["actual_spike"].astype(int)
)

plot_df["predicted_spike"] = (
    plot_df["predicted_spike"].astype(int)
)


# ------------------------------------------------------------
# 6. Calculate confusion matrix
# ------------------------------------------------------------
cm = confusion_matrix(
    plot_df["actual_spike"],
    plot_df["predicted_spike"],
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()


# ------------------------------------------------------------
# 7. Create percentage matrix for annotation
# ------------------------------------------------------------
total = cm.sum()

cm_percent = (
    cm / total * 100
)


# ------------------------------------------------------------
# 8. Create formatted annotations
# ------------------------------------------------------------
annotations = cm.astype(str)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        annotations[i, j] = (
            f"{cm[i, j]:,}\n"
            f"({cm_percent[i, j]:.1f}%)"
        )


# ------------------------------------------------------------
# 9. Create figure
# ------------------------------------------------------------
fig, ax = plt.subplots(
    figsize=(8.5, 7),
    dpi=150
)


# ------------------------------------------------------------
# 10. Heatmap
# ------------------------------------------------------------
sns.heatmap(
    cm,
    annot=annotations,
    fmt="",
    cmap="Blues",
    linewidths=1.2,
    linecolor="white",
    cbar=False,
    square=True,
    annot_kws={
        "fontsize": 14,
        "fontweight": "bold"
    },
    ax=ax
)


# ------------------------------------------------------------
# 11. Labels
# ------------------------------------------------------------
ax.set_xticklabels(
    [
        "Predicted Non-Spike",
        "Predicted Spike"
    ],
    fontsize=10
)

ax.set_yticklabels(
    [
        "Actual Non-Spike",
        "Actual Spike"
    ],
    fontsize=10,
    rotation=90
)


# ------------------------------------------------------------
# 12. Titles
# ------------------------------------------------------------
ax.set_title(
    "ARIMA-GARCH Volatility-Spike Confusion Matrix",
    fontsize=15,
    fontweight="bold",
    pad=14
)

ax.set_xlabel(
    "Predicted Class",
    fontsize=11,
    labelpad=10
)

ax.set_ylabel(
    "Actual Class",
    fontsize=11,
    labelpad=10
)


# ------------------------------------------------------------
# 13. Small explanatory subtitle
# ------------------------------------------------------------
fig.text(
    0.5,
    0.02,
    (
        f"TN = {tn:,}   FP = {fp:,}   "
        f"FN = {fn:,}   TP = {tp:,}"
    ),
    ha="center",
    fontsize=10
)


# ------------------------------------------------------------
# 14. Layout
# ------------------------------------------------------------
fig.tight_layout(
    rect=[0, 0.06, 1, 1]
)


# ------------------------------------------------------------
# 15. Save figure
# ------------------------------------------------------------
figure_path = (
    FIGURES_DIR
    / "figure_4_10_arima_garch_confusion_matrix.png"
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
print("[FIG 4.10 HERE]")
print("=" * 75)

print(
    "ARIMA-GARCH Volatility-Spike Confusion Matrix"
)

print("\nSource:")
print(FORECAST_PATH)

print("\nConfusion Matrix:")
print(cm)

print(
    f"\nTN = {tn:,}"
    f"\nFP = {fp:,}"
    f"\nFN = {fn:,}"
    f"\nTP = {tp:,}"
)

print("\nSaved to:")
print(figure_path)

print("\nCompleted successfully.")