#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 14:19:59 2026

@author: setup
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CHAPTER 4 - BVAR RESULTS
# [FIG 4.11 HERE]
# BVAR Log1p Volatility-Spike Confusion Matrix
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
df = pd.read_csv(
    FORECAST_PATH
)


# ------------------------------------------------------------
# 4. Required columns
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
# 5. Prepare classification data
# ------------------------------------------------------------
cm_df = (
    df[
        [
            "actual_spike",
            "predicted_spike"
        ]
    ]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .copy()
)

cm_df["actual_spike"] = (
    cm_df["actual_spike"]
    .astype(int)
)

cm_df["predicted_spike"] = (
    cm_df["predicted_spike"]
    .astype(int)
)


# ------------------------------------------------------------
# 6. Build confusion matrix manually
# ------------------------------------------------------------
actual = cm_df["actual_spike"].to_numpy()
predicted = cm_df["predicted_spike"].to_numpy()

tn = np.sum(
    (actual == 0) & (predicted == 0)
)

fp = np.sum(
    (actual == 0) & (predicted == 1)
)

fn = np.sum(
    (actual == 1) & (predicted == 0)
)

tp = np.sum(
    (actual == 1) & (predicted == 1)
)

cm = np.array(
    [
        [tn, fp],
        [fn, tp]
    ]
)


# ------------------------------------------------------------
# 7. Percentages
# ------------------------------------------------------------
total = cm.sum()

cm_percent = (
    cm / total * 100
)


# ------------------------------------------------------------
# 8. Annotation labels
# ------------------------------------------------------------
annotations = np.empty(
    cm.shape,
    dtype=object
)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):

        annotations[i, j] = (
            f"{cm[i, j]:,}\n"
            f"{cm_percent[i, j]:.1f}%"
        )


# ------------------------------------------------------------
# 9. Create figure
# ------------------------------------------------------------
fig, ax = plt.subplots(
    figsize=(8.5, 7),
    dpi=150
)


# ------------------------------------------------------------
# 10. Heatmap using matplotlib
# ------------------------------------------------------------
image = ax.imshow(
    cm,
    cmap="Blues",
    aspect="equal"
)


# ------------------------------------------------------------
# 11. Add annotations
# ------------------------------------------------------------
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):

        ax.text(
            j,
            i,
            annotations[i, j],
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold"
        )


# ------------------------------------------------------------
# 12. Cell borders
# ------------------------------------------------------------
ax.set_xticks(
    np.arange(-0.5, 2, 1),
    minor=True
)

ax.set_yticks(
    np.arange(-0.5, 2, 1),
    minor=True
)

ax.grid(
    which="minor",
    color="white",
    linestyle="-",
    linewidth=2
)

ax.tick_params(
    which="minor",
    bottom=False,
    left=False
)


# ------------------------------------------------------------
# 13. Axis labels
# ------------------------------------------------------------
ax.set_xticks(
    [0, 1]
)

ax.set_yticks(
    [0, 1]
)

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
    fontsize=10
)


# ------------------------------------------------------------
# 14. Title and labels
# ------------------------------------------------------------
ax.set_title(
    "BVAR Log1p Volatility-Spike Confusion Matrix",
    fontsize=15,
    fontweight="bold",
    pad=15
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
# 15. Colorbar
# ------------------------------------------------------------
cbar = fig.colorbar(
    image,
    ax=ax,
    fraction=0.046,
    pad=0.04
)

cbar.set_label(
    "Number of Observations",
    fontsize=10
)


# ------------------------------------------------------------
# 16. Additional summary
# ------------------------------------------------------------
fig.text(
    0.5,
    0.02,
    (
        f"TN = {tn:,}    "
        f"FP = {fp:,}    "
        f"FN = {fn:,}    "
        f"TP = {tp:,}"
    ),
    ha="center",
    fontsize=10
)


# ------------------------------------------------------------
# 17. Layout
# ------------------------------------------------------------
fig.tight_layout(
    rect=[0, 0.06, 1, 1]
)


# ------------------------------------------------------------
# 18. Save Figure 4.11
# ------------------------------------------------------------
figure_path = (
    FIGURES_DIR
    / "figure_4_11_bvar_log1p_confusion_matrix.png"
)

fig.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ------------------------------------------------------------
# 19. Print verification
# ------------------------------------------------------------
print("\n" + "=" * 75)
print("[FIG 4.11 HERE]")
print("=" * 75)

print(
    "BVAR Log1p Volatility-Spike "
    "Confusion Matrix"
)

print("\nSource:")
print(FORECAST_PATH)

print("\nObservations:")
print(f"{len(cm_df):,}")

print("\nConfusion matrix:")
print(cm)

print("\nTN =", tn)
print("FP =", fp)
print("FN =", fn)
print("TP =", tp)

print("\nSaved to:")
print(figure_path)

print("\nCompleted successfully.")