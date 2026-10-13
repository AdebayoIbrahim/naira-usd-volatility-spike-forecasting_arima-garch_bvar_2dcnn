#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 23:01:46 2026

@author: setup
"""

"""
FIGURE 4.14
ROC Curve for the Selected Multivariate 2D-CNN

Source:
results/cnn_A_final_test_predictions.csv

Output:
figures/figure_4_14_cnn_roc_curve.png
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


# ============================================================
# 1. PATHS
# ============================================================

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_FILE = RESULTS_DIR / "cnn_A_final_test_predictions.csv"
OUTPUT_FILE = FIGURES_DIR / "figure_4_14_cnn_roc_curve.png"


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
# 3. VERIFY COLUMNS
# ============================================================

required_columns = [
    "actual_spike",
    "predicted_spike_probability"
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

y_true = pd.to_numeric(
    df["actual_spike"],
    errors="coerce"
)

y_score = pd.to_numeric(
    df["predicted_spike_probability"],
    errors="coerce"
)

valid = y_true.notna() & y_score.notna()

y_true = y_true[valid].astype(int)
y_score = y_score[valid].astype(float)


# ============================================================
# 5. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_true,
    y_score
)

auc_value = roc_auc_score(
    y_true,
    y_score
)


# ============================================================
# 6. PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(8, 7))

ax.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"CNN-A (ROC-AUC = {auc_value:.4f})"
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
    label="Random classifier"
)

ax.set_xlabel(
    "False Positive Rate",
    fontsize=11
)

ax.set_ylabel(
    "True Positive Rate",
    fontsize=11
)

ax.set_title(
    "ROC Curve for the Selected Multivariate 2D-CNN",
    fontsize=13,
    pad=12
)

ax.grid(
    True,
    linestyle="--",
    linewidth=0.5,
    alpha=0.35
)

ax.legend(
    loc="lower right",
    frameon=True
)

plt.tight_layout()


# ============================================================
# 7. SAVE
# ============================================================

fig.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 8. REPORT
# ============================================================

print("\n" + "=" * 75)
print("FIGURE 4.14 CREATED")
print("=" * 75)

print(f"Observations : {len(y_true)}")
print(f"ROC-AUC      : {auc_value:.6f}")
print(f"Output       : {OUTPUT_FILE}")