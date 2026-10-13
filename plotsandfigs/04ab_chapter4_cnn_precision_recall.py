#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 23:03:22 2026

@author: setup
"""

"""
FIGURE 4.15
Precision-Recall Curve for the Selected Multivariate 2D-CNN

Source:
results/cnn_A_final_test_predictions.csv

Output:
figures/figure_4_15_cnn_precision_recall.png
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score
)


# ============================================================
# 1. PATHS
# ============================================================

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_FILE = RESULTS_DIR / "cnn_A_final_test_predictions.csv"
OUTPUT_FILE = FIGURES_DIR / "figure_4_15_cnn_precision_recall.png"


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
# 5. PRECISION-RECALL CURVE
# ============================================================

precision, recall, thresholds = precision_recall_curve(
    y_true,
    y_score
)

pr_auc = average_precision_score(
    y_true,
    y_score
)

baseline = y_true.mean()


# ============================================================
# 6. PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(8, 7))

ax.plot(
    recall,
    precision,
    linewidth=2,
    label=f"CNN-A (PR-AUC = {pr_auc:.4f})"
)

# Class-prevalence baseline
ax.axhline(
    baseline,
    linestyle="--",
    linewidth=1,
    label=f"Spike prevalence = {baseline:.3f}"
)


# ============================================================
# 7. AXES AND LABELS
# ============================================================

ax.set_xlabel(
    "Recall",
    fontsize=11
)

ax.set_ylabel(
    "Precision",
    fontsize=11
)

ax.set_title(
    "Precision-Recall Curve for the Selected Multivariate 2D-CNN",
    fontsize=13,
    pad=12
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)

ax.grid(
    True,
    linestyle="--",
    linewidth=0.5,
    alpha=0.35
)

ax.legend(
    loc="lower left",
    frameon=True
)

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
print("FIGURE 4.15 CREATED")
print("=" * 75)

print(f"Observations       : {len(y_true)}")
print(f"Actual spikes      : {y_true.sum()}")
print(f"Spike prevalence   : {baseline:.6f}")
print(f"PR-AUC             : {pr_auc:.6f}")
print(f"Output             : {OUTPUT_FILE}")