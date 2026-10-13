#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 20:16:49 2026

@author: setup
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CHAPTER 4 - EDA
# [FIG 4.7 HERE]
# Correlation Matrix of Transformed Variables
# ============================================================


# ------------------------------------------------------------
# 1. Project paths
# ------------------------------------------------------------
PROJECT_ROOT = Path.cwd()

DATA_PATH = (
    PROJECT_ROOT
    / "data_cleaned"
    / "forecasting_dataset_2006_2025_spike_gap_target_safe.csv"
)

FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Check dataset
# ------------------------------------------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )


# ------------------------------------------------------------
# 3. Load data
# ------------------------------------------------------------
df = pd.read_csv(
    DATA_PATH,
    parse_dates=["date"]
)

df = (
    df.sort_values("date")
      .reset_index(drop=True)
)


# ------------------------------------------------------------
# 4. Variables for correlation analysis
# ------------------------------------------------------------
variables = {
    "FX Log Return": "fx_log_return",
    "FX Squared Return": "fx_return_squared",
    "Brent Log Return": "brent_log_return",
    "Reserves Log Change": "reserves_log_change",
}


missing_columns = [
    column
    for column in variables.values()
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}\n\n"
        f"Available columns:\n{list(df.columns)}"
    )


# ------------------------------------------------------------
# 5. Prepare correlation dataset
# ------------------------------------------------------------
corr_df = (
    df[list(variables.values())]
    .dropna()
    .rename(columns={
        column: label
        for label, column in variables.items()
    })
)


# ------------------------------------------------------------
# 6. Calculate Pearson correlation matrix
# ------------------------------------------------------------
corr_matrix = corr_df.corr(method="pearson")


# ------------------------------------------------------------
# 7. Save correlation matrix
# ------------------------------------------------------------
csv_path = (
    RESULTS_DIR
    / "figure_4_7_correlation_matrix.csv"
)

corr_matrix.round(6).to_csv(csv_path)


# ------------------------------------------------------------
# 8. Create heatmap
# ------------------------------------------------------------
plt.figure(
    figsize=(9, 7),
    dpi=150
)

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
    linewidths=0.6,
    linecolor="white",
    cbar_kws={
        "label": "Pearson Correlation"
    },
    annot_kws={
        "fontsize": 10
    }
)

plt.title(
    "Correlation Among Transformed Study Variables",
    fontsize=15,
    fontweight="bold",
    pad=14
)

plt.xticks(
    rotation=35,
    ha="right",
    fontsize=10
)

plt.yticks(
    rotation=0,
    fontsize=10
)

plt.tight_layout()


# ------------------------------------------------------------
# 9. Save figure
# ------------------------------------------------------------
figure_path = (
    FIGURES_DIR
    / "figure_4_7_correlation_heatmap.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 10. Print verification
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("[FIG 4.7 HERE]")
print("=" * 70)

print(
    "Correlation Among Transformed Study Variables"
)

print(
    f"\nComplete observations used: "
    f"{len(corr_df):,}"
)

print("\nCorrelation matrix:")
print(
    corr_matrix.round(4).to_string()
)

print("\nCorrelation matrix CSV saved to:")
print(csv_path)

print("\nFigure saved to:")
print(figure_path)

print("\nCompleted successfully.")