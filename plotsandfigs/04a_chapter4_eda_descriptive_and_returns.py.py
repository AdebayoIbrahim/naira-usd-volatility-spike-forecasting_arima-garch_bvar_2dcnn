#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 10:27:17 2026

@author: setup
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CHAPTER 4 - EDA
# [TABLE 4.1 HERE]
# Descriptive statistics of the main transformed variables
#
# [FIG 4.1 HERE]
# Daily Naira-US dollar log returns
# ============================================================


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------
PROJECT_ROOT = Path.cwd()

DATA_PATH = (
    PROJECT_ROOT
    / "data_cleaned"
    / "forecasting_dataset_2006_2025_spike_gap_target_safe.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Load data
# ------------------------------------------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}\n\n"
        "Make sure Spyder is using the repository root as its "
        "working directory."
    )

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)


# ------------------------------------------------------------
# 3. Variables for descriptive statistics
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
        f"Missing required columns: {missing_columns}"
    )


# ------------------------------------------------------------
# 4. Create descriptive statistics
# ------------------------------------------------------------
rows = []

for label, column in variables.items():
    series = df[column].dropna()

    rows.append(
        {
            "Variable": label,
            "N": series.size,
            "Mean": series.mean(),
            "Std. Dev.": series.std(),
            "Minimum": series.min(),
            "25th Percentile": series.quantile(0.25),
            "Median": series.median(),
            "75th Percentile": series.quantile(0.75),
            "Maximum": series.max(),
            "Skewness": series.skew(),
            "Kurtosis": series.kurt(),
        }
    )

table_4_1 = pd.DataFrame(rows)

# Round only for presentation
table_4_1_display = table_4_1.copy()

numeric_columns = table_4_1_display.columns.drop(
    ["Variable", "N"]
)

table_4_1_display[numeric_columns] = (
    table_4_1_display[numeric_columns]
    .round(6)
)

# N should remain integer
table_4_1_display["N"] = table_4_1_display["N"].astype(int)


# ------------------------------------------------------------
# 5. Save Table 4.1
# ------------------------------------------------------------
table_csv = RESULTS_DIR / "table_4_1_descriptive_statistics.csv"
table_tex = RESULTS_DIR / "table_4_1_descriptive_statistics.tex"

table_4_1_display.to_csv(table_csv, index=False)

# LaTeX version for thesis use
latex_table = table_4_1_display.to_latex(
    index=False,
    escape=False,
    caption="Descriptive Statistics of the Main Transformed Variables",
    label="tab:4.1",
    float_format="%.6f"
)

table_tex.write_text(latex_table, encoding="utf-8")


# ------------------------------------------------------------
# 6. Figure 4.1 - FX log returns
# ------------------------------------------------------------
fx = df.dropna(subset=["fx_log_return"]).copy()

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    fx["date"],
    fx["fx_log_return"],
    linewidth=0.8
)

ax.axhline(0, linewidth=0.8)

ax.set_title(
    "Daily Naira-US Dollar Log Returns, 2006–2025"
)

ax.set_xlabel("Date")
ax.set_ylabel("Log Return (percentage points)")

fig.tight_layout()

figure_4_1 = FIGURES_DIR / "figure_4_1_fx_log_returns.png"

fig.savefig(
    figure_4_1,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ------------------------------------------------------------
# 7. Output confirmation
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("CHAPTER 4 EDA OUTPUTS")
print("=" * 60)

print("\n[TABLE 4.1 HERE]")
print("Descriptive Statistics of the Main Transformed Variables")
print(f"Saved to: {table_csv}")
print(f"LaTeX saved to: {table_tex}")

print("\n[FIG 4.1 HERE]")
print("Daily Naira-US Dollar Log Returns, 2006–2025")
print(f"Saved to: {figure_4_1}")

print("\nCompleted successfully.")