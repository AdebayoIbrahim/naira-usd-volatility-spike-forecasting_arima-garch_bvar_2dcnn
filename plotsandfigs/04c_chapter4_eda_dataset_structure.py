#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 10:55:00 2026

@author: setup
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CHAPTER 4 - EDA
# [TABLE 4.2 HERE]
# Sample of the Final Forecasting Dataset
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
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Load dataset
# ------------------------------------------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["date", "target_date"]
)

df = (
    df.sort_values("date")
      .reset_index(drop=True)
)


# ------------------------------------------------------------
# 3. Columns to display
# ------------------------------------------------------------
display_columns = [
    "date",
    "target_date",
    "fx_log_return",
    "fx_return_squared",
    "brent_log_return",
    "reserves_log_change",
    "target_volatility",
    "target_spike",
    "gap_days",
    "long_gap",
    "segment_id",
]


missing_columns = [
    col
    for col in display_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ------------------------------------------------------------
# 4. Build representative sample
# ------------------------------------------------------------

# A. Early ordinary observations
early_rows = df.head(5)

# B. Some target spikes
spike_rows = (
    df[df["target_spike"] == 1]
    .head(5)
)

# C. Rows around long-gap boundaries
long_gap_indices = (
    df.index[df["long_gap"] == 1]
    .tolist()
)

gap_indices = []

for idx in long_gap_indices[:3]:
    start = max(0, idx - 2)
    end = min(len(df), idx + 3)
    gap_indices.extend(range(start, end))

gap_rows = df.loc[
    sorted(set(gap_indices))
]

# D. Combine and remove duplicates
sample = pd.concat(
    [
        early_rows,
        spike_rows,
        gap_rows
    ],
    axis=0
).drop_duplicates()

# Keep a manageable number of rows
sample = (
    sample
    .sort_values("date")
    .head(20)
    [display_columns]
    .copy()
)


# ------------------------------------------------------------
# 5. Format for presentation
# ------------------------------------------------------------
sample["date"] = sample["date"].dt.strftime("%Y-%m-%d")
sample["target_date"] = sample["target_date"].dt.strftime(
    "%Y-%m-%d"
)

numeric_columns = [
    "fx_log_return",
    "fx_return_squared",
    "brent_log_return",
    "reserves_log_change",
    "target_volatility",
]

sample[numeric_columns] = (
    sample[numeric_columns]
    .round(6)
)

sample["target_spike"] = sample["target_spike"].astype("Int64")
sample["long_gap"] = sample["long_gap"].astype("Int64")
sample["segment_id"] = sample["segment_id"].astype("Int64")


# ------------------------------------------------------------
# 6. Save CSV
# ------------------------------------------------------------
csv_path = (
    RESULTS_DIR
    / "table_4_2_sample_forecasting_dataset.csv"
)

sample.to_csv(
    csv_path,
    index=False
)


# ------------------------------------------------------------
# 7. Save LaTeX
# ------------------------------------------------------------
latex_path = (
    RESULTS_DIR
    / "table_4_2_sample_forecasting_dataset.tex"
)

latex_table = sample.to_latex(
    index=False,
    escape=False,
    caption="Sample of the Final Forecasting Dataset",
    label="tab:4.2"
)

latex_path.write_text(
    latex_table,
    encoding="utf-8"
)


# ------------------------------------------------------------
# 8. Print information for checking
# ------------------------------------------------------------
print("\n" + "=" * 65)
print("[TABLE 4.2 HERE]")
print("=" * 65)

print("Sample of the Final Forecasting Dataset")

print("\nDataset shape:")
print(df.shape)

print("\nColumns shown:")
for col in display_columns:
    print(f" - {col}")

print("\nSample observations:")
print(sample.to_string(index=False))

print("\nSaved CSV:")
print(csv_path)

print("\nSaved LaTeX:")
print(latex_path)

print("\nCompleted successfully.")