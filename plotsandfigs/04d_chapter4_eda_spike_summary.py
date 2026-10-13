#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 11:04:26 2026

@author: setup
"""

from pathlib import Path
import pandas as pd


# ============================================================
# CHAPTER 4 - EDA
# [TABLE 4.3 HERE]
# Volatility Spike Distribution by Forecasting Period
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
# 2. Load final forecasting dataset
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
# 3. Required columns
# ------------------------------------------------------------
required_columns = [
    "target_date",
    "target_spike",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ------------------------------------------------------------
# 4. Keep only valid forecasting targets
# ------------------------------------------------------------
df = df.dropna(
    subset=["target_date", "target_spike"]
).copy()


# ------------------------------------------------------------
# 5. Assign forecasting period
# ------------------------------------------------------------
df["period"] = df["target_date"].apply(
    lambda date: (
        "Training (2006–2020)"
        if date.year <= 2020
        else "Test (2021–2025)"
    )
)


# ------------------------------------------------------------
# 6. Calculate spike summary
# ------------------------------------------------------------
summary = (
    df.groupby("period", sort=False)
      .agg(
          observations=("target_spike", "size"),
          spikes=("target_spike", "sum")
      )
      .reset_index()
)

summary["non_spikes"] = (
    summary["observations"] - summary["spikes"]
)

summary["spike_rate_percent"] = (
    summary["spikes"]
    / summary["observations"]
    * 100
)

summary["non_spike_rate_percent"] = (
    summary["non_spikes"]
    / summary["observations"]
    * 100
)


# ------------------------------------------------------------
# 7. Add overall row
# ------------------------------------------------------------
overall = pd.DataFrame(
    {
        "period": ["Overall"],
        "observations": [len(df)],
        "spikes": [df["target_spike"].sum()],
        "non_spikes": [
            len(df) - df["target_spike"].sum()
        ],
        "spike_rate_percent": [
            df["target_spike"].mean() * 100
        ],
        "non_spike_rate_percent": [
            (1 - df["target_spike"].mean()) * 100
        ],
    }
)

summary = pd.concat(
    [summary, overall],
    ignore_index=True
)


# ------------------------------------------------------------
# 8. Format table
# ------------------------------------------------------------
summary["observations"] = (
    summary["observations"].astype(int)
)

summary["spikes"] = (
    summary["spikes"].astype(int)
)

summary["non_spikes"] = (
    summary["non_spikes"].astype(int)
)

summary[
    [
        "spike_rate_percent",
        "non_spike_rate_percent",
    ]
] = summary[
    [
        "spike_rate_percent",
        "non_spike_rate_percent",
    ]
].round(2)


# ------------------------------------------------------------
# 9. Save CSV
# ------------------------------------------------------------
csv_path = (
    RESULTS_DIR
    / "table_4_3_volatility_spike_distribution.csv"
)

summary.to_csv(
    csv_path,
    index=False
)


# ------------------------------------------------------------
# 10. Save LaTeX
# ------------------------------------------------------------
latex_path = (
    RESULTS_DIR
    / "table_4_3_volatility_spike_distribution.tex"
)

latex_table = summary.to_latex(
    index=False,
    escape=False,
    caption=(
        "Distribution of Volatility Spikes "
        "Across Forecasting Periods"
    ),
    label="tab:4.3"
)

latex_path.write_text(
    latex_table,
    encoding="utf-8"
)


# ------------------------------------------------------------
# 11. Print results
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("[TABLE 4.3 HERE]")
print("=" * 70)

print(
    "Distribution of Volatility Spikes "
    "Across Forecasting Periods"
)

print("\n")
print(summary.to_string(index=False))

print("\nCSV saved to:")
print(csv_path)

print("\nLaTeX saved to:")
print(latex_path)

print("\nCompleted successfully.")