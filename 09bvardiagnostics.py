#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 19:37:13 2026

@author: setup
"""

import pandas as pd
import numpy as np

# ============================================================
# STEP 09P: BVAR PROBABILITY / FORECAST DIAGNOSTIC
# ============================================================

INPUT_FILE = "results/bvar8_final_lambda01_common_sample_forecasts.csv"

df = pd.read_csv(INPUT_FILE)

print("=" * 90)
print("STEP 09P: BVAR PROBABILITY / FORECAST DIAGNOSTIC")
print("=" * 90)

print("\nColumns available:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 1. IDENTIFY IMPORTANT COLUMNS
# ------------------------------------------------------------

prob_col = "predicted_spike_probability"
vol_col = "forecast_volatility"

# Try to locate actual spike column automatically
possible_actual = [
    "actual_spike",
    "target_spike",
    "actual_target_spike"
]

actual_col = next(
    (c for c in possible_actual if c in df.columns),
    None
)

if actual_col is None:
    raise ValueError(
        f"Could not find actual spike column. Available columns: {df.columns.tolist()}"
    )


# ------------------------------------------------------------
# 2. BASIC DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("1. PROBABILITY DISTRIBUTION")
print("=" * 90)

p = df[prob_col].dropna()

print(f"Number of probabilities: {len(p)}")
print(f"Minimum: {p.min():.6f}")
print(f"25th percentile: {p.quantile(0.25):.6f}")
print(f"Median: {p.median():.6f}")
print(f"Mean: {p.mean():.6f}")
print(f"75th percentile: {p.quantile(0.75):.6f}")
print(f"Maximum: {p.max():.6f}")


# ------------------------------------------------------------
# 3. HOW MANY ARE CLOSE TO 0.50?
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("2. PROBABILITIES AROUND THE 0.50 CUTOFF")
print("=" * 90)

ranges = [
    ("< 0.30", p < 0.30),
    ("0.30 - 0.40", (p >= 0.30) & (p < 0.40)),
    ("0.40 - 0.45", (p >= 0.40) & (p < 0.45)),
    ("0.45 - 0.50", (p >= 0.45) & (p < 0.50)),
    ("0.50 - 0.55", (p >= 0.50) & (p < 0.55)),
    ("0.55 - 0.60", (p >= 0.55) & (p < 0.60)),
    ("0.60 - 0.70", (p >= 0.60) & (p < 0.70)),
    (">= 0.70", p >= 0.70),
]

for label, mask in ranges:
    print(f"{label:15s}: {mask.sum():4d}")


# ------------------------------------------------------------
# 4. FORECAST DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("3. CONTINUOUS FORECAST DISTRIBUTION")
print("=" * 90)

v = df[vol_col].dropna()

print(f"Minimum: {v.min():.6f}")
print(f"25th percentile: {v.quantile(0.25):.6f}")
print(f"Median: {v.median():.6f}")
print(f"Mean: {v.mean():.6f}")
print(f"75th percentile: {v.quantile(0.75):.6f}")
print(f"95th percentile: {v.quantile(0.95):.6f}")
print(f"Maximum: {v.max():.6f}")


# ------------------------------------------------------------
# 5. COMPARE FORECASTS ON ACTUAL SPIKE VS NON-SPIKE DAYS
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("4. FORECAST SEPARATION")
print("=" * 90)

grouped = df.groupby(actual_col).agg(
    mean_probability=(prob_col, "mean"),
    median_probability=(prob_col, "median"),
    mean_forecast=(vol_col, "mean"),
    median_forecast=(vol_col, "median"),
    count=(prob_col, "size")
)

print(grouped)


# ------------------------------------------------------------
# 6. ACTUAL SPIKE RATE WITHIN PROBABILITY BINS
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("5. EMPIRICAL SPIKE RATE BY PREDICTED PROBABILITY")
print("=" * 90)

bins = [0, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1.0]
labels = [
    "<0.30",
    "0.30-0.40",
    "0.40-0.45",
    "0.45-0.50",
    "0.50-0.55",
    "0.55-0.60",
    "0.60-0.70",
    ">=0.70"
]

df["prob_bin"] = pd.cut(
    df[prob_col],
    bins=bins,
    labels=labels,
    include_lowest=True
)

calibration = df.groupby("prob_bin", observed=False).agg(
    observations=(actual_col, "size"),
    mean_predicted_probability=(prob_col, "mean"),
    actual_spike_rate=(actual_col, "mean")
)

print(calibration)


# ------------------------------------------------------------
# 7. CHECK HOW MANY PREDICTIONS CROSS 0.50
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("6. 0.50 CUTOFF CHECK")
print("=" * 90)

below = (df[prob_col] < 0.50).sum()
above = (df[prob_col] >= 0.50).sum()

print(f"Probability < 0.50 : {below}")
print(f"Probability >= 0.50: {above}")

print(f"\nPredicted spike rate: {above / len(df):.4%}")
print(f"Actual spike rate:    {df[actual_col].mean():.4%}")


# ------------------------------------------------------------
# 8. CHECK WHETHER FORECAST IS ALMOST CONSTANT
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("7. FORECAST VARIABILITY")
print("=" * 90)

print(f"Forecast standard deviation: {v.std():.6f}")
print(f"Probability standard deviation: {p.std():.6f}")

print("\nUnique probability values:")
print(p.nunique())


print("\n" + "=" * 90)
print("DIAGNOSTIC COMPLETE")
print("=" * 90)