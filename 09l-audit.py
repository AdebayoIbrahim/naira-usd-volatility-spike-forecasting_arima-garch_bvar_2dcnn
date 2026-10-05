#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 19:15:41 2026

@author: setup
"""

# ============================================================
# STEP 09M: BVAR(8) SPIKE-SCORE AUDIT
# ============================================================
#
# Purpose:
#   The BVAR produces TWO continuous quantities:
#
#   1. forecast_volatility
#   2. predicted_spike_probability
#
#   This script calculates ROC-AUC using BOTH scores.
#
#   This is an audit only. It does not change the model,
#   threshold, probability cutoff, or test predictions.
# ============================================================

import pandas as pd
from sklearn.metrics import roc_auc_score


# ============================================================
# 1. SETTINGS
# ============================================================

INPUT_FILE = (
    "results/"
    "bvar8_spike_walk_forward_forecasts.csv"
)


# ============================================================
# 2. LOAD FORECASTS
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)

print("=" * 90)
print("STEP 09M: BVAR(8) SPIKE-SCORE AUDIT")
print("=" * 90)

print(
    f"\nForecast observations: {len(df):,}"
)


# ============================================================
# 3. REQUIRED COLUMNS
# ============================================================

required_columns = [
    "actual_spike",
    "forecast_volatility",
    "predicted_spike_probability",
    "predicted_spike",
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        "Missing required columns: "
        f"{missing}"
    )


# ============================================================
# 4. BASIC CHECKS
# ============================================================

actual = df[
    "actual_spike"
].astype(int)

volatility_score = df[
    "forecast_volatility"
].astype(float)

probability_score = df[
    "predicted_spike_probability"
].astype(float)

predicted_binary = df[
    "predicted_spike"
].astype(int)


if actual.nunique() < 2:
    raise ValueError(
        "AUC cannot be computed because the actual spike "
        "labels contain only one class."
    )


# ============================================================
# 5. CALCULATE BOTH AUCs
# ============================================================

auc_volatility = roc_auc_score(
    actual,
    volatility_score
)

auc_probability = roc_auc_score(
    actual,
    probability_score
)


# ============================================================
# 6. CLASSIFICATION SUMMARY
# ============================================================

actual_spikes = int(
    actual.sum()
)

predicted_spikes = int(
    predicted_binary.sum()
)

actual_spike_rate = (
    actual_spikes
    / len(df)
)

predicted_spike_rate = (
    predicted_spikes
    / len(df)
)

mean_probability = (
    probability_score.mean()
)

min_probability = (
    probability_score.min()
)

max_probability = (
    probability_score.max()
)


# ============================================================
# 7. PRINT RESULTS
# ============================================================

print("\n" + "=" * 90)
print("AUC AUDIT")
print("=" * 90)

print(
    f"\nROC-AUC using forecast_volatility: "
    f"{auc_volatility:.10f}"
)

print(
    f"ROC-AUC using predicted_spike_probability: "
    f"{auc_probability:.10f}"
)


print("\nSPIKE SUMMARY")
print("--------------------------------")

print(
    f"Actual spikes: "
    f"{actual_spikes:,}"
)

print(
    f"Actual spike rate: "
    f"{actual_spike_rate:.4%}"
)

print(
    f"Predicted spikes at 0.50 cutoff: "
    f"{predicted_spikes:,}"
)

print(
    f"Predicted spike rate: "
    f"{predicted_spike_rate:.4%}"
)

print(
    f"Mean predicted probability: "
    f"{mean_probability:.6f}"
)

print(
    f"Minimum predicted probability: "
    f"{min_probability:.6f}"
)

print(
    f"Maximum predicted probability: "
    f"{max_probability:.6f}"
)


# ============================================================
# 8. PROBABILITY RANGE COUNTS
# ============================================================

print("\nPROBABILITY DISTRIBUTION")
print("--------------------------------")

bins = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
]

counts = pd.cut(
    probability_score,
    bins=bins,
    include_lowest=True,
    right=True
).value_counts(
    sort=False
)

print(
    counts.to_string()
)


print(
    "\nNo model settings were changed by this audit."
)
