# -*- coding: utf-8 -*-
"""
Step 06B: Candidate spike thresholds using squared daily FX returns.

Purpose:
    Examine different percentile-based thresholds for defining
    a volatility spike.

Method:
    - Volatility proxy = squared daily log return
    - Thresholds are calculated using TRAINING DATA ONLY
    - Training period ends on 2020-12-31
"""

import os
import pandas as pd
import numpy as np


# ============================================================
# 1. FILE PATHS
# ============================================================

INPUT_FILE = "data_cleaned/daily_transformed_dataset_2006_2025.csv"
OUTPUT_FILE = "results/return_squared_threshold_candidates.csv"

TRAIN_END = "2020-12-31"

PERCENTILES = [0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99]


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("STEP 06B: CANDIDATE SPIKE THRESHOLDS")
print("=" * 70)

print(f"\nRows loaded: {len(df):,}")
print(f"Columns: {list(df.columns)}")


# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "date",
    "fx_log_return"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# 4. CLEAN REQUIRED VARIABLES
# ============================================================

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df["fx_log_return"] = pd.to_numeric(
    df["fx_log_return"],
    errors="coerce"
)

df = df.dropna(
    subset=["date", "fx_log_return"]
).copy()

df = df.sort_values("date").reset_index(drop=True)


# ============================================================
# 5. CREATE SQUARED RETURN
# ============================================================

df["fx_return_squared"] = (
    df["fx_log_return"] ** 2
)


# ============================================================
# 6. SELECT TRAINING DATA ONLY
# ============================================================

train = df[
    df["date"] <= pd.Timestamp(TRAIN_END)
].copy()

print("\nTraining period:")
print(f"Start: {train['date'].min().date()}")
print(f"End:   {train['date'].max().date()}")
print(f"Observations: {len(train):,}")


# ============================================================
# 7. BASIC INFORMATION
# ============================================================

zero_count = (
    train["fx_return_squared"] == 0
).sum()

zero_percentage = (
    zero_count / len(train)
) * 100

print("\nSquared-return information:")
print(
    f"Mean:   {train['fx_return_squared'].mean():.8f}"
)
print(
    f"Median: {train['fx_return_squared'].median():.8f}"
)
print(
    f"Maximum: {train['fx_return_squared'].max():.8f}"
)

print(
    f"Zero squared returns: "
    f"{zero_count:,} "
    f"({zero_percentage:.2f}%)"
)


# ============================================================
# 8. CALCULATE CANDIDATE THRESHOLDS
# ============================================================

results = []

for percentile in PERCENTILES:

    # Threshold from TRAINING DATA ONLY
    threshold = train["fx_return_squared"].quantile(
        percentile
    )

    # Equivalent absolute log-return threshold
    abs_return_threshold = np.sqrt(threshold)

    # Approximate percentage change
    percentage_threshold = (
        np.exp(abs_return_threshold) - 1
    ) * 100

    # Classify training observations as spikes
    spike_mask = (
        train["fx_return_squared"] > threshold
    )

    spike_count = spike_mask.sum()

    spike_percentage = (
        spike_count / len(train)
    ) * 100

    results.append({
        "percentile": percentile * 100,
        "threshold_r_squared": threshold,
        "absolute_log_return_threshold": abs_return_threshold,
        "approx_return_percent_threshold": percentage_threshold,
        "training_spike_count": spike_count,
        "training_spike_percentage": spike_percentage
    })


# ============================================================
# 9. CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CANDIDATE SPIKE THRESHOLDS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        formatters={
            "percentile": lambda x: f"{x:.0f}%",
            "threshold_r_squared": lambda x: f"{x:.8f}",
            "absolute_log_return_threshold": lambda x: f"{x:.6f}",
            "approx_return_percent_threshold": lambda x: f"{x:.2f}%",
            "training_spike_count": lambda x: f"{int(x):,}",
            "training_spike_percentage": lambda x: f"{x:.2f}%"
        }
    )
)


# ============================================================
# 11. SAVE RESULTS
# ============================================================

os.makedirs("results", exist_ok=True)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("RESULT SAVED")
print("=" * 70)
print(f"File: {OUTPUT_FILE}")
print("=" * 70)