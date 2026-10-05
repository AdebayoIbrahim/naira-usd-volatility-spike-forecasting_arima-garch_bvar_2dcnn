#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 12:29:54 2026

@author: setup
"""

# -*- coding: utf-8 -*-
"""
Step 07: Build the new forecasting dataset for volatility-spike prediction.

Methodology:
    1. Daily FX volatility proxy = squared daily log return.
    2. Spike threshold = 90th percentile of squared returns among
       NON-ZERO training returns only.
    3. Threshold is estimated using training data only and frozen.
    4. Target = next observation's squared FX return.
    5. Target spike = 1 if next squared return exceeds the frozen threshold.
    6. Rolling 20-observation volatility is retained ONLY as a predictor.
    7. Rolling volatility is calculated separately within continuous
       data segments so long gaps are not crossed.

Train/test split:
    Training targets: up to 2020-12-31
    Test targets:     from 2021-01-01 onward
"""

import os
import numpy as np
import pandas as pd


# ============================================================
# 1. FILE PATHS
# ============================================================

INPUT_FILE = (
    "data_cleaned/daily_transformed_dataset_2006_2025.csv"
)

OUTPUT_FILE = (
    "data_cleaned/forecasting_dataset_2006_2025_spike.csv"
)

THRESHOLD_FILE = (
    "results/training_spike_threshold_new.csv"
)

SUMMARY_FILE = (
    "results/forecasting_dataset_spike_summary.csv"
)


# ============================================================
# 2. SETTINGS
# ============================================================

TRAIN_END = pd.Timestamp("2020-12-31")

ROLLING_WINDOW = 20

SPIKE_PERCENTILE = 0.90


# ============================================================
# 3. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 80)
print("STEP 07: NEW VOLATILITY-SPIKE FORECASTING DATASET")
print("=" * 80)

print(f"\nRows loaded: {len(df):,}")


# ============================================================
# 4. VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "date",
    "buying_rate",
    "fx_log_return",
    "brent_log_return",
    "reserves_log_change",
    "gap_days",
    "long_gap"
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
# 5. CLEAN DATA
# ============================================================

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

numeric_columns = [
    "buying_rate",
    "fx_log_return",
    "brent_log_return",
    "reserves_log_change",
    "gap_days",
    "long_gap"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df = df.dropna(
    subset=[
        "date",
        "fx_log_return"
    ]
).copy()

df = df.sort_values(
    "date"
).reset_index(drop=True)


# ============================================================
# 6. IDENTIFY CONTINUOUS DATA SEGMENTS
# ============================================================

# A new segment begins after a long gap.
df["segment_id"] = (
    df["long_gap"]
    .fillna(0)
    .astype(int)
    .cumsum()
)


# ============================================================
# 7. CREATE SQUARED RETURN
# ============================================================

df["fx_return_squared"] = (
    df["fx_log_return"] ** 2
)


# ============================================================
# 8. GAP-SAFE 20-OBSERVATION ROLLING VOLATILITY
# ============================================================

df["fx_vol_20obs"] = (
    df.groupby("segment_id")["fx_log_return"]
    .transform(
        lambda x: x.rolling(
            window=ROLLING_WINDOW,
            min_periods=ROLLING_WINDOW
        ).std()
    )
)


# ============================================================
# 9. ESTIMATE SPIKE THRESHOLD FROM TRAINING NON-ZERO RETURNS
# ============================================================

training_mask = (
    df["date"] <= TRAIN_END
)

training = df[
    training_mask
].copy()

nonzero_training = training[
    training["fx_log_return"] != 0
].copy()

if nonzero_training.empty:
    raise ValueError(
        "No non-zero training returns available."
    )


threshold = nonzero_training[
    "fx_return_squared"
].quantile(
    SPIKE_PERCENTILE
)


# Equivalent absolute stored log-return threshold
stored_return_threshold = np.sqrt(
    threshold
)

# Raw log-return scale
raw_log_return_threshold = (
    stored_return_threshold / 100
)

# Approximate simple percentage movement
approx_percent_movement = (
    np.exp(raw_log_return_threshold) - 1
) * 100


# ============================================================
# 10. FREEZE THE THRESHOLD
# ============================================================

df["spike_threshold"] = threshold


# ============================================================
# 11. CREATE NEXT-OBSERVATION TARGET
# ============================================================

# The target belongs to the NEXT observation.
df["target_volatility"] = (
    df["fx_return_squared"].shift(-1)
)

df["target_date"] = (
    df["date"].shift(-1)
)


# ============================================================
# 12. CREATE BINARY SPIKE TARGET
# ============================================================

df["target_spike"] = np.where(
    df["target_volatility"].isna(),
    np.nan,
    (
        df["target_volatility"] > threshold
    ).astype(int)
)


# ============================================================
# 13. REMOVE THE FINAL ROW
# ============================================================

# There is no future observation available for the final row.
df = df[
    df["target_volatility"].notna()
].copy()


# ============================================================
# 14. LABEL TRAIN / TEST
# ============================================================

df["dataset"] = np.where(
    df["target_date"] <= TRAIN_END,
    "train",
    "test"
)


# ============================================================
# 15. SELECT FINAL MODEL COLUMNS
# ============================================================

final_columns = [
    "date",
    "target_date",
    "dataset",
    "fx_log_return",
    "brent_log_return",
    "reserves_log_change",
    "fx_vol_20obs",
    "fx_return_squared",
    "target_volatility",
    "target_spike",
    "gap_days",
    "long_gap",
    "segment_id"
]

forecast_df = df[
    final_columns
].copy()


# ============================================================
# 16. BASIC COUNTS
# ============================================================

train_df = forecast_df[
    forecast_df["dataset"] == "train"
].copy()

test_df = forecast_df[
    forecast_df["dataset"] == "test"
].copy()


train_spikes = (
    train_df["target_spike"] == 1
).sum()

test_spikes = (
    test_df["target_spike"] == 1
).sum()


# ============================================================
# 17. PRINT SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("SPIKE THRESHOLD")
print("=" * 80)

print(
    f"Percentile: {SPIKE_PERCENTILE * 100:.0f}%"
)

print(
    f"Threshold r²: {threshold:.8f}"
)

print(
    f"Absolute stored log-return threshold: "
    f"{stored_return_threshold:.6f}"
)

print(
    f"Equivalent raw log-return threshold: "
    f"{raw_log_return_threshold:.8f}"
)

print(
    f"Approximate absolute price movement: "
    f"{approx_percent_movement:.4f}%"
)


print("\n" + "=" * 80)
print("DATASET SUMMARY")
print("=" * 80)

print(
    f"Total forecasting rows: "
    f"{len(forecast_df):,}"
)

print(
    f"Training rows: "
    f"{len(train_df):,}"
)

print(
    f"Test rows: "
    f"{len(test_df):,}"
)

print(
    f"Training spikes: "
    f"{train_spikes:,}"
)

print(
    f"Training spike rate: "
    f"{(train_spikes / len(train_df)) * 100:.2f}%"
)

print(
    f"Test spikes: "
    f"{test_spikes:,}"
)

print(
    f"Test spike rate: "
    f"{(test_spikes / len(test_df)) * 100:.2f}%"
)


# ============================================================
# 18. CHECK TARGET DATES
# ============================================================

print("\n" + "=" * 80)
print("FORECAST TARGET EXAMPLE")
print("=" * 80)

print(
    forecast_df[
        [
            "date",
            "target_date",
            "fx_log_return",
            "target_volatility",
            "target_spike"
        ]
    ].head(10).to_string(index=False)
)


# ============================================================
# 19. MISSING VALUES
# ============================================================

print("\n" + "=" * 80)
print("MISSING VALUES")
print("=" * 80)

print(
    forecast_df[
        [
            "fx_log_return",
            "brent_log_return",
            "reserves_log_change",
            "fx_vol_20obs",
            "target_volatility",
            "target_spike"
        ]
    ].isna().sum()
)


# ============================================================
# 20. SAVE THRESHOLD
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

threshold_df = pd.DataFrame({
    "training_end": [TRAIN_END.date()],
    "method": [
        "90th percentile of squared FX returns among non-zero training returns"
    ],
    "percentile": [90],
    "threshold_r_squared": [threshold],
    "absolute_stored_log_return_threshold": [
        stored_return_threshold
    ],
    "equivalent_raw_log_return_threshold": [
        raw_log_return_threshold
    ],
    "approx_price_movement_percent": [
        approx_percent_movement
    ],
    "training_nonzero_returns": [
        len(nonzero_training)
    ],
    "training_spike_count": [
        train_spikes
    ]
})

threshold_df.to_csv(
    THRESHOLD_FILE,
    index=False
)


# ============================================================
# 21. SAVE DATASET
# ============================================================

os.makedirs(
    "data_cleaned",
    exist_ok=True
)

forecast_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 22. SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame({
    "metric": [
        "total_rows",
        "training_rows",
        "test_rows",
        "training_spikes",
        "training_spike_rate",
        "test_spikes",
        "test_spike_rate",
        "spike_threshold_r_squared"
    ],
    "value": [
        len(forecast_df),
        len(train_df),
        len(test_df),
        train_spikes,
        (train_spikes / len(train_df)) * 100,
        test_spikes,
        (test_spikes / len(test_df)) * 100,
        threshold
    ]
})

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# 23. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    f"Forecasting dataset: {OUTPUT_FILE}"
)

print(
    f"Spike threshold:     {THRESHOLD_FILE}"
)

print(
    f"Summary:              {SUMMARY_FILE}"
)

print("=" * 80)