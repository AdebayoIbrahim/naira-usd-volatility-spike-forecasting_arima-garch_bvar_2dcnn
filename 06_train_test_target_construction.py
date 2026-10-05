#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 17:18:03 2026

@author: setup
"""

import pandas as pd
import numpy as np
import os


# ============================================================
# 1. LOAD VOLATILITY DATASET
# ============================================================

input_file = (
    "data_cleaned/daily_volatility_dataset_2006_2025.csv"
)

df = pd.read_csv(
    input_file,
    parse_dates=["date"]
)

df = df.sort_values("date").reset_index(drop=True)


# ============================================================
# 2. DEFINE TRAIN / TEST PERIOD
# ============================================================

train_end = pd.Timestamp("2020-12-31")
test_start = pd.Timestamp("2021-01-01")


# ============================================================
# 3. CHECK DATE RANGE
# ============================================================

print("\nFull dataset:")
print("Start:", df["date"].min())
print("End:", df["date"].max())


# ============================================================
# 4. CALCULATE TRAINING-ONLY SPIKE THRESHOLD
# ============================================================

training_volatility = df.loc[
    df["date"] <= train_end,
    "fx_vol_20obs"
].dropna()


spike_threshold = training_volatility.quantile(0.90)


print("\nTraining-only spike threshold")
print("--------------------------------")
print("Training end:", train_end.date())
print("90th percentile:", spike_threshold)
print("Number of training volatility observations:",
      len(training_volatility))


# ============================================================
# 5. CREATE NEXT-PERIOD TARGET
# ============================================================

df["target_date"] = df["date"].shift(-1)

df["target_volatility"] = (
    df["fx_vol_20obs"].shift(-1)
)


# ============================================================
# 6. CREATE SPIKE LABEL
# ============================================================

df["target_spike"] = np.where(
    df["target_volatility"].isna(),
    np.nan,
    (
        df["target_volatility"] > spike_threshold
    ).astype(int)
)


# ============================================================
# 7. CREATE TRAIN / TEST LABEL
# ============================================================

# The split is based on the DATE BEING FORECAST,
# not merely the date containing the predictors.

df["dataset"] = np.where(
    df["target_date"] <= train_end,
    "train",
    np.where(
        df["target_date"] >= test_start,
        "test",
        np.nan
    )
)


# ============================================================
# 8. KEEP RELEVANT FORECASTING OBSERVATIONS
# ============================================================

forecast_df = df[
    df["dataset"].isin(["train", "test"])
].copy()


# ============================================================
# 9. REMOVE ROWS WITHOUT A VALID TARGET
# ============================================================

forecast_df = forecast_df.dropna(
    subset=["target_volatility", "target_spike"]
)


forecast_df["target_spike"] = (
    forecast_df["target_spike"]
    .astype(int)
)


# ============================================================
# 10. SUMMARY
# ============================================================

print("\nForecasting dataset:")
print("--------------------")

print(
    forecast_df[
        [
            "date",
            "target_date",
            "fx_vol_20obs",
            "target_volatility",
            "target_spike",
            "dataset"
        ]
    ].head(10)
)


print("\nDataset counts:")
print(
    forecast_df["dataset"].value_counts()
)


# ============================================================
# 11. SPIKE COUNTS
# ============================================================

print("\nSpike distribution:")
print(
    forecast_df
    .groupby("dataset")["target_spike"]
    .value_counts()
    .unstack(fill_value=0)
)


# ============================================================
# 12. SPIKE PROPORTIONS
# ============================================================

spike_summary = (
    forecast_df
    .groupby("dataset")["target_spike"]
    .agg(
        total="count",
        spikes="sum",
        spike_rate="mean"
    )
)

print("\nSpike summary:")
print(spike_summary)


# ============================================================
# 13. SAVE
# ============================================================

os.makedirs("data_cleaned", exist_ok=True)

output_file = (
    "data_cleaned/"
    "forecasting_dataset_2006_2025.csv"
)

forecast_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# 14. SAVE THRESHOLD
# ============================================================

threshold_file = (
    "results/training_spike_threshold.csv"
)

os.makedirs("results", exist_ok=True)

pd.DataFrame({
    "threshold_method": ["90th_percentile"],
    "volatility_measure": ["fx_vol_20obs"],
    "training_end": [train_end.date()],
    "spike_threshold": [spike_threshold]
}).to_csv(
    threshold_file,
    index=False
)
    
showstable = pd.read_csv(output_file)    


print("\nSaved forecasting dataset:")
print(output_file)

print("\nSaved threshold:")
print(threshold_file)