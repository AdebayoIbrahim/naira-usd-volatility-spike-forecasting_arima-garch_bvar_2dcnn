#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 16:41:36 2026

@author: setup
"""

import pandas as pd
import os


# ============================================================
# 1. FILE PATHS
# ============================================================

input_file = (
    "data_cleaned/forecasting_dataset_2006_2025.csv"
)

threshold_file = (
    "results/training_spike_threshold.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(
    input_file,
    parse_dates=["date", "target_date"]
)

threshold_df = pd.read_csv(threshold_file)


# ============================================================
# 3. READ FINAL SPIKE THRESHOLD
# ============================================================

spike_threshold = threshold_df.loc[
    0, "spike_threshold"
]

print("\nFinal training spike threshold:")
print(spike_threshold)


# ============================================================
# 4. DEFINE MODEL VARIABLES
# ============================================================

predictor_columns = [
    "fx_log_return",
    "brent_log_return",
    "reserves_log_change",
    "fx_vol_20obs"
]

target_columns = [
    "target_volatility",
    "target_spike"
]

identification_columns = [
    "date",
    "target_date",
    "dataset"
]


# ============================================================
# 5. CREATE MASTER MODEL INPUT DATASET
# ============================================================

master_columns = (
    identification_columns
    + predictor_columns
    + target_columns
)

model_input = df[master_columns].copy()


# ============================================================
# 6. SORT CHRONOLOGICALLY
# ============================================================

model_input = (
    model_input
    .sort_values("date")
    .reset_index(drop=True)
)


# ============================================================
# 7. CHECK FOR DATA LEAKAGE VARIABLES
# ============================================================

print("\nVariables being used as predictors:")
for col in predictor_columns:
    print(" -", col)

print("\nVariables being used as targets:")
for col in target_columns:
    print(" -", col)


# ============================================================
# 8. MISSING-VALUE CHECK
# ============================================================

print("\nMissing values in model inputs:")
print(
    model_input[
        predictor_columns + target_columns
    ].isna().sum()
)


# ============================================================
# 9. CREATE MULTIVARIATE MODEL DATASET
# ============================================================
# BVAR and CNN need all four predictor variables.
# Therefore, rows with missing predictor values are removed.
#
# We do NOT fill or interpolate these values.

multivariate_input = model_input.dropna(
    subset=predictor_columns
).copy()


# ============================================================
# 10. TRAIN / TEST COUNTS
# ============================================================

print("\nMASTER DATASET COUNTS:")
print(
    model_input["dataset"].value_counts()
)


print("\nMULTIVARIATE DATASET COUNTS:")
print(
    multivariate_input["dataset"].value_counts()
)


# ============================================================
# 11. TRAIN / TEST DATE RANGES
# ============================================================

print("\nMASTER DATE RANGES:")

for group_name, group in model_input.groupby("dataset"):

    print(
        f"\n{group_name.upper()}:"
    )

    print(
        "Forecast origin:",
        group["date"].min().date(),
        "→",
        group["date"].max().date()
    )

    print(
        "Forecast target:",
        group["target_date"].min().date(),
        "→",
        group["target_date"].max().date()
    )


# ============================================================
# 12. CHECK TARGET DISTRIBUTION
# ============================================================

print("\nTARGET SPIKE DISTRIBUTION:")

spike_distribution = (
    model_input
    .groupby("dataset")["target_spike"]
    .value_counts()
    .unstack(fill_value=0)
)

print(spike_distribution)


# ============================================================
# 13. CREATE SUMMARY TABLE
# ============================================================

summary_rows = []

for dataset_name in ["train", "test"]:

    master_group = model_input[
        model_input["dataset"] == dataset_name
    ]

    multi_group = multivariate_input[
        multivariate_input["dataset"] == dataset_name
    ]

    summary_rows.append({
        "dataset": dataset_name,
        "master_rows": len(master_group),
        "multivariate_rows": len(multi_group),
        "rows_removed_for_missing_predictors":
            len(master_group) - len(multi_group),
        "spikes": int(
            master_group["target_spike"].sum()
        ),
        "spike_rate":
            master_group["target_spike"].mean()
    })

summary = pd.DataFrame(summary_rows)


print("\nMODEL INPUT SUMMARY:")
print(summary)


# ============================================================
# 14. SAVE OUTPUTS
# ============================================================

os.makedirs("data_cleaned", exist_ok=True)
os.makedirs("results", exist_ok=True)


master_output = (
    "data_cleaned/"
    "model_input_master_2006_2025.csv"
)

multivariate_output = (
    "data_cleaned/"
    "multivariate_model_input_2006_2025.csv"
)

summary_output = (
    "results/"
    "model_input_summary.csv"
)


model_input.to_csv(
    master_output,
    index=False
)

multivariate_input.to_csv(
    multivariate_output,
    index=False
)

summary.to_csv(
    summary_output,
    index=False
)


# ============================================================
# 15. FINAL MESSAGE
# ============================================================

print("\nSaved master model input:")
print(master_output)

print("\nSaved multivariate model input:")
print(multivariate_output)

print("\nSaved model input summary:")
print(summary_output)