#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 11:43:13 2026

@author: setup
"""

# ============================================================
# STEP 08E: PERSISTENCE BASELINE - COMMON-SAMPLE EVALUATION
# ============================================================
#
# Naive persistence benchmark:
#
#   v_hat(t+1) = v_t = r_t^2
#
# The benchmark uses the exact same 1,083 forecast origins
# used by the BVAR(8) and ARIMA-GARCH common-sample evaluation.
# ============================================================

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ============================================================
# 1. SETTINGS
# ============================================================

INPUT_FILE = (
    "data_cleaned/"
    "forecasting_dataset_2006_2025_spike_gap_target_safe.csv"
)

BVAR_GRID_FILE = (
    "results/"
    "bvar8_spike_walk_forward_forecasts.csv"
)

OUTPUT_FILE = (
    "results/"
    "persistence_common_sample_forecasts.csv"
)

METRICS_FILE = (
    "results/"
    "persistence_common_sample_metrics.csv"
)

TRAIN_END = "2020-12-31"

# Locked project threshold for the NEW squared-return target.
SPIKE_THRESHOLD = 0.10927724


# ============================================================
# 2. CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


# ============================================================
# 3. LOAD GAP-SAFE TARGET DATA
# ============================================================

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["date", "target_date"]
)

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)


# ============================================================
# 4. LOAD BVAR COMMON-SAMPLE GRID
# ============================================================
#
# We use the BVAR forecast file only to obtain the exact
# origin_date -> target_date pairs retained for the common
# evaluation sample.
# ============================================================

bvar = pd.read_csv(
    BVAR_GRID_FILE,
    parse_dates=["origin_date", "target_date"]
)

common_grid = (
    bvar[
        [
            "origin_date",
            "target_date"
        ]
    ]
    .drop_duplicates()
    .sort_values("origin_date")
    .reset_index(drop=True)
)


print("=" * 90)
print("STEP 08E: PERSISTENCE COMMON-SAMPLE EVALUATION")
print("=" * 90)

print(
    f"\nAnalysis-ready rows loaded: {len(df):,}"
)

print(
    f"BVAR common-sample origins loaded: "
    f"{len(common_grid):,}"
)

print(
    f"Frozen spike threshold: "
    f"{SPIKE_THRESHOLD:.8f}"
)


# ============================================================
# 5. CHECK COMMON-SAMPLE GRID
# ============================================================

if len(common_grid) != 1083:

    raise ValueError(
        "Expected exactly 1,083 common-sample origins, "
        f"but found {len(common_grid):,}."
    )

print(
    f"\nCommon evaluation origins confirmed: "
    f"{len(common_grid):,}"
)

print(
    f"First origin: "
    f"{common_grid['origin_date'].min().date()}"
)

print(
    f"Last origin:  "
    f"{common_grid['origin_date'].max().date()}"
)


# ============================================================
# 6. BUILD PERSISTENCE FORECASTS
# ============================================================
#
# At origin t:
#
#   predicted_volatility = r_t^2
#
# The actual target is the next-observation squared return,
# already stored on the origin row as target_volatility.
# ============================================================

origin_data = df[
    [
        "date",
        "target_date",
        "fx_return_squared",
        "target_volatility",
        "target_spike",
        "segment_id"
    ]
].copy()

origin_data = origin_data.rename(
    columns={
        "date": "origin_date"
    }
)

forecast_results = (
    common_grid
    .merge(
        origin_data,
        on=["origin_date", "target_date"],
        how="left",
        validate="one_to_one"
    )
)


# ============================================================
# 7. DATA VALIDATION
# ============================================================

if forecast_results["fx_return_squared"].isna().any():

    missing = int(
        forecast_results["fx_return_squared"].isna().sum()
    )

    raise ValueError(
        f"{missing} common-sample origins could not be "
        "matched to the forecasting dataset."
    )

if forecast_results["target_volatility"].isna().any():

    missing = int(
        forecast_results["target_volatility"].isna().sum()
    )

    raise ValueError(
        f"{missing} common-sample observations have missing "
        "actual target volatility."
    )

if forecast_results["target_spike"].isna().any():

    raise ValueError(
        "Missing actual spike labels in common sample."
    )


# ============================================================
# 8. PERSISTENCE FORECAST
# ============================================================

forecast_results["forecast_volatility"] = (
    forecast_results["fx_return_squared"]
)

forecast_results["actual_volatility"] = (
    forecast_results["target_volatility"].astype(float)
)

forecast_results["actual_spike"] = (
    forecast_results["target_spike"].astype(int)
)

forecast_results["predicted_spike"] = (
    forecast_results["forecast_volatility"]
    > SPIKE_THRESHOLD
).astype(int)

forecast_results["spike_threshold"] = (
    SPIKE_THRESHOLD
)


# ============================================================
# 9. FORECAST ERRORS
# ============================================================

forecast_results["error"] = (
    forecast_results["actual_volatility"]
    - forecast_results["forecast_volatility"]
)

forecast_results["absolute_error"] = (
    forecast_results["error"].abs()
)

forecast_results["squared_error"] = (
    forecast_results["error"] ** 2
)


# ============================================================
# 10. NUMERICAL FORECAST METRICS
# ============================================================

mae = (
    forecast_results["absolute_error"]
    .mean()
)

mse = (
    forecast_results["squared_error"]
    .mean()
)

rmse = np.sqrt(mse)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

actual = (
    forecast_results["actual_spike"]
)

predicted = (
    forecast_results["predicted_spike"]
)

tn = int(
    ((predicted == 0) & (actual == 0)).sum()
)

fp = int(
    ((predicted == 1) & (actual == 0)).sum()
)

fn = int(
    ((predicted == 0) & (actual == 1)).sum()
)

tp = int(
    ((predicted == 1) & (actual == 1)).sum()
)


# ============================================================
# 12. CLASSIFICATION METRICS
# ============================================================

accuracy = (
    (tp + tn)
    / (tp + tn + fp + fn)
)

precision = (
    tp / (tp + fp)
    if (tp + fp) > 0
    else np.nan
)

recall = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else np.nan
)

f1 = (
    2 * precision * recall
    / (precision + recall)
    if (
        pd.notna(precision)
        and pd.notna(recall)
        and (precision + recall) > 0
    )
    else np.nan
)


# ============================================================
# 13. ROC-AUC
# ============================================================
#
# Continuous persistence forecast is the ROC score.
# Higher current squared return = stronger evidence of
# a volatility spike tomorrow.
# ============================================================

auc = roc_auc_score(
    actual,
    forecast_results["forecast_volatility"]
)


# ============================================================
# 14. PRINT RESULTS
# ============================================================

print("\n" + "=" * 90)
print("PERSISTENCE COMMON-SAMPLE RESULTS")
print("=" * 90)

print(
    f"\nForecast observations: "
    f"{len(forecast_results):,}"
)

print(
    f"MAE:  {mae:.10f}"
)

print(
    f"MSE:  {mse:.10f}"
)

print(
    f"RMSE: {rmse:.10f}"
)

print("\nCONFUSION MATRIX")
print("--------------------------------")

print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)

print("\nCLASSIFICATION METRICS")
print("--------------------------------")

print(
    f"Accuracy:  {accuracy:.10f}"
)

print(
    f"Precision: {precision:.10f}"
)

print(
    f"Recall:    {recall:.10f}"
)

print(
    f"F1:        {f1:.10f}"
)

print(
    f"ROC-AUC:   {auc:.10f}"
)


# ============================================================
# 15. FIRST 10 FORECASTS
# ============================================================

print("\nFIRST 10 FORECASTS")
print("--------------------------------")

print(
    forecast_results[
        [
            "origin_date",
            "target_date",
            "fx_return_squared",
            "forecast_volatility",
            "actual_volatility",
            "actual_spike",
            "predicted_spike"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# 16. SAVE FORECASTS
# ============================================================

forecast_results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 17. SAVE METRICS
# ============================================================

metrics = pd.DataFrame({
    "model": [
        "Persistence"
    ],

    "forecast_observations": [
        len(forecast_results)
    ],

    "MAE": [
        mae
    ],

    "MSE": [
        mse
    ],

    "RMSE": [
        rmse
    ],

    "TN": [
        tn
    ],

    "FP": [
        fp
    ],

    "FN": [
        fn
    ],

    "TP": [
        tp
    ],

    "Accuracy": [
        accuracy
    ],

    "Precision": [
        precision
    ],

    "Recall": [
        recall
    ],

    "F1": [
        f1
    ],

    "AUC": [
        auc
    ],

    "spike_threshold": [
        SPIKE_THRESHOLD
    ]
})

metrics.to_csv(
    METRICS_FILE,
    index=False
)


print("\n" + "=" * 90)
print("RESULT FILES SAVED")
print("=" * 90)

print(
    f"\nForecasts: "
    f"{OUTPUT_FILE}"
)

print(
    f"Metrics:   "
    f"{METRICS_FILE}"
)

print(
    "\nCommon-sample persistence evaluation completed."
)
