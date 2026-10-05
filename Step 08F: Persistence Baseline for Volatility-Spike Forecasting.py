#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 17:35:08 2026

@author: setup
"""

# -*- coding: utf-8 -*-

"""
Step 08F: Persistence Baseline for Volatility-Spike Forecasting

Purpose:
    Establish a simple benchmark before comparing ARIMA-GARCH,
    BVAR, and CNN.

Baseline idea:
    Today's squared FX return is used as tomorrow's
    predicted squared FX return.

    predicted_volatility(t+1) = r_t^2

Spike threshold:
    Frozen training threshold from Step 07.

Metrics:
    MAE
    MSE
    RMSE
    Accuracy
    Precision
    Recall
    F1
    ROC-AUC
    Specificity
"""

import os

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# 1. FILE PATHS
# ============================================================

INPUT_FILE = (
    "data_cleaned/forecasting_dataset_2006_2025_spike.csv"
)

THRESHOLD_FILE = (
    "results/training_spike_threshold_new.csv"
)

OUTPUT_FILE = (
    "results/persistence_baseline_spike_test_metrics.csv"
)

FORECAST_OUTPUT = (
    "results/persistence_baseline_spike_forecasts.csv"
)

CONFUSION_OUTPUT = (
    "results/persistence_baseline_spike_confusion_matrix.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df["target_date"] = pd.to_datetime(
    df["target_date"],
    errors="coerce"
)

numeric_columns = [
    "fx_log_return",
    "target_volatility",
    "target_spike",
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
        "target_date",
        "fx_log_return",
        "target_volatility",
        "target_spike"
    ]
).copy()


df = df.sort_values(
    "date"
).reset_index(
    drop=True
)


# ============================================================
# 3. LOAD FROZEN THRESHOLD
# ============================================================

if not os.path.exists(THRESHOLD_FILE):

    raise FileNotFoundError(
        f"Threshold file not found:\n{THRESHOLD_FILE}"
    )


threshold_df = pd.read_csv(
    THRESHOLD_FILE
)

SPIKE_THRESHOLD = float(
    threshold_df.loc[
        0,
        "threshold_r_squared"
    ]
)


# ============================================================
# 4. TEST DATA
# ============================================================

test = df[
    df["target_date"] > pd.Timestamp("2020-12-31")
].copy()


# Skip targets immediately after long gaps,
# matching the ARIMA-GARCH evaluation.
test = test[
    test["long_gap"].fillna(0).astype(int) != 1
].copy()


test = test.reset_index(
    drop=True
)


# ============================================================
# 5. CREATE PERSISTENCE FORECAST
# ============================================================

# Today's squared return becomes tomorrow's forecast.

test["forecast_squared_return"] = (
    test["fx_log_return"] ** 2
)


test["actual_squared_return"] = (
    test["target_volatility"]
)


# ============================================================
# 6. CREATE SPIKE PREDICTIONS
# ============================================================

test["predicted_spike"] = (
    test["forecast_squared_return"]
    > SPIKE_THRESHOLD
).astype(int)


test["actual_spike"] = (
    test["target_spike"]
    .astype(int)
)


# ============================================================
# 7. CONTINUOUS METRICS
# ============================================================

errors = (
    test["forecast_squared_return"]
    - test["actual_squared_return"]
)

mae = np.abs(
    errors
).mean()

mse = (
    errors ** 2
).mean()

rmse = np.sqrt(
    mse
)


# ============================================================
# 8. CLASSIFICATION METRICS
# ============================================================

y_true = test[
    "actual_spike"
].to_numpy()

y_pred = test[
    "predicted_spike"
].to_numpy()

y_score = test[
    "forecast_squared_return"
].to_numpy()


accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

auc = roc_auc_score(
    y_true,
    y_score
)


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()


specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else np.nan
)


# ============================================================
# 10. PRINT RESULTS
# ============================================================

print("=" * 90)
print("STEP 08F: PERSISTENCE BASELINE")
print("=" * 90)

print(
    f"\nFrozen spike threshold: "
    f"{SPIKE_THRESHOLD:.8f}"
)

print(
    f"Test observations: "
    f"{len(test):,}"
)

print("\n" + "=" * 90)
print("CONTINUOUS METRICS")
print("=" * 90)

print(
    f"MAE:  {mae:.10f}"
)

print(
    f"MSE:  {mse:.10f}"
)

print(
    f"RMSE: {rmse:.10f}"
)


print("\n" + "=" * 90)
print("SPIKE CLASSIFICATION METRICS")
print("=" * 90)

print(
    f"Accuracy:   {accuracy:.6f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Precision:  {precision:.6f} "
    f"({precision * 100:.2f}%)"
)

print(
    f"Recall:     {recall:.6f} "
    f"({recall * 100:.2f}%)"
)

print(
    f"F1:         {f1:.6f} "
    f"({f1 * 100:.2f}%)"
)

print(
    f"ROC-AUC:    {auc:.6f}"
)

print(
    f"Specificity:{specificity:.6f} "
    f"({specificity * 100:.2f}%)"
)


print("\n" + "=" * 90)
print("CONFUSION MATRIX")
print("=" * 90)

print(
    "                 Predicted"
)

print(
    "                 Normal   Spike"
)

print(
    f"Actual Normal    {tn:6d}  {fp:6d}"
)

print(
    f"Actual Spike     {fn:6d}  {tp:6d}"
)


# ============================================================
# 11. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame({
    "model": [
        "Persistence baseline"
    ],

    "test_observations": [
        len(test)
    ],

    "spike_threshold": [
        SPIKE_THRESHOLD
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

    "ROC_AUC": [
        auc
    ],

    "Specificity": [
        specificity
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
    ]
})


# ============================================================
# 12. SAVE CONFUSION MATRIX
# ============================================================

confusion_df = pd.DataFrame({
    "actual": [
        "normal",
        "normal",
        "spike",
        "spike"
    ],

    "predicted": [
        "normal",
        "spike",
        "normal",
        "spike"
    ],

    "count": [
        tn,
        fp,
        fn,
        tp
    ]
})


# ============================================================
# 13. SAVE FORECASTS
# ============================================================

forecast_columns = [
    "date",
    "target_date",
    "fx_log_return",
    "forecast_squared_return",
    "actual_squared_return",
    "predicted_spike",
    "actual_spike"
]

test[
    forecast_columns
].to_csv(
    FORECAST_OUTPUT,
    index=False
)


# ============================================================
# 14. SAVE OUTPUTS
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

metrics_df.to_csv(
    OUTPUT_FILE,
    index=False
)

confusion_df.to_csv(
    CONFUSION_OUTPUT,
    index=False
)


print("\n" + "=" * 90)
print("FILES SAVED")
print("=" * 90)

print(
    f"Metrics:          {OUTPUT_FILE}"
)

print(
    f"Forecasts:        {FORECAST_OUTPUT}"
)

print(
    f"Confusion matrix: {CONFUSION_OUTPUT}"
)

print("=" * 90)