#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 18:50:01 2026

@author: setup
"""

# ============================================================
# STEP 09J: BVAR(8) FORECAST EVALUATION
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "results/"
    "bvar8_spike_walk_forward_forecasts.csv"
)

METRICS_OUTPUT = (
    "results/"
    "bvar8_spike_test_metrics.csv"
)

CONFUSION_OUTPUT = (
    "results/"
    "bvar8_spike_confusion_matrix.csv"
)

EVALUATED_OUTPUT = (
    "results/"
    "bvar8_spike_evaluated_forecasts.csv"
)


# ============================================================
# OUTPUT FOLDER
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


# ============================================================
# LOAD FORECASTS
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)

df["origin_date"] = pd.to_datetime(
    df["origin_date"]
)

df["target_date"] = pd.to_datetime(
    df["target_date"]
)

print("=" * 90)
print("STEP 09J: BVAR(8) FORECAST EVALUATION")
print("=" * 90)

print(
    f"\nForecast observations loaded: "
    f"{len(df):,}"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "origin_date",
    "target_date",
    "actual_volatility",
    "forecast_volatility",
    "actual_spike",
    "predicted_spike_probability",
    "predicted_spike"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: "
        f"{missing_columns}"
    )


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

df = df.dropna(
    subset=[
        "actual_volatility",
        "forecast_volatility",
        "actual_spike",
        "predicted_spike_probability",
        "predicted_spike"
    ]
).copy()

print(
    f"Valid evaluation observations: "
    f"{len(df):,}"
)


# ============================================================
# EXTRACT VARIABLES
# ============================================================

actual_volatility = (
    df["actual_volatility"]
    .to_numpy(dtype=float)
)

forecast_volatility = (
    df["forecast_volatility"]
    .to_numpy(dtype=float)
)

actual_spike = (
    df["actual_spike"]
    .to_numpy(dtype=int)
)

predicted_spike = (
    df["predicted_spike"]
    .to_numpy(dtype=int)
)

spike_probability = (
    df["predicted_spike_probability"]
    .to_numpy(dtype=float)
)


# ============================================================
# CONTINUOUS FORECAST METRICS
# ============================================================

mae = mean_absolute_error(
    actual_volatility,
    forecast_volatility
)

mse = mean_squared_error(
    actual_volatility,
    forecast_volatility
)

rmse = np.sqrt(
    mse
)


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(
    actual_spike,
    predicted_spike
)

precision = precision_score(
    actual_spike,
    predicted_spike,
    zero_division=0
)

recall = recall_score(
    actual_spike,
    predicted_spike,
    zero_division=0
)

f1 = f1_score(
    actual_spike,
    predicted_spike,
    zero_division=0
)


# ============================================================
# ROC-AUC
# ============================================================

if len(np.unique(actual_spike)) == 2:

    auc = roc_auc_score(
        actual_spike,
        spike_probability
    )

else:

    auc = np.nan


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    actual_spike,
    predicted_spike,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()


# ============================================================
# SPECIFICITY
# ============================================================

if (tn + fp) > 0:

    specificity = (
        tn / (tn + fp)
    )

else:

    specificity = np.nan


# ============================================================
# SPIKE COUNTS
# ============================================================

actual_normal = int(
    (actual_spike == 0).sum()
)

actual_spikes = int(
    (actual_spike == 1).sum()
)

predicted_normal = int(
    (predicted_spike == 0).sum()
)

predicted_spikes = int(
    (predicted_spike == 1).sum()
)


# ============================================================
# SPIKE RATES
# ============================================================

actual_spike_rate = (
    actual_spikes / len(df)
)

predicted_spike_rate = (
    predicted_spikes / len(df)
)


# ============================================================
# METRICS TABLE
# ============================================================

metrics = pd.DataFrame([
    {
        "model": "BVAR(8)",

        "observations": len(df),

        "mae": mae,

        "mse": mse,

        "rmse": rmse,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "roc_auc": auc,

        "specificity": specificity,

        "actual_normal":
            actual_normal,

        "actual_spikes":
            actual_spikes,

        "predicted_normal":
            predicted_normal,

        "predicted_spikes":
            predicted_spikes,

        "actual_spike_rate":
            actual_spike_rate,

        "predicted_spike_rate":
            predicted_spike_rate,

        "mean_predicted_spike_probability":
            spike_probability.mean()
    }
])


# ============================================================
# CONFUSION MATRIX TABLE
# ============================================================

confusion_df = pd.DataFrame({

    "actual_class": [
        "Normal",
        "Normal",
        "Spike",
        "Spike"
    ],

    "predicted_class": [
        "Normal",
        "Spike",
        "Normal",
        "Spike"
    ],

    "count": [
        tn,
        fp,
        fn,
        tp
    ]
})


# ============================================================
# SAVE EVALUATED FORECASTS
# ============================================================

df.to_csv(
    EVALUATED_OUTPUT,
    index=False
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics.to_csv(
    METRICS_OUTPUT,
    index=False
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

confusion_df.to_csv(
    CONFUSION_OUTPUT,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 90)
print("BVAR(8) CONTINUOUS VOLATILITY METRICS")
print("=" * 90)

print(
    f"\nMAE  : {mae:.6f}"
)

print(
    f"MSE  : {mse:.6f}"
)

print(
    f"RMSE : {rmse:.6f}"
)


print("\n" + "=" * 90)
print("BVAR(8) SPIKE CLASSIFICATION METRICS")
print("=" * 90)

print(
    f"\nAccuracy    : {accuracy:.6f}"
)

print(
    f"Precision   : {precision:.6f}"
)

print(
    f"Recall      : {recall:.6f}"
)

print(
    f"F1 Score    : {f1:.6f}"
)

print(
    f"ROC-AUC     : {auc:.6f}"
)

print(
    f"Specificity : {specificity:.6f}"
)


print("\n" + "=" * 90)
print("SPIKE COUNTS")
print("=" * 90)

print(
    f"\nActual normal observations: "
    f"{actual_normal:,}"
)

print(
    f"Actual spikes: "
    f"{actual_spikes:,}"
)

print(
    f"Predicted normal observations: "
    f"{predicted_normal:,}"
)

print(
    f"Predicted spikes: "
    f"{predicted_spikes:,}"
)


print("\n" + "=" * 90)
print("SPIKE RATES")
print("=" * 90)

print(
    f"\nActual spike rate: "
    f"{actual_spike_rate:.4%}"
)

print(
    f"Predicted spike rate: "
    f"{predicted_spike_rate:.4%}"
)

print(
    f"Mean predicted probability: "
    f"{spike_probability.mean():.6f}"
)


print("\n" + "=" * 90)
print("CONFUSION MATRIX")
print("=" * 90)

print(
    f"\nTrue Negatives  (TN): {tn:,}"
)

print(
    f"False Positives (FP): {fp:,}"
)

print(
    f"False Negatives (FN): {fn:,}"
)

print(
    f"True Positives   (TP): {tp:,}"
)


print("\n" + "=" * 90)
print("RESULTS SAVED")
print("=" * 90)

print(
    f"\nMetrics:"
    f"\n  {METRICS_OUTPUT}"
)

print(
    f"\nConfusion matrix:"
    f"\n  {CONFUSION_OUTPUT}"
)

print(
    f"\nEvaluated forecasts:"
    f"\n  {EVALUATED_OUTPUT}"
)

print("=" * 90)