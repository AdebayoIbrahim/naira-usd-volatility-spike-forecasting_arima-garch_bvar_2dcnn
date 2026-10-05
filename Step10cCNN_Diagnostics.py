#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 16:40:34 2026

@author: setup
"""

# ============================================================
# STEP 10C-DIAGNOSTIC
# Inspect CNN Validation Forecast Quality
# ============================================================

import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. LOAD VALIDATION FORECASTS
# ============================================================

FILE = "results/cnn_validation_predictions_v2.csv"

df = pd.read_csv(FILE)

print("=" * 70)
print("CNN VALIDATION DIAGNOSTIC")
print("=" * 70)

print(f"\nObservations: {len(df)}")


# ============================================================
# 2. BASIC CHECK
# ============================================================

required = [
    "target_date",
    "actual_volatility",
    "predicted_volatility",
    "actual_spike",
    "predicted_spike"
]

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )


# ============================================================
# 3. CONTINUOUS FORECAST METRICS
# ============================================================

actual = df[
    "actual_volatility"
].to_numpy()

predicted = df[
    "predicted_volatility"
].to_numpy()


mae = mean_absolute_error(
    actual,
    predicted
)

mse = mean_squared_error(
    actual,
    predicted
)

rmse = np.sqrt(mse)


print("\n" + "-" * 70)
print("CONTINUOUS VOLATILITY FORECAST")
print("-" * 70)

print(
    f"MAE  : {mae:.8f}"
)

print(
    f"MSE  : {mse:.8f}"
)

print(
    f"RMSE : {rmse:.8f}"
)

print(
    f"\nActual volatility mean: "
    f"{actual.mean():.8f}"
)

print(
    f"Predicted volatility mean: "
    f"{predicted.mean():.8f}"
)

print(
    f"Actual volatility max: "
    f"{actual.max():.8f}"
)

print(
    f"Predicted volatility max: "
    f"{predicted.max():.8f}"
)


# ============================================================
# 4. PREDICTION DISTRIBUTION
# ============================================================

print("\n" + "-" * 70)
print("PREDICTION DISTRIBUTION")
print("-" * 70)

for q in [
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    1.00
]:

    print(
        f"Predicted {q:.0%} percentile: "
        f"{np.quantile(predicted, q):.8f}"
    )


# ============================================================
# 5. SPIKE COUNTS
# ============================================================

SPIKE_THRESHOLD = 0.10927724

actual_spike = (
    actual > SPIKE_THRESHOLD
).astype(int)

predicted_spike = (
    predicted > SPIKE_THRESHOLD
).astype(int)


print("\n" + "-" * 70)
print("SPIKE THRESHOLD CHECK")
print("-" * 70)

print(
    f"Threshold: {SPIKE_THRESHOLD}"
)

print(
    f"Actual spikes: "
    f"{actual_spike.sum()}"
)

print(
    f"Predicted spikes: "
    f"{predicted_spike.sum()}"
)


# ============================================================
# 6. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    actual_spike,
    predicted_spike
)

tn, fp, fn, tp = cm.ravel()


print("\n" + "-" * 70)
print("CONFUSION MATRIX")
print("-" * 70)

print(
    f"TN = {tn}"
)

print(
    f"FP = {fp}"
)

print(
    f"FN = {fn}"
)

print(
    f"TP = {tp}"
)


# ============================================================
# 7. CLASSIFICATION METRICS
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


print("\n" + "-" * 70)
print("SPIKE CLASSIFICATION METRICS")
print("-" * 70)

print(
    f"Accuracy  : {accuracy:.6f}"
)

print(
    f"Precision : {precision:.6f}"
)

print(
    f"Recall    : {recall:.6f}"
)

print(
    f"F1        : {f1:.6f}"
)


# ============================================================
# 8. ROC-AUC USING CONTINUOUS CNN FORECAST
# ============================================================
#
# This is important.
#
# Even if the fixed threshold produces zero predicted spikes,
# ROC-AUC asks whether larger CNN forecasts tend to correspond
# to actual spike observations.
#
# ============================================================

if len(np.unique(actual_spike)) == 2:

    auc = roc_auc_score(
        actual_spike,
        predicted
    )

    ap = average_precision_score(
        actual_spike,
        predicted
    )

    print("\n" + "-" * 70)
    print("RANKING ABILITY")
    print("-" * 70)

    print(
        f"ROC-AUC: {auc:.6f}"
    )

    print(
        f"Average Precision: {ap:.6f}"
    )

else:

    print(
        "\nROC-AUC cannot be calculated because "
        "validation contains only one class."
    )


# ============================================================
# 9. ACTUAL SPIKE FORECAST VALUES
# ============================================================

spike_rows = df[
    df["actual_spike"] == 1
].copy()

print("\n" + "-" * 70)
print("FORECASTS ON ACTUAL SPIKE DAYS")
print("-" * 70)

print(
    f"Number of actual spikes: "
    f"{len(spike_rows)}"
)

print(
    f"Mean CNN forecast on actual spikes: "
    f"{spike_rows['predicted_volatility'].mean():.8f}"
)

print(
    f"Maximum CNN forecast on actual spikes: "
    f"{spike_rows['predicted_volatility'].max():.8f}"
)


# ============================================================
# 10. COMPARISON WITH NON-SPIKES
# ============================================================

non_spike_rows = df[
    df["actual_spike"] == 0
].copy()

print("\n" + "-" * 70)
print("SPIKE VS NON-SPIKE FORECASTS")
print("-" * 70)

print(
    f"Mean forecast on NON-SPIKES: "
    f"{non_spike_rows['predicted_volatility'].mean():.8f}"
)

print(
    f"Mean forecast on SPIKES:     "
    f"{spike_rows['predicted_volatility'].mean():.8f}"
)


# ============================================================
# 11. FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)

print(
    "\nThis diagnostic does not modify the model "
    "or the test set."
)

print(
    "It only determines whether the current CNN "
    "contains useful spike-ranking information."
)