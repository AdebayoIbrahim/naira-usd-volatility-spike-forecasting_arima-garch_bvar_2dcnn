#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 17:54:51 2026

@author: setup
"""

# ============================================================
# STEP 10D
# FINAL CNN TEST EVALUATION
#
# FROZEN MODEL:
#     models/cnn_2d_multitask_model.keras
#
# FROZEN CNN DECISION THRESHOLD:
#     0.65
#
# FROZEN VOLATILITY SPIKE THRESHOLD:
#     0.10927724
#
# FINAL TEST PERIOD:
#     2021-01-04 -> 2025-10-29
#
# IMPORTANT:
#     NO TRAINING
#     NO VALIDATION
#     NO THRESHOLD TUNING
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss
)


# ============================================================
# 2. PATHS
# ============================================================

DATA_DIR = "data_cleaned"
RESULTS_DIR = "results"
MODEL_DIR = "models"


TEST_X_FILE = os.path.join(
    DATA_DIR,
    "cnn_X_test_scaled.npy"
)

TEST_Y_FILE = os.path.join(
    DATA_DIR,
    "cnn_y_test_volatility.npy"
)

TEST_META_FILE = os.path.join(
    DATA_DIR,
    "cnn_test_metadata.csv"
)


MODEL_FILE = os.path.join(
    MODEL_DIR,
    "cnn_2d_multitask_model.keras"
)

THRESHOLD_FILE = os.path.join(
    RESULTS_DIR,
    "cnn_selected_threshold.txt"
)


OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "cnn_final_test_predictions.csv"
)

METRICS_FILE = os.path.join(
    RESULTS_DIR,
    "cnn_final_test_metrics.csv"
)


# ============================================================
# 3. FROZEN VALUES
# ============================================================

SPIKE_THRESHOLD = 0.10927724


# ============================================================
# 4. HEADER
# ============================================================

print("=" * 70)
print("STEP 10D: FINAL CNN TEST EVALUATION")
print("=" * 70)

print("\nThis step performs FINAL evaluation only.")

print("No model training will occur.")
print("No threshold selection will occur.")


# ============================================================
# 5. LOAD TEST INPUTS
# ============================================================

print("\nLoading final test inputs...")

X_test = np.load(
    TEST_X_FILE
)

print(
    f"Test input shape: {X_test.shape}"
)


# ============================================================
# 6. LOAD TEST TARGET
# ============================================================

print("\nLoading final test volatility targets...")

y_test_vol = np.load(
    TEST_Y_FILE
)

print(
    f"Test target shape: {y_test_vol.shape}"
)


# ============================================================
# 7. LOAD TEST METADATA
# ============================================================

print("\nLoading test metadata...")

test_meta = pd.read_csv(
    TEST_META_FILE
)

test_meta["target_date"] = pd.to_datetime(
    test_meta["target_date"]
)

test_meta = test_meta.reset_index(
    drop=True
)


# ============================================================
# 8. VALIDATE TEST DATA
# ============================================================

print("\n" + "-" * 70)
print("TEST DATA VALIDATION")
print("-" * 70)


if X_test.ndim != 4:
    raise ValueError(
        "Test input must have 4 dimensions."
    )


if X_test.shape[1:] != (4, 20, 1):
    raise ValueError(
        "Expected test input shape "
        "(samples, 4, 20, 1)."
    )


if len(X_test) != len(y_test_vol):
    raise ValueError(
        "Test X and y lengths do not match."
    )


if len(X_test) != len(test_meta):
    raise ValueError(
        "Test data and metadata lengths do not match."
    )


if np.isnan(X_test).any():
    raise ValueError(
        "NaN detected in test inputs."
    )


if np.isinf(X_test).any():
    raise ValueError(
        "Inf detected in test inputs."
    )


if np.isnan(y_test_vol).any():
    raise ValueError(
        "NaN detected in test target."
    )


if np.any(y_test_vol < 0):
    raise ValueError(
        "Negative volatility detected."
    )


print(
    "PASS: Final test data is valid."
)


# ============================================================
# 9. LOAD FROZEN CNN THRESHOLD
# ============================================================

print("\nLoading frozen CNN classification threshold...")

if not os.path.exists(THRESHOLD_FILE):
    raise FileNotFoundError(
        f"Threshold file not found: {THRESHOLD_FILE}"
    )


with open(
    THRESHOLD_FILE,
    "r",
    encoding="utf-8"
) as file:

    classification_threshold = float(
        file.read().strip()
    )


print(
    f"Frozen CNN probability threshold: "
    f"{classification_threshold:.2f}"
)


# Safety check
if not np.isclose(
    classification_threshold,
    0.65
):

    raise ValueError(
        "The saved threshold is not the expected "
        "frozen validation threshold of 0.65."
    )


# ============================================================
# 10. LOAD FROZEN MODEL
# ============================================================

print("\nLoading frozen CNN model...")

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_FILE}"
    )


model = tf.keras.models.load_model(
    MODEL_FILE
)


print(
    "PASS: Frozen CNN loaded."
)


# ============================================================
# 11. GENERATE FINAL TEST PREDICTIONS
# ============================================================
#
# IMPORTANT:
# model.predict() only makes predictions.
# It does NOT train the model.
#
# ============================================================

print("\n" + "=" * 70)
print("GENERATING FINAL TEST PREDICTIONS")
print("=" * 70)


predictions = model.predict(
    X_test,
    verbose=1
)


# ============================================================
# 12. EXTRACT MODEL OUTPUTS
# ============================================================
#
# Output 1:
#     volatility forecast in log1p scale
#
# Output 2:
#     spike probability
#
# ============================================================

test_pred_logvol = (
    predictions[0]
    .reshape(-1)
)


test_pred_spike_probability = (
    predictions[1]
    .reshape(-1)
)


# ============================================================
# 13. RETURN VOLATILITY TO ORIGINAL SCALE
# ============================================================

test_pred_volatility = np.expm1(
    test_pred_logvol
)


# Ensure non-negative forecasts
test_pred_volatility = np.maximum(
    test_pred_volatility,
    0.0
)


# ============================================================
# 14. CREATE ACTUAL SPIKE LABELS
# ============================================================
#
# Actual spike definition:
#
#     v_t > 0.10927724
#
# ============================================================

y_test_spike = (
    y_test_vol
    > SPIKE_THRESHOLD
).astype(np.int32)


# ============================================================
# 15. CREATE CNN SPIKE PREDICTIONS
# ============================================================
#
# Frozen CNN probability threshold:
#
#     P(spike) >= 0.65
#
# ============================================================

y_test_pred_spike = (
    test_pred_spike_probability
    >= classification_threshold
).astype(np.int32)


# ============================================================
# 16. TEST PERIOD
# ============================================================

print("\n" + "-" * 70)
print("FINAL TEST PERIOD")
print("-" * 70)


print(
    f"First test date: "
    f"{test_meta['target_date'].min()}"
)


print(
    f"Last test date: "
    f"{test_meta['target_date'].max()}"
)


print(
    f"Total test observations: "
    f"{len(y_test_vol)}"
)


# ============================================================
# 17. SPIKE COUNTS
# ============================================================

actual_spikes = int(
    y_test_spike.sum()
)

actual_nonspikes = int(
    len(y_test_spike)
    - actual_spikes
)


predicted_spikes = int(
    y_test_pred_spike.sum()
)

predicted_nonspikes = int(
    len(y_test_pred_spike)
    - predicted_spikes
)


print("\n" + "-" * 70)
print("TEST SPIKE DISTRIBUTION")
print("-" * 70)


print(
    f"Actual non-spikes: "
    f"{actual_nonspikes}"
)


print(
    f"Actual spikes:     "
    f"{actual_spikes}"
)


print(
    f"Actual spike rate: "
    f"{actual_spikes / len(y_test_spike):.4%}"
)


print(
    f"\nPredicted non-spikes: "
    f"{predicted_nonspikes}"
)


print(
    f"Predicted spikes:     "
    f"{predicted_spikes}"
)


print(
    f"Predicted spike rate: "
    f"{predicted_spikes / len(y_test_pred_spike):.4%}"
)


# ============================================================
# 18. CONTINUOUS VOLATILITY METRICS
# ============================================================

mae = mean_absolute_error(
    y_test_vol,
    test_pred_volatility
)


mse = mean_squared_error(
    y_test_vol,
    test_pred_volatility
)


rmse = np.sqrt(
    mse
)


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
    f"\nActual mean volatility: "
    f"{np.mean(y_test_vol):.8f}"
)


print(
    f"Predicted mean volatility: "
    f"{np.mean(test_pred_volatility):.8f}"
)


print(
    f"Actual median volatility: "
    f"{np.median(y_test_vol):.8f}"
)


print(
    f"Predicted median volatility: "
    f"{np.median(test_pred_volatility):.8f}"
)


print(
    f"Actual maximum volatility: "
    f"{np.max(y_test_vol):.8f}"
)


print(
    f"Predicted maximum volatility: "
    f"{np.max(test_pred_volatility):.8f}"
)


# ============================================================
# 19. CONFUSION MATRIX
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    y_test_spike,
    y_test_pred_spike,
    labels=[0, 1]
).ravel()


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
# 20. CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_test_spike,
    y_test_pred_spike
)


balanced_accuracy = balanced_accuracy_score(
    y_test_spike,
    y_test_pred_spike
)


precision = precision_score(
    y_test_spike,
    y_test_pred_spike,
    zero_division=0
)


recall = recall_score(
    y_test_spike,
    y_test_pred_spike,
    zero_division=0
)


f1 = f1_score(
    y_test_spike,
    y_test_pred_spike,
    zero_division=0
)


# Specificity
if (
    tn + fp
) > 0:

    specificity = (
        tn / (tn + fp)
    )

else:

    specificity = 0.0


print("\n" + "-" * 70)
print("FINAL SPIKE CLASSIFICATION METRICS")
print("-" * 70)


print(
    f"Accuracy:          {accuracy:.6f}"
)


print(
    f"Balanced Accuracy: {balanced_accuracy:.6f}"
)


print(
    f"Precision:         {precision:.6f}"
)


print(
    f"Recall:            {recall:.6f}"
)


print(
    f"Specificity:       {specificity:.6f}"
)


print(
    f"F1:                {f1:.6f}"
)


# ============================================================
# 21. THRESHOLD-INDEPENDENT METRICS
# ============================================================
#
# These use the CNN probability scores directly.
#
# No probability cutoff is needed for AUC.
#
# ============================================================

if len(
    np.unique(y_test_spike)
) == 2:

    roc_auc = roc_auc_score(
        y_test_spike,
        test_pred_spike_probability
    )


    pr_auc = average_precision_score(
        y_test_spike,
        test_pred_spike_probability
    )


    brier = brier_score_loss(
        y_test_spike,
        test_pred_spike_probability
    )


else:

    roc_auc = np.nan
    pr_auc = np.nan
    brier = np.nan


print("\n" + "-" * 70)
print("THRESHOLD-INDEPENDENT METRICS")
print("-" * 70)


print(
    f"ROC-AUC:    {roc_auc:.6f}"
)


print(
    f"PR-AUC:     {pr_auc:.6f}"
)


print(
    f"Brier Score:{brier:.6f}"
)


# ============================================================
# 22. TEST PROBABILITY DISTRIBUTION
# ============================================================

print("\n" + "-" * 70)
print("TEST SPIKE-PROBABILITY DISTRIBUTION")
print("-" * 70)


for q in [
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    1.00
]:

    value = np.quantile(
        test_pred_spike_probability,
        q
    )

    print(
        f"{q:.0%} percentile: "
        f"{value:.6f}"
    )


# ============================================================
# 23. SPIKE VS NON-SPIKE PROBABILITIES
# ============================================================

actual_spike_probabilities = (
    test_pred_spike_probability[
        y_test_spike == 1
    ]
)


actual_nonspike_probabilities = (
    test_pred_spike_probability[
        y_test_spike == 0
    ]
)


print("\n" + "-" * 70)
print("SPIKE VS NON-SPIKE CNN PROBABILITIES")
print("-" * 70)


print(
    f"Mean probability on actual spikes: "
    f"{np.mean(actual_spike_probabilities):.8f}"
)


print(
    f"Mean probability on non-spikes: "
    f"{np.mean(actual_nonspike_probabilities):.8f}"
)


# ============================================================
# 24. SPIKE VS NON-SPIKE VOLATILITY FORECASTS
# ============================================================

actual_spike_vol_forecasts = (
    test_pred_volatility[
        y_test_spike == 1
    ]
)


actual_nonspike_vol_forecasts = (
    test_pred_volatility[
        y_test_spike == 0
    ]
)


print("\n" + "-" * 70)
print("SPIKE VS NON-SPIKE VOLATILITY FORECASTS")
print("-" * 70)


print(
    f"Mean volatility forecast on actual spikes: "
    f"{np.mean(actual_spike_vol_forecasts):.8f}"
)


print(
    f"Mean volatility forecast on non-spikes: "
    f"{np.mean(actual_nonspike_vol_forecasts):.8f}"
)


# ============================================================
# 25. FIRST 20 TEST PREDICTIONS
# ============================================================

print("\n" + "-" * 70)
print("FIRST 20 TEST PREDICTIONS")
print("-" * 70)


preview = pd.DataFrame({

    "target_date":
        test_meta[
            "target_date"
        ].iloc[:20],

    "actual_volatility":
        y_test_vol[:20],

    "predicted_volatility":
        test_pred_volatility[:20],

    "actual_spike":
        y_test_spike[:20],

    "predicted_spike_probability":
        test_pred_spike_probability[:20],

    "predicted_spike":
        y_test_pred_spike[:20]

})


print(
    preview.to_string(
        index=False
    )
)


# ============================================================
# 26. SAVE ALL TEST PREDICTIONS
# ============================================================

test_results = pd.DataFrame({

    "target_date":
        test_meta["target_date"],

    "actual_volatility":
        y_test_vol,

    "predicted_volatility":
        test_pred_volatility,

    "actual_spike":
        y_test_spike,

    "predicted_spike_probability":
        test_pred_spike_probability,

    "predicted_spike":
        y_test_pred_spike

})


test_results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 27. SAVE FINAL METRICS
# ============================================================

metrics = pd.DataFrame({

    "metric": [

        "MAE",

        "MSE",

        "RMSE",

        "Accuracy",

        "Balanced Accuracy",

        "Precision",

        "Recall",

        "Specificity",

        "F1",

        "ROC-AUC",

        "PR-AUC",

        "Brier Score",

        "Actual Spikes",

        "Predicted Spikes",

        "Actual Spike Rate",

        "Predicted Spike Rate",

        "Volatility Spike Threshold",

        "CNN Probability Threshold"

    ],

    "value": [

        mae,

        mse,

        rmse,

        accuracy,

        balanced_accuracy,

        precision,

        recall,

        specificity,

        f1,

        roc_auc,

        pr_auc,

        brier,

        actual_spikes,

        predicted_spikes,

        actual_spikes / len(y_test_spike),

        predicted_spikes / len(y_test_pred_spike),

        SPIKE_THRESHOLD,

        classification_threshold

    ]

})


metrics.to_csv(
    METRICS_FILE,
    index=False
)


# ============================================================
# 28. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 10D FINISHED SUCCESSFULLY")
print("=" * 70)


print(
    "\nFinal test period:"
)

print(
    f"{test_meta['target_date'].min()} "
    f"→ "
    f"{test_meta['target_date'].max()}"
)


print(
    "\nFrozen CNN probability threshold:"
)

print(
    f"{classification_threshold:.2f}"
)


print(
    "\nFrozen volatility spike threshold:"
)

print(
    f"{SPIKE_THRESHOLD:.8f}"
)


print(
    "\nROC-AUC:"
)

print(
    f"{roc_auc:.6f}"
)


print(
    "\nPR-AUC:"
)

print(
    f"{pr_auc:.6f}"
)


print(
    "\nF1:"
)

print(
    f"{f1:.6f}"
)


print(
    "\nPrecision:"
)

print(
    f"{precision:.6f}"
)


print(
    "\nRecall:"
)

print(
    f"{recall:.6f}"
)


print(
    "\nRMSE:"
)

print(
    f"{rmse:.6f}"
)


print(
    "\nSaved predictions:"
)

print(
    OUTPUT_FILE
)


print(
    "\nSaved metrics:"
)

print(
    METRICS_FILE
)


print("\n" + "=" * 70)
print("TEST SET REMAINED COMPLETELY OUTSIDE MODEL SELECTION")
print("=" * 70)