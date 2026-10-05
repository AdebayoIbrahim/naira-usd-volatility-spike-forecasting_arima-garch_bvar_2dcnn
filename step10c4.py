#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 17:42:49 2026

@author: setup
"""

# ============================================================
# STEP 10C-V4
# CNN SPIKE-PROBABILITY THRESHOLD SELECTION
#
# Uses VALIDATION data only.
#
# Validation:
#     2014-01-02 -> 2020-12-31
#
# Final test:
#     2021-01-04 -> 2025-10-29
#
# The final test set is NOT used here.
#
# Selection criterion:
#     Maximum validation F1
#
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os

import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

VALIDATION_FILE = (
    "results/"
    "cnn_2d_multitask_validation_predictions.csv"
)

OUTPUT_FILE = (
    "results/"
    "cnn_threshold_search_validation.csv"
)

THRESHOLD_FILE = (
    "results/"
    "cnn_selected_threshold.txt"
)

SPIKE_THRESHOLD = 0.10927724

# Search from 0.01 to 0.99
THRESHOLD_GRID = np.arange(
    0.01,
    1.00,
    0.01
)


# ============================================================
# 3. HEADER
# ============================================================

print("=" * 70)
print("STEP 10C-V4: CNN SPIKE THRESHOLD SELECTION")
print("=" * 70)


# ============================================================
# 4. LOAD VALIDATION PREDICTIONS
# ============================================================

print("\nLoading validation predictions...")

df = pd.read_csv(
    VALIDATION_FILE
)

print(
    f"Validation observations: {len(df)}"
)


# ============================================================
# 5. REQUIRED COLUMNS
# ============================================================

required_columns = [
    "target_date",
    "actual_volatility",
    "predicted_volatility",
    "actual_spike",
    "predicted_spike_probability"
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
# 6. EXTRACT ACTUAL LABELS AND PROBABILITIES
# ============================================================

y_true = df[
    "actual_spike"
].astype(int).to_numpy()


y_probability = df[
    "predicted_spike_probability"
].astype(float).to_numpy()


# ============================================================
# 7. BASIC CHECKS
# ============================================================

if np.isnan(y_probability).any():

    raise ValueError(
        "NaN values found in spike probabilities."
    )


if np.isinf(y_probability).any():

    raise ValueError(
        "Infinite values found in spike probabilities."
    )


if np.any(
    (y_probability < 0)
    |
    (y_probability > 1)
):

    raise ValueError(
        "Spike probabilities must be between 0 and 1."
    )


actual_spike_count = int(
    y_true.sum()
)


print(
    f"\nActual validation spikes: "
    f"{actual_spike_count}"
)


print(
    f"Validation spike rate: "
    f"{actual_spike_count / len(y_true):.4%}"
)


# ============================================================
# 8. BASELINE AUC
# ============================================================
#
# AUC does NOT depend on the selected classification threshold.
#
# It tells us whether the probability score ranks spike days
# above non-spike days.
#
# ============================================================

roc_auc = roc_auc_score(
    y_true,
    y_probability
)


pr_auc = average_precision_score(
    y_true,
    y_probability
)


print("\n" + "-" * 70)
print("THRESHOLD-INDEPENDENT RANKING")
print("-" * 70)

print(
    f"ROC-AUC: "
    f"{roc_auc:.6f}"
)

print(
    f"PR-AUC:  "
    f"{pr_auc:.6f}"
)


# ============================================================
# 9. SEARCH OVER PROBABILITY THRESHOLDS
# ============================================================

results = []


for threshold in THRESHOLD_GRID:

    y_pred = (
        y_probability >= threshold
    ).astype(int)


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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


    # Specificity:
    #
    # TN / (TN + FP)
    #

    specificity_denominator = (
        tn + fp
    )


    if specificity_denominator > 0:

        specificity = (
            tn
            / specificity_denominator
        )

    else:

        specificity = 0.0


    predicted_spikes = int(
        y_pred.sum()
    )


    predicted_spike_rate = (
        predicted_spikes
        / len(y_pred)
    )


    results.append({

        "threshold":
            threshold,

        "tn":
            tn,

        "fp":
            fp,

        "fn":
            fn,

        "tp":
            tp,

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "specificity":
            specificity,

        "f1":
            f1,

        "predicted_spikes":
            predicted_spikes,

        "predicted_spike_rate":
            predicted_spike_rate

    })


# ============================================================
# 10. CONVERT RESULTS TO DATAFRAME
# ============================================================

threshold_results = pd.DataFrame(
    results
)


# ============================================================
# 11. FIND BEST F1 THRESHOLD
# ============================================================
#
# PRIMARY SELECTION CRITERION:
#
#     maximum F1
#
# If multiple thresholds have exactly the same F1,
# select the one with higher recall.
#
# If still tied, select the lower threshold.
#
# ============================================================

threshold_results = (
    threshold_results
    .sort_values(
        [
            "f1",
            "recall",
            "threshold"
        ],
        ascending=[
            False,
            False,
            True
        ]
    )
    .reset_index(drop=True)
)


best_row = threshold_results.iloc[0]


selected_threshold = float(
    best_row["threshold"]
)


# ============================================================
# 12. DISPLAY BEST THRESHOLD
# ============================================================

print("\n" + "=" * 70)
print("SELECTED VALIDATION THRESHOLD")
print("=" * 70)


print(
    f"\nSelected probability threshold: "
    f"{selected_threshold:.2f}"
)


print(
    f"F1: "
    f"{best_row['f1']:.6f}"
)


print(
    f"Precision: "
    f"{best_row['precision']:.6f}"
)


print(
    f"Recall: "
    f"{best_row['recall']:.6f}"
)


print(
    f"Specificity: "
    f"{best_row['specificity']:.6f}"
)


print(
    f"Accuracy: "
    f"{best_row['accuracy']:.6f}"
)


print(
    f"Predicted spikes: "
    f"{int(best_row['predicted_spikes'])}"
)


print(
    f"Predicted spike rate: "
    f"{best_row['predicted_spike_rate']:.4%}"
)


print(
    f"\nActual spikes: "
    f"{actual_spike_count}"
)


# ============================================================
# 13. DISPLAY CONFUSION MATRIX VALUES
# ============================================================

print("\n" + "-" * 70)
print("SELECTED CONFUSION MATRIX")
print("-" * 70)


print(
    f"TN = {int(best_row['tn'])}"
)

print(
    f"FP = {int(best_row['fp'])}"
)

print(
    f"FN = {int(best_row['fn'])}"
)

print(
    f"TP = {int(best_row['tp'])}"
)


# ============================================================
# 14. SHOW TOP 15 THRESHOLDS
# ============================================================
#
# This lets us see whether the optimum is stable or whether
# F1 changes sharply around the selected threshold.
#
# ============================================================

print("\n" + "-" * 70)
print("TOP 15 THRESHOLDS BY F1")
print("-" * 70)


display_columns = [
    "threshold",
    "precision",
    "recall",
    "specificity",
    "f1",
    "predicted_spikes"
]


print(
    threshold_results[
        display_columns
    ]
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# 15. CHECK NEIGHBORING THRESHOLDS
# ============================================================
#
# Examine the thresholds immediately around the selected one.
#
# ============================================================

print("\n" + "-" * 70)
print("NEIGHBORING THRESHOLDS")
print("-" * 70)


neighbor_mask = (
    threshold_results["threshold"]
    .between(
        max(
            0.01,
            selected_threshold - 0.05
        ),

        min(
            0.99,
            selected_threshold + 0.05
        )
    )
)


neighbor_results = (
    threshold_results[
        neighbor_mask
    ]
    .sort_values(
        "threshold"
    )
)


print(
    neighbor_results[
        display_columns
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 16. CHECK ORIGINAL 0.50 CUTOFF
# ============================================================

original_threshold = 0.50


original_row = (
    threshold_results[
        np.isclose(
            threshold_results["threshold"],
            original_threshold
        )
    ]
)


if len(original_row) == 1:

    original_row = (
        original_row.iloc[0]
    )

    print("\n" + "-" * 70)
    print("COMPARISON WITH ORIGINAL 0.50 CUTOFF")
    print("-" * 70)

    print(
        f"0.50 F1: "
        f"{original_row['f1']:.6f}"
    )

    print(
        f"0.50 Precision: "
        f"{original_row['precision']:.6f}"
    )

    print(
        f"0.50 Recall: "
        f"{original_row['recall']:.6f}"
    )

    print(
        f"0.50 predicted spikes: "
        f"{int(original_row['predicted_spikes'])}"
    )


# ============================================================
# 17. SAVE FULL THRESHOLD SEARCH
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


threshold_results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 18. SAVE SELECTED THRESHOLD
# ============================================================

with open(
    THRESHOLD_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        f"{selected_threshold:.2f}\n"
    )


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 10C-V4 FINISHED SUCCESSFULLY")
print("=" * 70)


print(
    f"\nSelected threshold: "
    f"{selected_threshold:.2f}"
)


print(
    "\nSelection criterion:"
)

print(
    "Maximum validation F1"
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
    "\nSaved threshold search:"
)

print(
    OUTPUT_FILE
)


print(
    "\nSaved selected threshold:"
)

print(
    THRESHOLD_FILE
)


print(
    "\nThe 2021–2025 test set remains untouched."
)


print(
    "\nNext: freeze this threshold and evaluate "
    "the CNN on the final 2021–2025 test set."
)