#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 17:46:07 2026

@author: setup
"""

# -*- coding: utf-8 -*-

"""
Step 08G: Exact ARIMA-GARCH vs Persistence Comparison

Purpose:
    Compare the ARIMA-GARCH model and the persistence baseline
    on EXACTLY the same test target dates.

This avoids comparing models on different numbers of observations.

Models:
    1. ARIMA(0,0,1)-GARCH(2,1)-t
    2. Persistence baseline

Persistence definition:
    predicted_volatility(t+1) = observed_squared_return(t)

Metrics:
    Continuous:
        MAE
        MSE
        RMSE

    Spike classification:
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

ARIMA_FILE = (
    "results/arima_garch_spike_evaluated_forecasts.csv"
)

PERSISTENCE_FILE = (
    "results/persistence_baseline_spike_forecasts.csv"
)

THRESHOLD_FILE = (
    "results/training_spike_threshold_new.csv"
)

COMPARISON_OUTPUT = (
    "results/arima_garch_vs_persistence_comparison.csv"
)

COMMON_FORECAST_OUTPUT = (
    "results/arima_garch_vs_persistence_common_forecasts.csv"
)


# ============================================================
# 2. CHECK FILES
# ============================================================

for file_path in [
    ARIMA_FILE,
    PERSISTENCE_FILE,
    THRESHOLD_FILE
]:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )


# ============================================================
# 3. LOAD DATA
# ============================================================

arima_df = pd.read_csv(
    ARIMA_FILE
)

persistence_df = pd.read_csv(
    PERSISTENCE_FILE
)

threshold_df = pd.read_csv(
    THRESHOLD_FILE
)


print("=" * 90)
print("STEP 08G: EXACT ARIMA-GARCH VS PERSISTENCE COMPARISON")
print("=" * 90)

print(
    f"\nARIMA-GARCH rows: "
    f"{len(arima_df):,}"
)

print(
    f"Persistence rows: "
    f"{len(persistence_df):,}"
)


# ============================================================
# 4. LOAD FROZEN THRESHOLD
# ============================================================

SPIKE_THRESHOLD = float(
    threshold_df.loc[
        0,
        "threshold_r_squared"
    ]
)


print(
    f"Frozen threshold: "
    f"{SPIKE_THRESHOLD:.8f}"
)


# ============================================================
# 5. VALIDATE REQUIRED COLUMNS
# ============================================================

arima_required = [
    "target_date",
    "forecast_squared_return",
    "actual_squared_return",
    "actual_spike"
]

persistence_required = [
    "target_date",
    "forecast_squared_return",
    "actual_squared_return",
    "actual_spike"
]


missing_arima = [
    col
    for col in arima_required
    if col not in arima_df.columns
]

missing_persistence = [
    col
    for col in persistence_required
    if col not in persistence_df.columns
]


if missing_arima:

    raise ValueError(
        f"Missing ARIMA-GARCH columns: "
        f"{missing_arima}"
    )


if missing_persistence:

    raise ValueError(
        f"Missing persistence columns: "
        f"{missing_persistence}"
    )


# ============================================================
# 6. CLEAN DATE COLUMNS
# ============================================================

arima_df["target_date"] = pd.to_datetime(
    arima_df["target_date"],
    errors="coerce"
)

persistence_df["target_date"] = pd.to_datetime(
    persistence_df["target_date"],
    errors="coerce"
)


# ============================================================
# 7. SORT DATA
# ============================================================

arima_df = arima_df.sort_values(
    "target_date"
).copy()

persistence_df = persistence_df.sort_values(
    "target_date"
).copy()


# ============================================================
# 8. SELECT REQUIRED COLUMNS
# ============================================================

arima_compare = arima_df[
    [
        "target_date",
        "forecast_squared_return",
        "actual_squared_return",
        "actual_spike"
    ]
].copy()


arima_compare = arima_compare.rename(
    columns={
        "forecast_squared_return":
            "arima_garch_forecast",

        "actual_squared_return":
            "arima_garch_actual_squared_return"
    }
)


persistence_compare = persistence_df[
    [
        "target_date",
        "forecast_squared_return",
        "actual_squared_return",
        "actual_spike"
    ]
].copy()


persistence_compare = persistence_compare.rename(
    columns={
        "forecast_squared_return":
            "persistence_forecast",

        "actual_squared_return":
            "persistence_actual_squared_return",

        "actual_spike":
            "persistence_actual_spike"
    }
)


# ============================================================
# 9. INNER JOIN ON EXACT TARGET DATE
# ============================================================

common_df = pd.merge(
    arima_compare,
    persistence_compare,
    on="target_date",
    how="inner"
)


print("\n" + "=" * 90)
print("COMMON TEST SAMPLE")
print("=" * 90)

print(
    f"Common target dates: "
    f"{len(common_df):,}"
)


# ============================================================
# 10. CHECK ACTUAL VALUES AGREE
# ============================================================

actual_difference = (
    common_df[
        "arima_garch_actual_squared_return"
    ]

    -

    common_df[
        "persistence_actual_squared_return"
    ]
).abs()


max_actual_difference = (
    actual_difference.max()
)


print(
    f"Maximum difference in actual squared returns: "
    f"{max_actual_difference:.12f}"
)


if max_actual_difference > 1e-10:

    raise RuntimeError(
        "The actual target values do not match "
        "between the two forecast files."
    )


# ============================================================
# 11. CHECK ACTUAL SPIKE LABELS
# ============================================================

common_df["actual_spike"] = (
    common_df["arima_garch_actual_spike"]
    if "arima_garch_actual_spike"
    in common_df.columns
    else arima_df.set_index(
        "target_date"
    ).loc[
        common_df["target_date"],
        "actual_spike"
    ].to_numpy()
)


# ============================================================
# 12. RECREATE SPIKE LABELS FROM THRESHOLD
# ============================================================

common_df["actual_spike"] = (
    common_df[
        "arima_garch_actual_squared_return"
    ]
    > SPIKE_THRESHOLD
).astype(int)


common_df["arima_garch_predicted_spike"] = (
    common_df[
        "arima_garch_forecast"
    ]
    > SPIKE_THRESHOLD
).astype(int)


common_df["persistence_predicted_spike"] = (
    common_df[
        "persistence_forecast"
    ]
    > SPIKE_THRESHOLD
).astype(int)


# ============================================================
# 13. FUNCTION TO CALCULATE MODEL METRICS
# ============================================================

def calculate_metrics(
    actual_values,
    predicted_values,
    actual_labels,
    predicted_labels
):

    errors = (
        predicted_values
        - actual_values
    )

    mae = np.mean(
        np.abs(errors)
    )

    mse = np.mean(
        errors ** 2
    )

    rmse = np.sqrt(
        mse
    )

    accuracy = accuracy_score(
        actual_labels,
        predicted_labels
    )

    precision = precision_score(
        actual_labels,
        predicted_labels,
        zero_division=0
    )

    recall = recall_score(
        actual_labels,
        predicted_labels,
        zero_division=0
    )

    f1 = f1_score(
        actual_labels,
        predicted_labels,
        zero_division=0
    )

    auc = roc_auc_score(
        actual_labels,
        predicted_values
    )

    cm = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": auc,
        "Specificity": specificity,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


# ============================================================
# 14. CALCULATE ARIMA-GARCH METRICS
# ============================================================

actual_values = common_df[
    "arima_garch_actual_squared_return"
].to_numpy(
    dtype=float
)


arima_predictions = common_df[
    "arima_garch_forecast"
].to_numpy(
    dtype=float
)


arima_labels = common_df[
    "arima_garch_predicted_spike"
].to_numpy(
    dtype=int
)


actual_labels = common_df[
    "actual_spike"
].to_numpy(
    dtype=int
)


arima_metrics = calculate_metrics(
    actual_values,
    arima_predictions,
    actual_labels,
    arima_labels
)


# ============================================================
# 15. CALCULATE PERSISTENCE METRICS
# ============================================================

persistence_predictions = common_df[
    "persistence_forecast"
].to_numpy(
    dtype=float
)


persistence_labels = common_df[
    "persistence_predicted_spike"
].to_numpy(
    dtype=int
)


persistence_metrics = calculate_metrics(
    actual_values,
    persistence_predictions,
    actual_labels,
    persistence_labels
)


# ============================================================
# 16. CREATE COMPARISON TABLE
# ============================================================

comparison_df = pd.DataFrame({

    "model": [

        "ARIMA(0,0,1)-GARCH(2,1)-t",

        "Persistence baseline"
    ],

    "test_observations": [

        len(common_df),

        len(common_df)
    ],

    "MAE": [

        arima_metrics["MAE"],

        persistence_metrics["MAE"]
    ],

    "MSE": [

        arima_metrics["MSE"],

        persistence_metrics["MSE"]
    ],

    "RMSE": [

        arima_metrics["RMSE"],

        persistence_metrics["RMSE"]
    ],

    "Accuracy": [

        arima_metrics["Accuracy"],

        persistence_metrics["Accuracy"]
    ],

    "Precision": [

        arima_metrics["Precision"],

        persistence_metrics["Precision"]
    ],

    "Recall": [

        arima_metrics["Recall"],

        persistence_metrics["Recall"]
    ],

    "F1": [

        arima_metrics["F1"],

        persistence_metrics["F1"]
    ],

    "ROC_AUC": [

        arima_metrics["ROC_AUC"],

        persistence_metrics["ROC_AUC"]
    ],

    "Specificity": [

        arima_metrics["Specificity"],

        persistence_metrics["Specificity"]
    ],

    "TN": [

        arima_metrics["TN"],

        persistence_metrics["TN"]
    ],

    "FP": [

        arima_metrics["FP"],

        persistence_metrics["FP"]
    ],

    "FN": [

        arima_metrics["FN"],

        persistence_metrics["FN"]
    ],

    "TP": [

        arima_metrics["TP"],

        persistence_metrics["TP"]
    ]
})


# ============================================================
# 17. CALCULATE ARIMA-GARCH IMPROVEMENT
# ============================================================

difference_df = pd.DataFrame({

    "comparison": [
        "ARIMA-GARCH minus Persistence"
    ],

    "MAE_difference": [
        arima_metrics["MAE"]
        - persistence_metrics["MAE"]
    ],

    "RMSE_difference": [
        arima_metrics["RMSE"]
        - persistence_metrics["RMSE"]
    ],

    "Accuracy_difference": [
        arima_metrics["Accuracy"]
        - persistence_metrics["Accuracy"]
    ],

    "Precision_difference": [
        arima_metrics["Precision"]
        - persistence_metrics["Precision"]
    ],

    "Recall_difference": [
        arima_metrics["Recall"]
        - persistence_metrics["Recall"]
    ],

    "F1_difference": [
        arima_metrics["F1"]
        - persistence_metrics["F1"]
    ],

    "ROC_AUC_difference": [
        arima_metrics["ROC_AUC"]
        - persistence_metrics["ROC_AUC"]
    ],

    "Specificity_difference": [
        arima_metrics["Specificity"]
        - persistence_metrics["Specificity"]
    ]
})


# Add comparison row
comparison_df = pd.concat(
    [
        comparison_df,
        difference_df.rename(
            columns={
                "MAE_difference": "MAE",
                "RMSE_difference": "RMSE",
                "Accuracy_difference": "Accuracy",
                "Precision_difference": "Precision",
                "Recall_difference": "Recall",
                "F1_difference": "F1",
                "ROC_AUC_difference": "ROC_AUC",
                "Specificity_difference":
                    "Specificity"
            }
        ).assign(
            model="Difference (ARIMA-GARCH - Persistence)"
        )[
            comparison_df.columns
        ]
    ],
    ignore_index=True
)


# ============================================================
# 18. PRINT RESULTS
# ============================================================

print("\n" + "=" * 90)
print("EXACT COMMON-SAMPLE COMPARISON")
print("=" * 90)

print(
    comparison_df[
        [
            "model",
            "test_observations",
            "MAE",
            "RMSE",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC_AUC",
            "Specificity",
            "TN",
            "FP",
            "FN",
            "TP"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 19. SAVE COMMON FORECASTS
# ============================================================

common_df.to_csv(
    COMMON_FORECAST_OUTPUT,
    index=False
)


# ============================================================
# 20. SAVE COMPARISON
# ============================================================

comparison_df.to_csv(
    COMPARISON_OUTPUT,
    index=False
)


# ============================================================
# 21. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 90)
print("FILES SAVED")
print("=" * 90)

print(
    f"Comparison: "
    f"{COMPARISON_OUTPUT}"
)

print(
    f"Common forecasts: "
    f"{COMMON_FORECAST_OUTPUT}"
)

print("=" * 90)