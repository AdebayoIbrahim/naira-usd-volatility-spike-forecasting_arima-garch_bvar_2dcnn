#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 23:28:23 2026

@author: setup
"""

"""
============================================================
TABLE 4.14
SPIKE CLASSIFICATION PERFORMANCE COMPARISON
============================================================

Models:
    1. ARIMA(0,0,1)-GARCH(2,1)-t
    2. BVAR(8)-log1p
    3. Multivariate 2D-CNN-A

PRIMARY OBJECTIVE:
------------------
Compare the ability of the three models to identify
volatility spikes in the Nigerian Naira/USD exchange rate.

COMMON SPIKE DEFINITION:
------------------------
target_volatility > 0.10927724

COMMON ACTUAL TARGET:
---------------------
The canonical target_volatility series from:

data_cleaned/
forecasting_dataset_2006_2025_spike_gap_target_safe.csv

MODEL DECISION RULES:
---------------------
ARIMA-GARCH:
    predicted volatility > 0.10927724

BVAR:
    predicted spike probability >= 0.50

CNN:
    predicted spike probability >= 0.328393

The model-specific cutoffs were frozen before final
out-of-sample evaluation.

OUTPUTS:
--------
results/table_4_14_spike_classification_comparison.csv
results/table_4_14_spike_classification_comparison.tex
============================================================
"""


# ============================================================
# 1. IMPORTS
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# ============================================================
# 2. PATHS
# ============================================================

RESULTS_DIR = Path("results")

DATA_DIR = Path("data_cleaned")


ARIMA_FILE = (
    RESULTS_DIR
    / "arima_garch_common_sample_forecasts.csv"
)

BVAR_FILE = (
    RESULTS_DIR
    / "bvar8_final_log1p_common_sample_forecasts.csv"
)

CNN_FILE = (
    RESULTS_DIR
    / "cnn_A_final_test_predictions.csv"
)

TARGET_FILE = (
    DATA_DIR
    / "forecasting_dataset_2006_2025_spike_gap_target_safe.csv"
)


OUTPUT_CSV = (
    RESULTS_DIR
    / "table_4_14_spike_classification_comparison.csv"
)

OUTPUT_TEX = (
    RESULTS_DIR
    / "table_4_14_spike_classification_comparison.tex"
)


# ============================================================
# 3. FROZEN THRESHOLDS
# ============================================================

SPIKE_THRESHOLD = 0.10927724

BVAR_CUTOFF = 0.50

CNN_CUTOFF = 0.328393


# ============================================================
# 4. HELPER FUNCTION
# ============================================================

def check_file(path, name):

    if not path.exists():

        raise FileNotFoundError(
            f"\n{name} file not found:\n"
            f"{path.resolve()}"
        )


# ============================================================
# 5. CHECK INPUT FILES
# ============================================================

check_file(
    ARIMA_FILE,
    "ARIMA-GARCH"
)

check_file(
    BVAR_FILE,
    "BVAR"
)

check_file(
    CNN_FILE,
    "CNN"
)

check_file(
    TARGET_FILE,
    "Canonical target"
)


# ============================================================
# 6. LOAD ARIMA-GARCH
# ============================================================

arima = pd.read_csv(
    ARIMA_FILE
)


required_arima = [
    "target_date",
    "predicted_volatility"
]


missing = [
    col
    for col in required_arima
    if col not in arima.columns
]


if missing:

    raise ValueError(
        "\nARIMA-GARCH file is missing:\n"
        f"{missing}\n\n"
        f"Available columns:\n"
        f"{list(arima.columns)}"
    )


arima = arima[
    [
        "target_date",
        "predicted_volatility"
    ]
].copy()


arima = arima.rename(
    columns={
        "predicted_volatility":
            "ARIMA_score"
    }
)


# ============================================================
# 7. LOAD BVAR
# ============================================================

bvar = pd.read_csv(
    BVAR_FILE
)


required_bvar = [
    "target_date",
    "predicted_spike_probability",
    "predicted_spike"
]


missing = [
    col
    for col in required_bvar
    if col not in bvar.columns
]


if missing:

    raise ValueError(
        "\nBVAR file is missing:\n"
        f"{missing}\n\n"
        f"Available columns:\n"
        f"{list(bvar.columns)}"
    )


bvar = bvar[
    [
        "target_date",
        "predicted_spike_probability",
        "predicted_spike"
    ]
].copy()


bvar = bvar.rename(
    columns={
        "predicted_spike_probability":
            "BVAR_probability",

        "predicted_spike":
            "BVAR_prediction"
    }
)


# ============================================================
# 8. LOAD CNN
# ============================================================

cnn = pd.read_csv(
    CNN_FILE
)


required_cnn = [
    "target_date",
    "predicted_spike_probability",
    "predicted_spike"
]


missing = [
    col
    for col in required_cnn
    if col not in cnn.columns
]


if missing:

    raise ValueError(
        "\nCNN file is missing:\n"
        f"{missing}\n\n"
        f"Available columns:\n"
        f"{list(cnn.columns)}"
    )


cnn = cnn[
    [
        "target_date",
        "predicted_spike_probability",
        "predicted_spike"
    ]
].copy()


cnn = cnn.rename(
    columns={
        "predicted_spike_probability":
            "CNN_probability",

        "predicted_spike":
            "CNN_prediction"
    }
)


# ============================================================
# 9. LOAD CANONICAL TARGET
# ============================================================

target = pd.read_csv(
    TARGET_FILE
)


required_target = [
    "target_date",
    "target_volatility"
]


missing = [
    col
    for col in required_target
    if col not in target.columns
]


if missing:

    raise ValueError(
        "\nCanonical target file is missing:\n"
        f"{missing}\n\n"
        f"Available columns:\n"
        f"{list(target.columns)}"
    )


target = target[
    [
        "target_date",
        "target_volatility"
    ]
].copy()


# ============================================================
# 10. DATE CONVERSION
# ============================================================

for df in [
    arima,
    bvar,
    cnn,
    target
]:

    df["target_date"] = pd.to_datetime(
        df["target_date"],
        errors="coerce"
    )


# ============================================================
# 11. NUMERIC CONVERSION
# ============================================================

arima["ARIMA_score"] = pd.to_numeric(
    arima["ARIMA_score"],
    errors="coerce"
)

bvar["BVAR_probability"] = pd.to_numeric(
    bvar["BVAR_probability"],
    errors="coerce"
)

cnn["CNN_probability"] = pd.to_numeric(
    cnn["CNN_probability"],
    errors="coerce"
)

target["target_volatility"] = pd.to_numeric(
    target["target_volatility"],
    errors="coerce"
)


# ============================================================
# 12. CLEAN INVALID VALUES
# ============================================================

arima = arima.dropna(
    subset=[
        "target_date",
        "ARIMA_score"
    ]
)

bvar = bvar.dropna(
    subset=[
        "target_date",
        "BVAR_probability"
    ]
)

cnn = cnn.dropna(
    subset=[
        "target_date",
        "CNN_probability"
    ]
)

target = target.dropna(
    subset=[
        "target_date",
        "target_volatility"
    ]
)


# ============================================================
# 13. REMOVE DUPLICATE DATES
# ============================================================

arima = (
    arima
    .sort_values("target_date")
    .drop_duplicates(
        "target_date"
    )
)

bvar = (
    bvar
    .sort_values("target_date")
    .drop_duplicates(
        "target_date"
    )
)

cnn = (
    cnn
    .sort_values("target_date")
    .drop_duplicates(
        "target_date"
    )
)

target = (
    target
    .sort_values("target_date")
    .drop_duplicates(
        "target_date"
    )
)


# ============================================================
# 14. CREATE COMMON SAMPLE
# ============================================================

comparison = (

    target

    .merge(
        arima,
        on="target_date",
        how="inner"
    )

    .merge(
        bvar,
        on="target_date",
        how="inner"
    )

    .merge(
        cnn,
        on="target_date",
        how="inner"
    )

)


# ============================================================
# 15. COMMON ACTUAL SPIKE
# ============================================================

comparison["actual_spike"] = (
    comparison["target_volatility"]
    > SPIKE_THRESHOLD
).astype(int)


# ============================================================
# 16. MODEL PREDICTIONS
# ============================================================

# ------------------------------------------------------------
# ARIMA-GARCH
# ------------------------------------------------------------

comparison["ARIMA_prediction"] = (
    comparison["ARIMA_score"]
    > SPIKE_THRESHOLD
).astype(int)


# ------------------------------------------------------------
# BVAR
# ------------------------------------------------------------

comparison["BVAR_prediction"] = (
    comparison["BVAR_probability"]
    >= BVAR_CUTOFF
).astype(int)


# ------------------------------------------------------------
# CNN
# ------------------------------------------------------------

comparison["CNN_prediction"] = (
    comparison["CNN_probability"]
    >= CNN_CUTOFF
).astype(int)


# ============================================================
# 17. CHECK SAMPLE
# ============================================================

comparison = (
    comparison
    .dropna()
    .sort_values("target_date")
    .reset_index(drop=True)
)


if comparison.empty:

    raise ValueError(
        "\nNo common target dates available "
        "for spike comparison."
    )


# ============================================================
# 18. SPECIFICITY FUNCTION
# ============================================================

def calculate_specificity(
    actual,
    predicted
):

    actual = np.asarray(
        actual
    )

    predicted = np.asarray(
        predicted
    )

    tn = np.sum(
        (actual == 0)
        &
        (predicted == 0)
    )

    fp = np.sum(
        (actual == 0)
        &
        (predicted == 1)
    )

    if (
        tn + fp
    ) == 0:

        return np.nan

    return tn / (
        tn + fp
    )


# ============================================================
# 19. MODEL METRICS FUNCTION
# ============================================================

def model_metrics(
    actual,
    predicted,
    score,
    model_name,
    cutoff
):

    actual = np.asarray(
        actual,
        dtype=int
    )

    predicted = np.asarray(
        predicted,
        dtype=int
    )

    score = np.asarray(
        score,
        dtype=float
    )


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    tn = int(
        np.sum(
            (actual == 0)
            &
            (predicted == 0)
        )
    )

    fp = int(
        np.sum(
            (actual == 0)
            &
            (predicted == 1)
        )
    )

    fn = int(
        np.sum(
            (actual == 1)
            &
            (predicted == 0)
        )
    )

    tp = int(
        np.sum(
            (actual == 1)
            &
            (predicted == 1)
        )
    )


    # --------------------------------------------------------
    # Classification metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        actual,
        predicted
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            actual,
            predicted
        )
    )

    precision = precision_score(
        actual,
        predicted,
        zero_division=0
    )

    recall = recall_score(
        actual,
        predicted,
        zero_division=0
    )

    specificity = calculate_specificity(
        actual,
        predicted
    )

    f1 = f1_score(
        actual,
        predicted,
        zero_division=0
    )


    # --------------------------------------------------------
    # Ranking metrics
    # --------------------------------------------------------

    roc_auc = roc_auc_score(
        actual,
        score
    )

    pr_auc = average_precision_score(
        actual,
        score
    )


    # --------------------------------------------------------
    # Return metrics
    # --------------------------------------------------------

    return {

        "Model":
            model_name,

        "Observations":
            len(actual),

        "Actual Spikes":
            int(actual.sum()),

        "Predicted Spikes":
            int(predicted.sum()),

        "Probability/Volatility Cutoff":
            cutoff,

        "Accuracy":
            accuracy,

        "Balanced Accuracy":
            balanced_accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "Specificity":
            specificity,

        "F1":
            f1,

        "ROC-AUC":
            roc_auc,

        "PR-AUC":
            pr_auc,

        "TN":
            tn,

        "FP":
            fp,

        "FN":
            fn,

        "TP":
            tp

    }


# ============================================================
# 20. COMMON ACTUAL VECTOR
# ============================================================

actual = (
    comparison["actual_spike"]
)


# ============================================================
# 21. CALCULATE METRICS
# ============================================================

results = []


# ------------------------------------------------------------
# ARIMA-GARCH
# ------------------------------------------------------------

results.append(
    model_metrics(

        actual=actual,

        predicted=(
            comparison[
                "ARIMA_prediction"
            ]
        ),

        score=(
            comparison[
                "ARIMA_score"
            ]
        ),

        model_name=(
            "ARIMA(0,0,1)-"
            "GARCH(2,1)-t"
        ),

        cutoff=SPIKE_THRESHOLD

    )
)


# ------------------------------------------------------------
# BVAR
# ------------------------------------------------------------

results.append(
    model_metrics(

        actual=actual,

        predicted=(
            comparison[
                "BVAR_prediction"
            ]
        ),

        score=(
            comparison[
                "BVAR_probability"
            ]
        ),

        model_name=(
            "BVAR(8)-log1p"
        ),

        cutoff=BVAR_CUTOFF

    )
)


# ------------------------------------------------------------
# CNN
# ------------------------------------------------------------

results.append(
    model_metrics(

        actual=actual,

        predicted=(
            comparison[
                "CNN_prediction"
            ]
        ),

        score=(
            comparison[
                "CNN_probability"
            ]
        ),

        model_name=(
            "Multivariate 2D-CNN-A"
        ),

        cutoff=CNN_CUTOFF

    )
)


results = pd.DataFrame(
    results
)


# ============================================================
# 22. RANK MODELS
# ============================================================

# Higher is better for all classification metrics.

for metric in [
    "Accuracy",
    "Balanced Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1",
    "ROC-AUC",
    "PR-AUC"
]:

    results[
        f"{metric} Rank"
    ] = (

        results[
            metric
        ]

        .rank(
            method="min",
            ascending=False
        )

        .astype(int)

    )


# ============================================================
# 23. OVERALL CLASSIFICATION RANK
# ============================================================

rank_columns = [
    "Accuracy Rank",
    "Balanced Accuracy Rank",
    "Precision Rank",
    "Recall Rank",
    "Specificity Rank",
    "F1 Rank",
    "ROC-AUC Rank",
    "PR-AUC Rank"
]


results["Average Rank"] = (
    results[
        rank_columns
    ]
    .mean(axis=1)
)


results["Overall Rank"] = (
    results[
        "Average Rank"
    ]
    .rank(
        method="min",
        ascending=True
    )
    .astype(int)
)


# ============================================================
# 24. ROUND DISPLAY VALUES
# ============================================================

display = results.copy()


metric_columns = [

    "Accuracy",
    "Balanced Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1",
    "ROC-AUC",
    "PR-AUC"

]


for col in metric_columns:

    display[col] = (
        display[col]
        .round(4)
    )


display["Average Rank"] = (
    display["Average Rank"]
    .round(2)
)


display[
    "Probability/Volatility Cutoff"
] = (
    display[
        "Probability/Volatility Cutoff"
    ]
    .round(6)
)


# ============================================================
# 25. SAVE RESULTS
# ============================================================

display.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# 26. SAVE LATEX
# ============================================================

latex_table = display.to_latex(
    index=False,
    escape=True,
    caption=(
        "Out-of-sample volatility spike classification "
        "performance of the ARIMA-GARCH, BVAR and "
        "multivariate 2D-CNN models on common target dates."
    ),
    label="tab:spike_classification_comparison",
    float_format="%.4f"
)


OUTPUT_TEX.write_text(
    latex_table,
    encoding="utf-8"
)


# ============================================================
# 27. SAVE COMMON-SAMPLE SPIKE DATA
# ============================================================

COMMON_OUTPUT = (
    RESULTS_DIR
    / "table_4_14_common_sample_spike_predictions.csv"
)


comparison[
    [
        "target_date",
        "target_volatility",
        "actual_spike",
        "ARIMA_score",
        "ARIMA_prediction",
        "BVAR_probability",
        "BVAR_prediction",
        "CNN_probability",
        "CNN_prediction"
    ]
].to_csv(
    COMMON_OUTPUT,
    index=False
)


# ============================================================
# 28. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 100)
print("TABLE 4.14")
print("SPIKE CLASSIFICATION PERFORMANCE COMPARISON")
print("=" * 100)


print(
    f"\nCommon observations : "
    f"{len(comparison):,}"
)


print(
    f"Start date          : "
    f"{comparison['target_date'].min().date()}"
)


print(
    f"End date            : "
    f"{comparison['target_date'].max().date()}"
)


print(
    f"\nActual spikes       : "
    f"{int(actual.sum()):,}"
)


print(
    f"Actual spike rate   : "
    f"{actual.mean():.4%}"
)


print("\n")
print("-" * 100)
print("MODEL RESULTS")
print("-" * 100)


print(
    display[
        [
            "Model",
            "Observations",
            "Actual Spikes",
            "Predicted Spikes",
            "Accuracy",
            "Balanced Accuracy",
            "Precision",
            "Recall",
            "Specificity",
            "F1",
            "ROC-AUC",
            "PR-AUC",
            "TN",
            "FP",
            "FN",
            "TP",
            "Overall Rank"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 29. IDENTIFY BEST MODEL
# ============================================================

best_model = (
    results
    .sort_values(
        "Overall Rank"
    )
    .iloc[0]
)


print("\n")
print("-" * 100)
print("BEST OVERALL SPIKE CLASSIFIER")
print("-" * 100)


print(
    f"\nModel       : "
    f"{best_model['Model']}"
)


print(
    f"F1          : "
    f"{best_model['F1']:.6f}"
)


print(
    f"ROC-AUC     : "
    f"{best_model['ROC-AUC']:.6f}"
)


print(
    f"PR-AUC      : "
    f"{best_model['PR-AUC']:.6f}"
)


print(
    f"Recall      : "
    f"{best_model['Recall']:.6f}"
)


print(
    f"Precision   : "
    f"{best_model['Precision']:.6f}"
)


# ============================================================
# 30. FILES
# ============================================================

print("\n")
print("=" * 100)
print("FILES SAVED")
print("=" * 100)


print(
    f"\nTable CSV:\n"
    f"{OUTPUT_CSV}"
)


print(
    f"\nTable LaTeX:\n"
    f"{OUTPUT_TEX}"
)


print(
    f"\nCommon-sample spike data:\n"
    f"{COMMON_OUTPUT}"
)


print("\n")
print("=" * 100)
print("TABLE 4.14 COMPLETE")
print("=" * 100)