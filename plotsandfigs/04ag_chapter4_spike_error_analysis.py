#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 23:54:04 2026

@author: setup
"""

"""
============================================================
TABLE 4.15
SPIKE DETECTION AND ERROR ANALYSIS
============================================================

Models:
    1. ARIMA(0,0,1)-GARCH(2,1)-t
    2. BVAR(8)-log1p
    3. Multivariate 2D-CNN-A

All models use the COMMON target dates.

COMMON SPIKE DEFINITION:
    target_volatility > 0.10927724

MODEL DECISION RULES:
    ARIMA-GARCH:
        forecast volatility > 0.10927724

    BVAR:
        spike probability >= 0.50

    CNN:
        spike probability >= 0.328393

OUTPUTS:
    results/table_4_15_spike_error_analysis.csv
    results/table_4_15_spike_error_analysis.tex
    results/table_4_15_common_spike_error_predictions.csv
============================================================
"""


# ============================================================
# 1. IMPORTS
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 2. PATHS
# ============================================================

RESULTS_DIR = Path("results")
DATA_DIR = Path("data_cleaned")

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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
    / "table_4_15_spike_error_analysis.csv"
)

OUTPUT_TEX = (
    RESULTS_DIR
    / "table_4_15_spike_error_analysis.tex"
)

COMMON_OUTPUT = (
    RESULTS_DIR
    / "table_4_15_common_spike_error_predictions.csv"
)


# ============================================================
# 3. FROZEN THRESHOLDS
# ============================================================

SPIKE_THRESHOLD = 0.10927724

BVAR_CUTOFF = 0.50

CNN_CUTOFF = 0.328393


# ============================================================
# 4. CHECK INPUT FILES
# ============================================================

input_files = {

    "ARIMA-GARCH":
        ARIMA_FILE,

    "BVAR":
        BVAR_FILE,

    "CNN":
        CNN_FILE,

    "Canonical target":
        TARGET_FILE
}


for name, file_path in input_files.items():

    if not file_path.exists():

        raise FileNotFoundError(
            f"\n{name} file not found:\n"
            f"{file_path.resolve()}"
        )


# ============================================================
# 5. LOAD CANONICAL TARGET
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
    "predicted_spike_probability"
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
        "predicted_spike_probability"
    ]
].copy()


bvar = bvar.rename(
    columns={
        "predicted_spike_probability":
            "BVAR_score"
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
    "predicted_spike_probability"
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
        "predicted_spike_probability"
    ]
].copy()


cnn = cnn.rename(
    columns={
        "predicted_spike_probability":
            "CNN_score"
    }
)


# ============================================================
# 9. CONVERT DATA TYPES
# ============================================================

for df in [
    target,
    arima,
    bvar,
    cnn
]:

    df["target_date"] = pd.to_datetime(
        df["target_date"],
        errors="coerce"
    )


target["target_volatility"] = pd.to_numeric(
    target["target_volatility"],
    errors="coerce"
)

arima["ARIMA_score"] = pd.to_numeric(
    arima["ARIMA_score"],
    errors="coerce"
)

bvar["BVAR_score"] = pd.to_numeric(
    bvar["BVAR_score"],
    errors="coerce"
)

cnn["CNN_score"] = pd.to_numeric(
    cnn["CNN_score"],
    errors="coerce"
)


# ============================================================
# 10. REMOVE INVALID OBSERVATIONS
# ============================================================

target = target.dropna()

arima = arima.dropna()

bvar = bvar.dropna()

cnn = cnn.dropna()


# ============================================================
# 11. REMOVE DUPLICATE DATES
# ============================================================

target = (
    target
    .drop_duplicates("target_date")
)

arima = (
    arima
    .drop_duplicates("target_date")
)

bvar = (
    bvar
    .drop_duplicates("target_date")
)

cnn = (
    cnn
    .drop_duplicates("target_date")
)


# ============================================================
# 12. BUILD COMMON SAMPLE
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


if comparison.empty:

    raise ValueError(
        "\nNo common observations were found."
    )


comparison = (
    comparison
    .sort_values("target_date")
    .reset_index(drop=True)
)


# ============================================================
# 13. DEFINE ACTUAL SPIKES
# ============================================================

comparison["actual_spike"] = (
    comparison["target_volatility"]
    > SPIKE_THRESHOLD
).astype(int)


# ============================================================
# 14. DEFINE MODEL PREDICTIONS
# ============================================================

comparison["ARIMA_prediction"] = (
    comparison["ARIMA_score"]
    > SPIKE_THRESHOLD
).astype(int)


comparison["BVAR_prediction"] = (
    comparison["BVAR_score"]
    >= BVAR_CUTOFF
).astype(int)


comparison["CNN_prediction"] = (
    comparison["CNN_score"]
    >= CNN_CUTOFF
).astype(int)


# ============================================================
# 15. ERROR ANALYSIS FUNCTION
# ============================================================

def calculate_error_metrics(
    actual,
    predicted
):

    actual = np.asarray(
        actual,
        dtype=int
    )

    predicted = np.asarray(
        predicted,
        dtype=int
    )


    # --------------------------------------------------------
    # Confusion matrix components
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
    # Rates
    # --------------------------------------------------------

    total_spikes = (
        tp + fn
    )

    total_nonspikes = (
        tn + fp
    )


    recall = (
        tp / total_spikes
        if total_spikes > 0
        else np.nan
    )


    false_negative_rate = (
        fn / total_spikes
        if total_spikes > 0
        else np.nan
    )


    false_positive_rate = (
        fp / total_nonspikes
        if total_nonspikes > 0
        else np.nan
    )


    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else np.nan
    )


    return {

        "TN":
            tn,

        "FP":
            fp,

        "FN":
            fn,

        "TP":
            tp,

        "Precision":
            precision,

        "Recall":
            recall,

        "False Positive Rate":
            false_positive_rate,

        "False Negative Rate":
            false_negative_rate

    }


# ============================================================
# 16. CALCULATE ERROR METRICS
# ============================================================

model_definitions = [

    (
        "ARIMA(0,0,1)-GARCH(2,1)-t",
        "ARIMA_prediction"
    ),

    (
        "BVAR(8)-log1p",
        "BVAR_prediction"
    ),

    (
        "Multivariate 2D-CNN-A",
        "CNN_prediction"
    )

]


results = []


for model_name, prediction_column in model_definitions:

    metrics = calculate_error_metrics(

        comparison[
            "actual_spike"
        ],

        comparison[
            prediction_column
        ]

    )


    results.append({

        "Model":
            model_name,

        "Observations":
            len(comparison),

        "Actual Spikes":
            int(
                comparison[
                    "actual_spike"
                ].sum()
            ),

        "Predicted Spikes":
            int(
                comparison[
                    prediction_column
                ].sum()
            ),

        "TN":
            metrics["TN"],

        "FP":
            metrics["FP"],

        "FN":
            metrics["FN"],

        "TP":
            metrics["TP"],

        "Precision":
            metrics["Precision"],

        "Recall":
            metrics["Recall"],

        "False Positive Rate":
            metrics[
                "False Positive Rate"
            ],

        "False Negative Rate":
            metrics[
                "False Negative Rate"
            ]

    })


results = pd.DataFrame(
    results
)


# ============================================================
# 17. ROUND DISPLAY METRICS
# ============================================================

percentage_columns = [

    "Precision",
    "Recall",
    "False Positive Rate",
    "False Negative Rate"

]


display = results.copy()


for col in percentage_columns:

    display[col] = (
        display[col]
        .round(4)
    )


# ============================================================
# 18. SAVE TABLE
# ============================================================

display.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# 19. SAVE LATEX
# ============================================================

latex_table = display.to_latex(
    index=False,
    escape=True,
    caption=(
        "Spike detection errors and classification outcomes "
        "for the three forecasting models on common "
        "out-of-sample target dates."
    ),
    label="tab:spike_error_analysis",
    float_format="%.4f"
)


OUTPUT_TEX.write_text(
    latex_table,
    encoding="utf-8"
)


# ============================================================
# 20. SAVE OBSERVATION-LEVEL ERROR DATA
# ============================================================

comparison[
    [
        "target_date",
        "target_volatility",
        "actual_spike",
        "ARIMA_score",
        "ARIMA_prediction",
        "BVAR_score",
        "BVAR_prediction",
        "CNN_score",
        "CNN_prediction"
    ]
].to_csv(
    COMMON_OUTPUT,
    index=False
)


# ============================================================
# 21. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 100)
print("TABLE 4.15")
print("SPIKE DETECTION AND ERROR ANALYSIS")
print("=" * 100)


print(
    f"\nCommon observations : "
    f"{len(comparison):,}"
)


print(
    f"Actual spikes       : "
    f"{int(comparison['actual_spike'].sum()):,}"
)


print(
    f"Actual spike rate   : "
    f"{comparison['actual_spike'].mean():.4%}"
)


print("\n")
print("-" * 100)
print("MODEL ERROR RESULTS")
print("-" * 100)


print(
    display.to_string(
        index=False
    )
)


# ============================================================
# 22. BEST AT MINIMISING MISSED SPIKES
# ============================================================

best_recall = (
    results
    .sort_values(
        "Recall",
        ascending=False
    )
    .iloc[0]
)


print("\n")
print("-" * 100)
print("BEST MODEL FOR MINIMISING MISSED SPIKES")
print("-" * 100)


print(
    f"\nModel  : "
    f"{best_recall['Model']}"
)


print(
    f"Recall : "
    f"{best_recall['Recall']:.6f}"
)


print(
    f"FN     : "
    f"{int(best_recall['FN'])}"
)


# ============================================================
# 23. BEST AT MINIMISING FALSE ALARMS
# ============================================================

best_specific = (
    results
    .sort_values(
        "False Positive Rate",
        ascending=True
    )
    .iloc[0]
)


print("\n")
print("-" * 100)
print("BEST MODEL FOR MINIMISING FALSE ALARMS")
print("-" * 100)


print(
    f"\nModel : "
    f"{best_specific['Model']}"
)


print(
    f"FPR   : "
    f"{best_specific['False Positive Rate']:.6f}"
)


print(
    f"FP    : "
    f"{int(best_specific['FP'])}"
)


# ============================================================
# 24. FILES
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
    f"\nObservation-level data:\n"
    f"{COMMON_OUTPUT}"
)


print("\n")
print("=" * 100)
print("TABLE 4.15 COMPLETE")
print("=" * 100)