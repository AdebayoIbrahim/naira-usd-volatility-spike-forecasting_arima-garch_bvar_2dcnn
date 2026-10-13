#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 23:41:21 2026

@author: setup
"""

"""
============================================================
FIGURE 4.16
ROC CURVES FOR THE THREE FORECASTING MODELS
============================================================

Models:
    1. ARIMA(0,0,1)-GARCH(2,1)-t
    2. BVAR(8)-log1p
    3. Multivariate 2D-CNN-A

All models are evaluated on the common target dates shared
by the three forecasting outputs.

ROC-AUC measures the ability of each model's continuous
score to rank actual volatility spikes above non-spikes.

Outputs:
    figures/figure_4_16_model_comparison_roc.png
    results/figure_4_16_roc_auc_values.csv
============================================================
"""


# ============================================================
# 1. IMPORTS
# ============================================================

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score
)


# ============================================================
# 2. PATHS
# ============================================================

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

FIGURES_DIR.mkdir(
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
    Path("data_cleaned")
    / "forecasting_dataset_2006_2025_spike_gap_target_safe.csv"
)


OUTPUT_FIGURE = (
    FIGURES_DIR
    / "figure_4_16_model_comparison_roc.png"
)

OUTPUT_VALUES = (
    RESULTS_DIR
    / "figure_4_16_roc_auc_values.csv"
)


SPIKE_THRESHOLD = 0.10927724


# ============================================================
# 3. CHECK FILES
# ============================================================

required_files = {
    "ARIMA-GARCH":
        ARIMA_FILE,

    "BVAR":
        BVAR_FILE,

    "CNN":
        CNN_FILE,

    "Canonical target":
        TARGET_FILE
}


for name, file_path in required_files.items():

    if not file_path.exists():

        raise FileNotFoundError(
            f"\n{name} file not found:\n"
            f"{file_path.resolve()}"
        )


# ============================================================
# 4. LOAD CANONICAL TARGET
# ============================================================

target = pd.read_csv(
    TARGET_FILE
)


target = target[
    [
        "target_date",
        "target_volatility"
    ]
].copy()


# ============================================================
# 5. LOAD ARIMA-GARCH
# ============================================================

arima = pd.read_csv(
    ARIMA_FILE
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
# 6. LOAD BVAR
# ============================================================

bvar = pd.read_csv(
    BVAR_FILE
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
# 7. LOAD CNN
# ============================================================

cnn = pd.read_csv(
    CNN_FILE
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
# 8. CONVERT DATES
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


# ============================================================
# 9. CONVERT NUMERIC VALUES
# ============================================================

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
# 10. REMOVE INVALID VALUES
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
    .drop_duplicates(
        "target_date"
    )
)

arima = (
    arima
    .drop_duplicates(
        "target_date"
    )
)

bvar = (
    bvar
    .drop_duplicates(
        "target_date"
    )
)

cnn = (
    cnn
    .drop_duplicates(
        "target_date"
    )
)


# ============================================================
# 12. MERGE ON COMMON TARGET DATES
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
# 13. DEFINE COMMON ACTUAL SPIKE
# ============================================================

comparison["actual_spike"] = (

    comparison[
        "target_volatility"
    ]

    > SPIKE_THRESHOLD

).astype(int)


# ============================================================
# 14. CHECK COMMON SAMPLE
# ============================================================

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
# 15. ROC CALCULATIONS
# ============================================================

model_scores = {

    "ARIMA-GARCH":
        comparison["ARIMA_score"],

    "BVAR":
        comparison["BVAR_score"],

    "2D-CNN":
        comparison["CNN_score"]

}


roc_results = []


for model_name, scores in model_scores.items():

    fpr, tpr, thresholds = roc_curve(
        comparison["actual_spike"],
        scores
    )

    auc_value = roc_auc_score(
        comparison["actual_spike"],
        scores
    )

    roc_results.append({

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

        "ROC_AUC":
            auc_value

    })


# ============================================================
# 16. SAVE AUC VALUES
# ============================================================

roc_values = pd.DataFrame(
    roc_results
)


roc_values = roc_values.sort_values(
    "ROC_AUC",
    ascending=False
)


roc_values.to_csv(
    OUTPUT_VALUES,
    index=False
)


# ============================================================
# 17. CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 8)
)


for model_name, scores in model_scores.items():

    fpr, tpr, _ = roc_curve(
        comparison["actual_spike"],
        scores
    )

    auc_value = roc_auc_score(
        comparison["actual_spike"],
        scores
    )

    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label=(
            f"{model_name} "
            f"(AUC = {auc_value:.4f})"
        )
    )


# Random-classifier reference

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
    label="Random classifier"
)


# ============================================================
# 18. LABELS
# ============================================================

ax.set_xlabel(
    "False Positive Rate",
    fontsize=11
)

ax.set_ylabel(
    "True Positive Rate",
    fontsize=11
)

ax.set_title(
    "ROC Curves for Volatility Spike Detection",
    fontsize=13,
    pad=12
)


ax.set_xlim(
    0,
    1
)

ax.set_ylim(
    0,
    1.02
)


ax.grid(
    True,
    linestyle="--",
    linewidth=0.5,
    alpha=0.35
)


ax.legend(
    loc="lower right",
    frameon=True
)


plt.tight_layout()


# ============================================================
# 19. SAVE FIGURE
# ============================================================

fig.savefig(
    OUTPUT_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.close(fig)


# ============================================================
# 20. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 90)
print("FIGURE 4.16")
print("ROC COMPARISON")
print("=" * 90)


print(
    f"\nCommon observations : "
    f"{len(comparison):,}"
)


print(
    f"Actual spikes       : "
    f"{int(comparison['actual_spike'].sum()):,}"
)


print(
    f"Spike rate          : "
    f"{comparison['actual_spike'].mean():.4%}"
)


print("\nROC-AUC values:")
print(
    roc_values.to_string(
        index=False
    )
)


print("\nFiles saved:")

print(
    f"Figure:\n"
    f"{OUTPUT_FIGURE}"
)

print(
    f"\nAUC values:\n"
    f"{OUTPUT_VALUES}"
)


print("\n")
print("=" * 90)
print("FIGURE 4.16 COMPLETE")
print("=" * 90)