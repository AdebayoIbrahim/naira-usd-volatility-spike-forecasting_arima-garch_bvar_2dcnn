#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 23:51:48 2026

@author: setup
"""

"""
============================================================
FIGURE 4.17
PRECISION-RECALL CURVES FOR THE THREE MODELS
============================================================

Models:
    1. ARIMA(0,0,1)-GARCH(2,1)-t
    2. BVAR(8)-log1p
    3. Multivariate 2D-CNN-A

All models are evaluated on the COMMON target dates.

Actual spike definition:
    target_volatility > 0.10927724

Scores:
    ARIMA-GARCH -> predicted volatility
    BVAR        -> predicted spike probability
    CNN         -> predicted spike probability

Outputs:
    figures/figure_4_17_model_comparison_precision_recall.png
    results/figure_4_17_pr_auc_values.csv
============================================================
"""


# ============================================================
# 1. IMPORTS
# ============================================================

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score
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
    / "figure_4_17_model_comparison_precision_recall.png"
)

OUTPUT_VALUES = (
    RESULTS_DIR
    / "figure_4_17_pr_auc_values.csv"
)


# ============================================================
# 3. FROZEN SPIKE THRESHOLD
# ============================================================

SPIKE_THRESHOLD = 0.10927724


# ============================================================
# 4. CHECK FILES
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
# 5. LOAD CANONICAL TARGET
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
# 6. LOAD ARIMA-GARCH SCORE
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
# 7. LOAD BVAR SCORE
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
# 8. LOAD CNN SCORE
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
# 9. CONVERT DATES
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
# 10. CONVERT NUMERIC VALUES
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
# 11. REMOVE INVALID OBSERVATIONS
# ============================================================

target = target.dropna()

arima = arima.dropna()

bvar = bvar.dropna()

cnn = cnn.dropna()


# ============================================================
# 12. REMOVE DUPLICATE TARGET DATES
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
# 13. MERGE ON COMMON TARGET DATES
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
# 14. DEFINE ACTUAL SPIKES
# ============================================================

comparison["actual_spike"] = (

    comparison[
        "target_volatility"
    ]

    > SPIKE_THRESHOLD

).astype(int)


# ============================================================
# 15. SORT COMMON SAMPLE
# ============================================================

comparison = (
    comparison
    .sort_values("target_date")
    .reset_index(drop=True)
)


if comparison.empty:

    raise ValueError(
        "\nNo common observations were found."
    )


# ============================================================
# 16. MODEL SCORE DEFINITIONS
# ============================================================

model_scores = {

    "ARIMA-GARCH":
        comparison["ARIMA_score"],

    "BVAR":
        comparison["BVAR_score"],

    "2D-CNN":
        comparison["CNN_score"]

}


# ============================================================
# 17. CALCULATE PR CURVES AND PR-AUC
# ============================================================

pr_results = []


for model_name, scores in model_scores.items():

    precision, recall, thresholds = (
        precision_recall_curve(
            comparison["actual_spike"],
            scores
        )
    )

    pr_auc = average_precision_score(
        comparison["actual_spike"],
        scores
    )

    pr_results.append({

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

        "Spike Prevalence":
            comparison[
                "actual_spike"
            ].mean(),

        "PR_AUC":
            pr_auc

    })


pr_values = pd.DataFrame(
    pr_results
)


pr_values = pr_values.sort_values(
    "PR_AUC",
    ascending=False
)


# ============================================================
# 18. SAVE PR-AUC VALUES
# ============================================================

pr_values.to_csv(
    OUTPUT_VALUES,
    index=False
)


# ============================================================
# 19. CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 8)
)


for model_name, scores in model_scores.items():

    precision, recall, _ = (
        precision_recall_curve(
            comparison["actual_spike"],
            scores
        )
    )

    pr_auc = average_precision_score(
        comparison["actual_spike"],
        scores
    )

    ax.plot(
        recall,
        precision,
        linewidth=2,
        label=(
            f"{model_name} "
            f"(PR-AUC = {pr_auc:.4f})"
        )
    )


# ============================================================
# 20. BASELINE
# ============================================================

spike_prevalence = (
    comparison["actual_spike"]
    .mean()
)


ax.axhline(
    spike_prevalence,
    linestyle="--",
    linewidth=1,
    label=(
        f"Spike prevalence = "
        f"{spike_prevalence:.3f}"
    )
)


# ============================================================
# 21. AXES
# ============================================================

ax.set_xlabel(
    "Recall",
    fontsize=11
)

ax.set_ylabel(
    "Precision",
    fontsize=11
)

ax.set_title(
    "Precision-Recall Curves for Volatility Spike Detection",
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
    loc="lower left",
    frameon=True
)


plt.tight_layout()


# ============================================================
# 22. SAVE FIGURE
# ============================================================

fig.savefig(
    OUTPUT_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.close(fig)


# ============================================================
# 23. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 90)
print("FIGURE 4.17")
print("PRECISION-RECALL MODEL COMPARISON")
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
    f"Spike prevalence    : "
    f"{spike_prevalence:.4%}"
)


print("\nPR-AUC values:")
print(
    pr_values.to_string(
        index=False
    )
)


print("\nFiles saved:")

print(
    f"Figure:\n"
    f"{OUTPUT_FIGURE}"
)

print(
    f"\nPR-AUC values:\n"
    f"{OUTPUT_VALUES}"
)


print("\n")
print("=" * 90)
print("FIGURE 4.17 COMPLETE")
print("=" * 90)