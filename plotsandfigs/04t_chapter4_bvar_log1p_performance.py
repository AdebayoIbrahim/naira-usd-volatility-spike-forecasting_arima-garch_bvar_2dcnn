#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 14:15:42 2026

@author: setup
"""

from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# CHAPTER 4 - BVAR RESULTS
# [TABLE 4.10 HERE]
# BVAR Log1p Out-of-Sample Forecasting Performance
# ============================================================


# ------------------------------------------------------------
# 1. Project paths
# ------------------------------------------------------------
PROJECT_ROOT = Path.cwd()

FORECAST_PATH = (
    PROJECT_ROOT
    / "results"
    / "bvar8_final_log1p_common_sample_forecasts.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Check file
# ------------------------------------------------------------
if not FORECAST_PATH.exists():
    raise FileNotFoundError(
        f"Forecast file not found:\n{FORECAST_PATH}"
    )


# ------------------------------------------------------------
# 3. Load file
# ------------------------------------------------------------
df = pd.read_csv(FORECAST_PATH)

print("\nFile being used:")
print(FORECAST_PATH)

print("\nColumns found:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 4. Required columns
# ------------------------------------------------------------
required_columns = [
    "actual_volatility",
    "forecast_volatility",
    "actual_spike",
    "predicted_spike",
    "predicted_spike_probability"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "\nRequired columns are missing.\n"
        f"Missing: {missing_columns}\n\n"
        f"Available columns:\n{df.columns.tolist()}"
    )


# ------------------------------------------------------------
# 5. Prepare data
# ------------------------------------------------------------
analysis = df[
    required_columns
].copy()

analysis = analysis.replace(
    [np.inf, -np.inf],
    np.nan
)

analysis = analysis.dropna(
    subset=required_columns
)


# ------------------------------------------------------------
# 6. Extract values
# ------------------------------------------------------------
actual_volatility = (
    analysis["actual_volatility"]
    .astype(float)
)

forecast_volatility = (
    analysis["forecast_volatility"]
    .astype(float)
)

actual_spike = (
    analysis["actual_spike"]
    .astype(int)
)

predicted_spike = (
    analysis["predicted_spike"]
    .astype(int)
)

predicted_probability = (
    analysis["predicted_spike_probability"]
    .astype(float)
)


# ------------------------------------------------------------
# 7. Continuous volatility metrics
# ------------------------------------------------------------
errors = (
    actual_volatility
    - forecast_volatility
)

mae = np.mean(
    np.abs(errors)
)

mse = np.mean(
    errors ** 2
)

rmse = np.sqrt(mse)


# ------------------------------------------------------------
# 8. Classification metrics
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# 9. Confusion matrix
# ------------------------------------------------------------
tn, fp, fn, tp = confusion_matrix(
    actual_spike,
    predicted_spike,
    labels=[0, 1]
).ravel()


# ------------------------------------------------------------
# 10. Specificity and balanced accuracy
# ------------------------------------------------------------
specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else np.nan
)

balanced_accuracy = (
    recall + specificity
) / 2


# ------------------------------------------------------------
# 11. ROC-AUC using predicted spike probability
# ------------------------------------------------------------
roc_auc_probability = roc_auc_score(
    actual_spike,
    predicted_probability
)


# ------------------------------------------------------------
# 12. ROC-AUC using continuous volatility forecast
# ------------------------------------------------------------
roc_auc_volatility = roc_auc_score(
    actual_spike,
    forecast_volatility
)


# ------------------------------------------------------------
# 13. Mean predicted spike probability
# ------------------------------------------------------------
mean_probability = (
    predicted_probability.mean()
)


# ------------------------------------------------------------
# 14. Spike counts
# ------------------------------------------------------------
actual_spikes = int(
    actual_spike.sum()
)

predicted_spikes = int(
    predicted_spike.sum()
)


# ------------------------------------------------------------
# 15. Build Table 4.10
# ------------------------------------------------------------
table_4_10 = pd.DataFrame(
    {
        "Metric": [
            "Number of Forecasts",
            "Actual Spikes",
            "Predicted Spikes",
            "Mean Spike Probability",
            "MAE",
            "MSE",
            "RMSE",
            "Accuracy",
            "Precision",
            "Recall",
            "Specificity",
            "Balanced Accuracy",
            "F1-score",
            "ROC-AUC (Probability)",
            "ROC-AUC (Volatility)",
            "True Negatives",
            "False Positives",
            "False Negatives",
            "True Positives"
        ],
        "Value": [
            len(analysis),
            actual_spikes,
            predicted_spikes,
            mean_probability,
            mae,
            mse,
            rmse,
            accuracy,
            precision,
            recall,
            specificity,
            balanced_accuracy,
            f1,
            roc_auc_probability,
            roc_auc_volatility,
            tn,
            fp,
            fn,
            tp
        ]
    }
)


# ------------------------------------------------------------
# 16. Round values
# ------------------------------------------------------------
decimal_metrics = [
    "Mean Spike Probability",
    "MAE",
    "MSE",
    "RMSE",
    "Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "Balanced Accuracy",
    "F1-score",
    "ROC-AUC (Probability)",
    "ROC-AUC (Volatility)"
]

for metric in decimal_metrics:

    mask = (
        table_4_10["Metric"] == metric
    )

    table_4_10.loc[
        mask,
        "Value"
    ] = (
        table_4_10.loc[
            mask,
            "Value"
        ]
        .astype(float)
        .round(6)
    )


# ------------------------------------------------------------
# 17. Save CSV
# ------------------------------------------------------------
csv_path = (
    RESULTS_DIR
    / "table_4_10_bvar_log1p_performance.csv"
)

table_4_10.to_csv(
    csv_path,
    index=False
)


# ------------------------------------------------------------
# 18. Save LaTeX
# ------------------------------------------------------------
latex_path = (
    RESULTS_DIR
    / "table_4_10_bvar_log1p_performance.tex"
)

latex_table = table_4_10.to_latex(
    index=False,
    escape=False,
    caption=(
        "BVAR Log1p Out-of-Sample "
        "Forecasting Performance"
    ),
    label="tab:4.10"
)

latex_path.write_text(
    latex_table,
    encoding="utf-8"
)


# ------------------------------------------------------------
# 19. Print results
# ------------------------------------------------------------
print("\n" + "=" * 80)
print("[TABLE 4.10 HERE]")
print("=" * 80)

print(
    "BVAR Log1p Out-of-Sample "
    "Forecasting Performance"
)

print("\nNumber of forecasts:")
print(f"{len(analysis):,}")

print("\nPerformance:")
print(
    table_4_10.to_string(index=False)
)

print("\nConfusion Matrix:")
print(
    f"TN = {tn}, "
    f"FP = {fp}, "
    f"FN = {fn}, "
    f"TP = {tp}"
)

print("\nCSV saved to:")
print(csv_path)

print("\nLaTeX saved to:")
print(latex_path)

print("\nCompleted successfully.")