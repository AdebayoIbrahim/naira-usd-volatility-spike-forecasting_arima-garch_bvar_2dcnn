#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 22:47:07 2026

@author: setup
"""

from pathlib import Path

import pandas as pd


# ============================================================
# CHAPTER 4 - CNN RESULTS
# [TABLE 4.11 HERE]
# CNN Validation Architecture Comparison
# ============================================================


# ------------------------------------------------------------
# 1. Project paths
# ------------------------------------------------------------
PROJECT_ROOT = Path.cwd()

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "cnn_v5_architecture_comparison.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Check input file
# ------------------------------------------------------------
if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"CNN comparison file not found:\n{INPUT_PATH}"
    )


# ------------------------------------------------------------
# 3. Load comparison results
# ------------------------------------------------------------
df = pd.read_csv(INPUT_PATH)

print("\nFile being used:")
print(INPUT_PATH)

print("\nColumns found:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 4. Required columns
# ------------------------------------------------------------
required_columns = [
    "model",
    "validation_observations",
    "actual_spikes",
    "predicted_spikes",
    "probability_threshold",
    "MAE",
    "RMSE",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "Specificity",
    "ROC_AUC",
    "PR_AUC",
    "Brier",
    "TN",
    "FP",
    "FN",
    "TP",
    "kernel"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"\nMissing required columns:\n"
        f"{missing_columns}\n\n"
        f"Available columns:\n"
        f"{df.columns.tolist()}"
    )


# ------------------------------------------------------------
# 5. Select presentation columns
# ------------------------------------------------------------
table = df[
    [
        "model",
        "kernel",
        "validation_observations",
        "actual_spikes",
        "predicted_spikes",
        "probability_threshold",
        "MAE",
        "RMSE",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Specificity",
        "ROC_AUC",
        "PR_AUC",
        "Brier",
    ]
].copy()


# ------------------------------------------------------------
# 6. Clean model names
# ------------------------------------------------------------
table["Model"] = (
    table["model"]
    .replace({
        "CNN_A_kernel_2x3": "CNN-A",
        "CNN_B_kernel_1x3": "CNN-B"
    })
)

table["Kernel"] = table["kernel"].astype(str)


# ------------------------------------------------------------
# 7. Rename columns
# ------------------------------------------------------------
table = table[
    [
        "Model",
        "Kernel",
        "validation_observations",
        "actual_spikes",
        "predicted_spikes",
        "probability_threshold",
        "MAE",
        "RMSE",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Specificity",
        "ROC_AUC",
        "PR_AUC",
        "Brier",
    ]
]

table.columns = [
    "Model",
    "Kernel",
    "Validation Observations",
    "Actual Spikes",
    "Predicted Spikes",
    "Probability Cutoff",
    "MAE",
    "RMSE",
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "Specificity",
    "ROC-AUC",
    "PR-AUC",
    "Brier Score",
]


# ------------------------------------------------------------
# 8. Determine selected architecture
# ------------------------------------------------------------
# Primary selection criterion:
# highest validation PR-AUC.
#
# Tie-breakers:
# ROC-AUC, then F1-score.

table["_PR_AUC"] = pd.to_numeric(
    table["PR-AUC"],
    errors="coerce"
)

table["_ROC_AUC"] = pd.to_numeric(
    table["ROC-AUC"],
    errors="coerce"
)

table["_F1"] = pd.to_numeric(
    table["F1-score"],
    errors="coerce"
)

selection_order = (
    table
    .sort_values(
        by=["_PR_AUC", "_ROC_AUC", "_F1"],
        ascending=[False, False, False]
    )
)

selected_model = selection_order.iloc[0]["Model"]

table["Selected"] = (
    table["Model"] == selected_model
)


# ------------------------------------------------------------
# 9. Drop helper columns
# ------------------------------------------------------------
table = table.drop(
    columns=["_PR_AUC", "_ROC_AUC", "_F1"]
)


# ------------------------------------------------------------
# 10. Round values
# ------------------------------------------------------------
integer_columns = [
    "Validation Observations",
    "Actual Spikes",
    "Predicted Spikes",
]

for column in integer_columns:
    table[column] = (
        pd.to_numeric(
            table[column],
            errors="coerce"
        )
        .astype(int)
    )


decimal_columns = [
    "Probability Cutoff",
    "MAE",
    "RMSE",
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "Specificity",
    "ROC-AUC",
    "PR-AUC",
    "Brier Score",
]

for column in decimal_columns:
    table[column] = (
        pd.to_numeric(
            table[column],
            errors="coerce"
        )
        .round(6)
    )


# ------------------------------------------------------------
# 11. Save CSV
# ------------------------------------------------------------
csv_path = (
    RESULTS_DIR
    / "table_4_11_cnn_architecture_selection.csv"
)

table.to_csv(
    csv_path,
    index=False
)


# ------------------------------------------------------------
# 12. Save LaTeX
# ------------------------------------------------------------
latex_path = (
    RESULTS_DIR
    / "table_4_11_cnn_architecture_selection.tex"
)

latex_table = table.to_latex(
    index=False,
    escape=False,
    caption="CNN Validation Architecture Comparison",
    label="tab:4.11"
)

latex_path.write_text(
    latex_table,
    encoding="utf-8"
)


# ------------------------------------------------------------
# 13. Print results
# ------------------------------------------------------------
print("\n" + "=" * 100)
print("[TABLE 4.11 HERE]")
print("=" * 100)

print(
    "CNN Validation Architecture Comparison"
)

print("\n")
print(
    table.to_string(index=False)
)

print("\nSelected architecture:")
print(selected_model)

print("\nSelection criterion:")
print("Highest validation PR-AUC")
print("Tie-breakers: ROC-AUC, then F1-score")

print("\nCSV saved to:")
print(csv_path)

print("\nLaTeX saved to:")
print(latex_path)

print("\nCompleted successfully.")