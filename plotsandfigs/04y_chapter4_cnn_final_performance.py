#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 22:59:05 2026

@author: setup
"""

"""
TABLE 4.12
Final Out-of-Sample Performance of the Selected Multivariate 2D-CNN

Source:
results/cnn_A_final_test_metrics.csv

Outputs:
results/table_4_12_cnn_final_performance.csv
results/table_4_12_cnn_final_performance.tex
"""

from pathlib import Path
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

RESULTS_DIR = Path("results")
SOURCE_FILE = RESULTS_DIR / "cnn_A_final_test_metrics.csv"

OUTPUT_CSV = RESULTS_DIR / "table_4_12_cnn_final_performance.csv"
OUTPUT_TEX = RESULTS_DIR / "table_4_12_cnn_final_performance.tex"


# ============================================================
# 2. LOAD FINAL CNN METRICS
# ============================================================

if not SOURCE_FILE.exists():
    raise FileNotFoundError(
        f"Source file not found: {SOURCE_FILE.resolve()}"
    )

df = pd.read_csv(SOURCE_FILE)

if df.empty:
    raise ValueError("The CNN metrics file is empty.")

row = df.iloc[0]


# ============================================================
# 3. BUILD THESIS TABLE
# ============================================================

table = pd.DataFrame([{
    "Model": row["model"],
    "Kernel": row["kernel"],
    "Window": int(row["window_size"]),
    "Test Period": f'{row["test_start"]} to {row["test_end"]}',

    # Continuous volatility forecasting
    "MAE": row["MAE"],
    "RMSE": row["RMSE"],

    # Spike classification
    "Accuracy": row["Accuracy"],
    "Precision": row["Precision"],
    "Recall": row["Recall"],
    "Specificity": row["Specificity"],
    "F1": row["F1"],
    "ROC-AUC": row["ROC_AUC"],
    "PR-AUC": row["PR_AUC"],
    "Brier": row["Brier_Score"],

    # Confusion matrix
    "TN": int(row["TN"]),
    "FP": int(row["FP"]),
    "FN": int(row["FN"]),
    "TP": int(row["TP"]),

    # Spike counts
    "Actual Spikes": int(row["actual_spikes"]),
    "Predicted Spikes": int(row["predicted_spikes"])
}])


# ============================================================
# 4. ROUND NUMERIC VALUES FOR DISPLAY
# ============================================================

display_table = table.copy()

for col in [
    "MAE", "RMSE", "Accuracy", "Precision", "Recall",
    "Specificity", "F1", "ROC-AUC", "PR-AUC", "Brier"
]:
    display_table[col] = display_table[col].astype(float).round(4)


# ============================================================
# 5. SAVE CSV
# ============================================================

display_table.to_csv(OUTPUT_CSV, index=False)


# ============================================================
# 6. SAVE LATEX
# ============================================================

latex_df = display_table.copy()

latex = latex_df.to_latex(
    index=False,
    escape=True,
    column_format="llrlrrrrrrrrrrrrrrrrr",
    caption=(
        "Final out-of-sample performance of the selected "
        "multivariate 2D-CNN model."
    ),
    label="tab:cnn_final_performance",
    float_format="%.4f"
)

# Clean escaped column labels where useful
latex = latex.replace("ROC-AUC", "ROC--AUC")
latex = latex.replace("PR-AUC", "PR--AUC")

OUTPUT_TEX.write_text(latex, encoding="utf-8")


# ============================================================
# 7. PRINT RESULTS
# ============================================================

print("\n" + "=" * 80)
print("TABLE 4.12 CREATED")
print("=" * 80)

print("\nSelected CNN:")
print(f"Model       : {row['model']}")
print(f"Kernel      : {row['kernel']}")
print(f"Window size : {int(row['window_size'])}")
print(f"Epochs      : {int(row['final_epochs'])}")
print(f"Test period : {row['test_start']} to {row['test_end']}")

print("\nContinuous volatility forecasting:")
print(f"MAE         : {row['MAE']:.6f}")
print(f"RMSE        : {row['RMSE']:.6f}")

print("\nSpike classification:")
print(f"Accuracy    : {row['Accuracy']:.6f}")
print(f"Precision   : {row['Precision']:.6f}")
print(f"Recall      : {row['Recall']:.6f}")
print(f"Specificity : {row['Specificity']:.6f}")
print(f"F1          : {row['F1']:.6f}")
print(f"ROC-AUC     : {row['ROC_AUC']:.6f}")
print(f"PR-AUC      : {row['PR_AUC']:.6f}")
print(f"Brier Score : {row['Brier_Score']:.6f}")

print("\nConfusion matrix:")
print(f"TN          : {int(row['TN'])}")
print(f"FP          : {int(row['FP'])}")
print(f"FN          : {int(row['FN'])}")
print(f"TP          : {int(row['TP'])}")

print("\nSpike counts:")
print(f"Actual      : {int(row['actual_spikes'])}")
print(f"Predicted   : {int(row['predicted_spikes'])}")

print("\nFiles saved:")
print(f"CSV         : {OUTPUT_CSV}")
print(f"LaTeX       : {OUTPUT_TEX}")