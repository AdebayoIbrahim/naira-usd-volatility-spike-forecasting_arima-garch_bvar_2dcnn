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
# CHAPTER 4 - ARIMA-GARCH RESULTS
# [TABLE 4.6 HERE]
# ARIMA-GARCH Out-of-Sample Forecasting Performance
# ============================================================


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------
PROJECT_ROOT = Path.cwd()

FORECAST_PATH = (
    PROJECT_ROOT
    / "results"
    / "arima_garch_common_sample_forecasts.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Check file
# ------------------------------------------------------------
if not FORECAST_PATH.exists():
    raise FileNotFoundError(
        f"File not found:\n{FORECAST_PATH}"
    )


# ------------------------------------------------------------
# 3. Read file first WITHOUT parse_dates
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
    "origin_date",
    "target_date",
    "actual_volatility",
    "predicted_volatility",
    "actual_spike",
    "predicted_spike"
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
        f"Available columns: {df.columns.tolist()}"
    )


# ------------------------------------------------------------
# 5. Convert dates AFTER checking columns
# ------------------------------------------------------------
df["origin_date"] = pd.to_datetime(
    df["origin_date"],
    errors="coerce"
)

df["target_date"] = pd.to_datetime(
    df["target_date"],
    errors="coerce"
)


# ------------------------------------------------------------
# 6. Sort chronologically
# ------------------------------------------------------------
df = (
    df.sort_values("target_date")
      .reset_index(drop=True)
)


# ------------------------------------------------------------
# 7. Prepare final analysis data
# ------------------------------------------------------------
analysis = df[
    required_columns
].copy()

analysis = analysis.replace(
    [np.inf, -np.inf],
    np.nan
)

analysis = analysis.dropna(
    subset=[
        "actual_volatility",
        "predicted_volatility",
        "actual_spike",
        "predicted_spike"
    ]
)


# ------------------------------------------------------------
# 8. Extract values
# ------------------------------------------------------------
actual_volatility = (
    analysis["actual_volatility"]
    .astype(float)
)

predicted_volatility = (
    analysis["predicted_volatility"]
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


# ------------------------------------------------------------
# 9. Continuous forecasting metrics
# ------------------------------------------------------------
error = (
    actual_volatility
    - predicted_volatility
)

mae = np.mean(
    np.abs(error)
)

mse = np.mean(
    error ** 2
)

rmse = np.sqrt(mse)


# ------------------------------------------------------------
# 10. Classification metrics
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
# 11. Confusion matrix
# ------------------------------------------------------------
tn, fp, fn, tp = confusion_matrix(
    actual_spike,
    predicted_spike,
    labels=[0, 1]
).ravel()


# ------------------------------------------------------------
# 12. Specificity and balanced accuracy
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
# 13. ROC-AUC
# ------------------------------------------------------------
roc_auc = roc_auc_score(
    actual_spike,
    predicted_volatility
)


# ------------------------------------------------------------
# 14. Build Table 4.6
# ------------------------------------------------------------
table_4_6 = pd.DataFrame(
    {
        "Metric": [
            "Number of Forecasts",
            "Actual Spikes",
            "Predicted Spikes",
            "MAE",
            "MSE",
            "RMSE",
            "Accuracy",
            "Precision",
            "Recall",
            "Specificity",
            "Balanced Accuracy",
            "F1-score",
            "ROC-AUC",
            "True Negatives",
            "False Positives",
            "False Negatives",
            "True Positives"
        ],
        "Value": [
            len(analysis),
            int(actual_spike.sum()),
            int(predicted_spike.sum()),
            mae,
            mse,
            rmse,
            accuracy,
            precision,
            recall,
            specificity,
            balanced_accuracy,
            f1,
            roc_auc,
            int(tn),
            int(fp),
            int(fn),
            int(tp)
        ]
    }
)


# ------------------------------------------------------------
# 15. Round values
# ------------------------------------------------------------
decimal_metrics = [
    "MAE",
    "MSE",
    "RMSE",
    "Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "Balanced Accuracy",
    "F1-score",
    "ROC-AUC"
]

for metric in decimal_metrics:

    mask = (
        table_4_6["Metric"] == metric
    )

    table_4_6.loc[
        mask,
        "Value"
    ] = (
        table_4_6.loc[
            mask,
            "Value"
        ]
        .astype(float)
        .round(6)
    )


# ------------------------------------------------------------
# 16. Save CSV
# ------------------------------------------------------------
csv_path = (
    RESULTS_DIR
    / "table_4_6_arima_garch_performance.csv"
)

table_4_6.to_csv(
    csv_path,
    index=False
)


# ------------------------------------------------------------
# 17. Save LaTeX
# ------------------------------------------------------------
latex_path = (
    RESULTS_DIR
    / "table_4_6_arima_garch_performance.tex"
)

latex_table = table_4_6.to_latex(
    index=False,
    escape=False,
    caption="ARIMA-GARCH Out-of-Sample Forecasting Performance",
    label="tab:4.6"
)

latex_path.write_text(
    latex_table,
    encoding="utf-8"
)


# ------------------------------------------------------------
# 18. Print results
# ------------------------------------------------------------
print("\n" + "=" * 75)
print("[TABLE 4.6 HERE]")
print("=" * 75)

print("\nForecast period:")
print(
    f"{analysis['target_date'].min().date()} "
    f"to "
    f"{analysis['target_date'].max().date()}"
)

print(
    f"\nNumber of forecasts: "
    f"{len(analysis):,}"
)

print("\nPerformance:")
print(
    table_4_6.to_string(index=False)
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