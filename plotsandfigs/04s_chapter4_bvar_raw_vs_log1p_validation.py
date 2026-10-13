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
# [TABLE 4.9 HERE]
# Pooled BVAR Raw vs Log1p Validation Performance
# ============================================================


# ------------------------------------------------------------
# 1. Project paths
# ------------------------------------------------------------
PROJECT_ROOT = Path.cwd()

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "bvar8_raw_vs_log1p_multiblock_validation_forecasts.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Check input file
# ------------------------------------------------------------
if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Validation forecast file not found:\n{INPUT_PATH}"
    )


# ------------------------------------------------------------
# 3. Load file
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
    "model_type",
    "actual_volatility",
    "actual_spike",
    "forecast_volatility",
    "predicted_spike_probability",
    "predicted_spike"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "\nMissing required columns:\n"
        f"{missing_columns}\n\n"
        f"Available columns:\n{df.columns.tolist()}"
    )


# ------------------------------------------------------------
# 5. Standardise model labels
# ------------------------------------------------------------
df["model_type"] = (
    df["model_type"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df["model_type"] = df["model_type"].replace(
    {
        "raw": "Raw",
        "log1p": "Log1p",
        "log": "Log1p"
    }
)


# ------------------------------------------------------------
# 6. Keep only Raw and Log1p
# ------------------------------------------------------------
df = df[
    df["model_type"].isin(["Raw", "Log1p"])
].copy()


if df.empty:
    raise ValueError(
        "No Raw or Log1p observations were found in "
        "'model_type'.\n\n"
        f"Available model types:\n"
        f"{df['model_type'].unique()}"
    )


# ------------------------------------------------------------
# 7. Convert numeric columns
# ------------------------------------------------------------
numeric_columns = [
    "actual_volatility",
    "actual_spike",
    "forecast_volatility",
    "predicted_spike_probability",
    "predicted_spike"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ------------------------------------------------------------
# 8. Calculate pooled metrics
# ------------------------------------------------------------
results = []


for model_type in ["Raw", "Log1p"]:

    subset = df[
        df["model_type"] == model_type
    ].copy()

    subset = subset.replace(
        [np.inf, -np.inf],
        np.nan
    )

    subset = subset.dropna(
        subset=[
            "actual_volatility",
            "actual_spike",
            "forecast_volatility",
            "predicted_spike"
        ]
    )

    if subset.empty:
        continue


    # --------------------------------------------------------
    # Extract values
    # --------------------------------------------------------
    actual_volatility = (
        subset["actual_volatility"]
        .astype(float)
    )

    forecast_volatility = (
        subset["forecast_volatility"]
        .astype(float)
    )

    actual_spike = (
        subset["actual_spike"]
        .astype(int)
    )

    predicted_spike = (
        subset["predicted_spike"]
        .astype(int)
    )


    # --------------------------------------------------------
    # Continuous metrics
    # --------------------------------------------------------
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


    # --------------------------------------------------------
    # Classification metrics
    # --------------------------------------------------------
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


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------
    tn, fp, fn, tp = confusion_matrix(
        actual_spike,
        predicted_spike,
        labels=[0, 1]
    ).ravel()


    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    balanced_accuracy = (
        recall + specificity
    ) / 2


    # --------------------------------------------------------
    # ROC-AUC using predicted spike probability
    # --------------------------------------------------------
    probability_subset = subset.dropna(
        subset=["predicted_spike_probability"]
    )

    if (
        not probability_subset.empty
        and probability_subset["actual_spike"].nunique() == 2
    ):

        roc_auc_probability = roc_auc_score(
            probability_subset["actual_spike"].astype(int),
            probability_subset["predicted_spike_probability"].astype(float)
        )

    else:

        roc_auc_probability = np.nan


    # --------------------------------------------------------
    # ROC-AUC using continuous volatility forecast
    # --------------------------------------------------------
    try:

        roc_auc_volatility = roc_auc_score(
            actual_spike,
            forecast_volatility
        )

    except ValueError:

        roc_auc_volatility = np.nan


    # --------------------------------------------------------
    # Brier score
    # --------------------------------------------------------
    if not probability_subset.empty:

        brier_score = np.mean(
            (
                probability_subset[
                    "predicted_spike_probability"
                ].astype(float)
                -
                probability_subset[
                    "actual_spike"
                ].astype(int)
            ) ** 2
        )

    else:

        brier_score = np.nan


    # --------------------------------------------------------
    # Spike counts
    # --------------------------------------------------------
    actual_spikes = int(
        actual_spike.sum()
    )

    predicted_spikes = int(
        predicted_spike.sum()
    )


    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------
    results.append(
        {
            "Transformation": model_type,
            "Observations": len(subset),
            "Actual Spikes": actual_spikes,
            "Predicted Spikes": predicted_spikes,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "Specificity": specificity,
            "Balanced Accuracy": balanced_accuracy,
            "F1-score": f1,
            "ROC-AUC (Probability)": roc_auc_probability,
            "ROC-AUC (Volatility)": roc_auc_volatility,
            "Brier Score": brier_score,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp
        }
    )


# ------------------------------------------------------------
# 9. Create final table
# ------------------------------------------------------------
table = pd.DataFrame(results)

if table.empty:
    raise ValueError(
        "No validation results were generated."
    )


# ------------------------------------------------------------
# 10. Round values
# ------------------------------------------------------------
decimal_columns = [
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
    "Brier Score"
]

for col in decimal_columns:
    table[col] = table[col].round(6)


# ------------------------------------------------------------
# 11. Save CSV
# ------------------------------------------------------------
csv_path = (
    RESULTS_DIR
    / "table_4_9_bvar_raw_vs_log1p_validation.csv"
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
    / "table_4_9_bvar_raw_vs_log1p_validation.tex"
)

latex_table = table.to_latex(
    index=False,
    escape=False,
    caption=(
        "Pooled BVAR Raw versus Log1p "
        "Validation Performance"
    ),
    label="tab:4.9"
)

latex_path.write_text(
    latex_table,
    encoding="utf-8"
)


# ------------------------------------------------------------
# 13. Print results
# ------------------------------------------------------------
print("\n" + "=" * 100)
print("[TABLE 4.9 HERE]")
print("=" * 100)

print(
    "Pooled BVAR Raw versus Log1p "
    "Validation Performance"
)

print("\n")
print(
    table.to_string(index=False)
)

print("\nCSV saved to:")
print(csv_path)

print("\nLaTeX saved to:")
print(latex_path)

print("\nCompleted successfully.")