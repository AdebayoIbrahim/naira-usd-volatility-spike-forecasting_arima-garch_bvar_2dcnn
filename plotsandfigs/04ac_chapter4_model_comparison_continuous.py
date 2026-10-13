"""
============================================================
TABLE 4.13
CONTINUOUS VOLATILITY FORECASTING COMPARISON
============================================================

Models compared:
    1. ARIMA(0,0,1)-GARCH(2,1)-t
    2. BVAR(8)-log1p
    3. Multivariate 2D-CNN-A

IMPORTANT:
------------
All models are evaluated against the SAME canonical
observed volatility series:

    target_volatility

from:

    data_cleaned/
    forecasting_dataset_2006_2025_spike_gap_target_safe.csv

The comparison is restricted to dates for which forecasts
from all three models are available.

Lower MAE, MSE and RMSE indicate better continuous
volatility forecasting performance.

OUTPUTS:
--------
results/table_4_13_continuous_model_comparison.csv
results/table_4_13_continuous_model_comparison.tex
============================================================
"""


# ============================================================
# 1. IMPORTS
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 2. DIRECTORIES
# ============================================================

RESULTS_DIR = Path("results")

DATA_DIR = Path("data_cleaned")

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. INPUT FILES
# ============================================================

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


# ============================================================
# 4. OUTPUT FILES
# ============================================================

OUTPUT_CSV = (
    RESULTS_DIR
    / "table_4_13_continuous_model_comparison.csv"
)

OUTPUT_TEX = (
    RESULTS_DIR
    / "table_4_13_continuous_model_comparison.tex"
)


# ============================================================
# 5. CHECK INPUT FILES
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
            f"\n{name} file was not found:\n"
            f"{file_path.resolve()}"
        )


# ============================================================
# 6. LOAD ARIMA-GARCH FORECASTS
# ============================================================

arima = pd.read_csv(
    ARIMA_FILE
)


required_arima_columns = [
    "target_date",
    "predicted_volatility"
]


missing_arima = [
    col
    for col in required_arima_columns
    if col not in arima.columns
]


if missing_arima:

    raise ValueError(
        "\nARIMA-GARCH file is missing:\n"
        f"{missing_arima}\n\n"
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
            "ARIMA_GARCH_forecast"
    }
)


# ============================================================
# 7. LOAD BVAR FORECASTS
# ============================================================

bvar = pd.read_csv(
    BVAR_FILE
)


required_bvar_columns = [
    "target_date",
    "forecast_volatility"
]


missing_bvar = [
    col
    for col in required_bvar_columns
    if col not in bvar.columns
]


if missing_bvar:

    raise ValueError(
        "\nBVAR file is missing:\n"
        f"{missing_bvar}\n\n"
        f"Available columns:\n"
        f"{list(bvar.columns)}"
    )


bvar = bvar[
    [
        "target_date",
        "forecast_volatility"
    ]
].copy()


bvar = bvar.rename(
    columns={
        "forecast_volatility":
            "BVAR_forecast"
    }
)


# ============================================================
# 8. LOAD CNN FORECASTS
# ============================================================

cnn = pd.read_csv(
    CNN_FILE
)


required_cnn_columns = [
    "target_date",
    "predicted_volatility"
]


missing_cnn = [
    col
    for col in required_cnn_columns
    if col not in cnn.columns
]


if missing_cnn:

    raise ValueError(
        "\nCNN file is missing:\n"
        f"{missing_cnn}\n\n"
        f"Available columns:\n"
        f"{list(cnn.columns)}"
    )


cnn = cnn[
    [
        "target_date",
        "predicted_volatility"
    ]
].copy()


cnn = cnn.rename(
    columns={
        "predicted_volatility":
            "CNN_forecast"
    }
)


# ============================================================
# 9. LOAD CANONICAL ACTUAL VOLATILITY
# ============================================================

target = pd.read_csv(
    TARGET_FILE
)


required_target_columns = [
    "target_date",
    "target_volatility"
]


missing_target = [
    col
    for col in required_target_columns
    if col not in target.columns
]


if missing_target:

    raise ValueError(
        "\nCanonical target file is missing:\n"
        f"{missing_target}\n\n"
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
# 10. CONVERT DATES
# ============================================================

arima["target_date"] = pd.to_datetime(
    arima["target_date"],
    errors="coerce"
)


bvar["target_date"] = pd.to_datetime(
    bvar["target_date"],
    errors="coerce"
)


cnn["target_date"] = pd.to_datetime(
    cnn["target_date"],
    errors="coerce"
)


target["target_date"] = pd.to_datetime(
    target["target_date"],
    errors="coerce"
)


# ============================================================
# 11. CONVERT FORECAST VALUES TO NUMERIC
# ============================================================

arima["ARIMA_GARCH_forecast"] = pd.to_numeric(
    arima["ARIMA_GARCH_forecast"],
    errors="coerce"
)


bvar["BVAR_forecast"] = pd.to_numeric(
    bvar["BVAR_forecast"],
    errors="coerce"
)


cnn["CNN_forecast"] = pd.to_numeric(
    cnn["CNN_forecast"],
    errors="coerce"
)


target["target_volatility"] = pd.to_numeric(
    target["target_volatility"],
    errors="coerce"
)


# ============================================================
# 12. REMOVE INVALID ROWS
# ============================================================

arima = arima.dropna(
    subset=[
        "target_date",
        "ARIMA_GARCH_forecast"
    ]
)


bvar = bvar.dropna(
    subset=[
        "target_date",
        "BVAR_forecast"
    ]
)


cnn = cnn.dropna(
    subset=[
        "target_date",
        "CNN_forecast"
    ]
)


target = target.dropna(
    subset=[
        "target_date",
        "target_volatility"
    ]
)


# ============================================================
# 13. REMOVE DUPLICATE TARGET DATES
# ============================================================

arima = (
    arima
    .sort_values("target_date")
    .drop_duplicates(
        subset=["target_date"],
        keep="first"
    )
)


bvar = (
    bvar
    .sort_values("target_date")
    .drop_duplicates(
        subset=["target_date"],
        keep="first"
    )
)


cnn = (
    cnn
    .sort_values("target_date")
    .drop_duplicates(
        subset=["target_date"],
        keep="first"
    )
)


target = (
    target
    .sort_values("target_date")
    .drop_duplicates(
        subset=["target_date"],
        keep="first"
    )
)


# ============================================================
# 14. MERGE THE THREE MODEL FORECASTS
# ============================================================

comparison = (
    arima
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
# 15. ADD THE CANONICAL ACTUAL TARGET
# ============================================================

comparison = (
    comparison
    .merge(
        target,
        on="target_date",
        how="inner"
    )
)


# ============================================================
# 16. CHECK COMMON SAMPLE
# ============================================================

comparison = (
    comparison
    .dropna(
        subset=[
            "target_date",
            "target_volatility",
            "ARIMA_GARCH_forecast",
            "BVAR_forecast",
            "CNN_forecast"
        ]
    )
    .sort_values("target_date")
    .reset_index(drop=True)
)


if comparison.empty:

    raise ValueError(
        "\nNo common target dates were found "
        "across all three models."
    )


# ============================================================
# 17. CHECK FOR NEGATIVE VOLATILITY VALUES
# ============================================================

if (
    comparison["target_volatility"]
    < 0
).any():

    raise ValueError(
        "\nNegative values were found in "
        "target_volatility."
    )


# ============================================================
# 18. DISPLAY COMMON SAMPLE INFORMATION
# ============================================================

common_start = (
    comparison["target_date"].min()
)

common_end = (
    comparison["target_date"].max()
)

common_n = len(
    comparison
)


print("\n")
print("=" * 90)
print("TABLE 4.13")
print("CONTINUOUS VOLATILITY FORECASTING COMPARISON")
print("=" * 90)


print(
    f"\nCommon observations : "
    f"{common_n:,}"
)


print(
    f"Common start date   : "
    f"{common_start.date()}"
)


print(
    f"Common end date     : "
    f"{common_end.date()}"
)


# ============================================================
# 19. METRIC FUNCTION
# ============================================================

def calculate_metrics(
    actual,
    forecast
):

    actual = np.asarray(
        actual,
        dtype=float
    )

    forecast = np.asarray(
        forecast,
        dtype=float
    )

    error = (
        actual
        - forecast
    )

    absolute_error = (
        np.abs(error)
    )

    squared_error = (
        error ** 2
    )

    mae = (
        np.mean(
            absolute_error
        )
    )

    mse = (
        np.mean(
            squared_error
        )
    )

    rmse = (
        np.sqrt(mse)
    )

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    }


# ============================================================
# 20. COMMON ACTUAL SERIES
# ============================================================

actual = (
    comparison["target_volatility"]
)


# ============================================================
# 21. CALCULATE METRICS FOR EACH MODEL
# ============================================================

model_definitions = [

    (
        "ARIMA(0,0,1)-GARCH(2,1)-t",
        "ARIMA_GARCH_forecast"
    ),

    (
        "BVAR(8)-log1p",
        "BVAR_forecast"
    ),

    (
        "Multivariate 2D-CNN-A",
        "CNN_forecast"
    )

]


results = []


for model_name, forecast_column in model_definitions:

    metrics = calculate_metrics(
        actual,
        comparison[
            forecast_column
        ]
    )

    results.append({

        "Model":
            model_name,

        "Observations":
            common_n,

        "Start":
            common_start.date(),

        "End":
            common_end.date(),

        "MAE":
            metrics["MAE"],

        "MSE":
            metrics["MSE"],

        "RMSE":
            metrics["RMSE"]

    })


results = pd.DataFrame(
    results
)


# ============================================================
# 22. RANK THE MODELS
# ============================================================

results["MAE Rank"] = (
    results["MAE"]
    .rank(
        method="min",
        ascending=True
    )
    .astype(int)
)


results["RMSE Rank"] = (
    results["RMSE"]
    .rank(
        method="min",
        ascending=True
    )
    .astype(int)
)


# ============================================================
# 23. OVERALL CONTINUOUS RANK
# ============================================================

results["Average Rank"] = (
    results[
        [
            "MAE Rank",
            "RMSE Rank"
        ]
    ]
    .mean(axis=1)
)


results["Overall Rank"] = (
    results["Average Rank"]
    .rank(
        method="min",
        ascending=True
    )
    .astype(int)
)


# ============================================================
# 24. ROUND DISPLAY VALUES
# ============================================================

display_table = results.copy()


display_table["MAE"] = (
    display_table["MAE"]
    .round(4)
)


display_table["MSE"] = (
    display_table["MSE"]
    .round(4)
)


display_table["RMSE"] = (
    display_table["RMSE"]
    .round(4)
)


display_table["Average Rank"] = (
    display_table["Average Rank"]
    .round(2)
)


# ============================================================
# 25. SAVE CSV
# ============================================================

display_table.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# 26. SAVE LATEX
# ============================================================

latex_table = display_table.to_latex(
    index=False,
    escape=True,
    caption=(
        "Continuous volatility forecasting performance "
        "of the ARIMA-GARCH, BVAR and multivariate 2D-CNN "
        "models on common out-of-sample target dates."
    ),
    label="tab:continuous_volatility_comparison",
    float_format="%.4f"
)


OUTPUT_TEX.write_text(
    latex_table,
    encoding="utf-8"
)


# ============================================================
# 27. PRINT RESULTS
# ============================================================

print("\n")
print("-" * 90)
print("MODEL RESULTS")
print("-" * 90)


print(
    display_table.to_string(
        index=False
    )
)


# ============================================================
# 28. IDENTIFY BEST MODEL
# ============================================================

best_model_row = (
    results
    .sort_values("Overall Rank")
    .iloc[0]
)


print("\n")
print("-" * 90)
print("BEST CONTINUOUS VOLATILITY MODEL")
print("-" * 90)


print(
    f"\nModel: "
    f"{best_model_row['Model']}"
)


print(
    f"MAE:   "
    f"{best_model_row['MAE']:.6f}"
)


print(
    f"RMSE:  "
    f"{best_model_row['RMSE']:.6f}"
)


# ============================================================
# 29. SAVE COMMON-SAMPLE FORECASTS
# ============================================================

COMMON_FORECAST_FILE = (
    RESULTS_DIR
    / "table_4_13_common_sample_forecasts.csv"
)


comparison.to_csv(
    COMMON_FORECAST_FILE,
    index=False
)


# ============================================================
# 30. FINAL OUTPUT INFORMATION
# ============================================================

print("\n")
print("=" * 90)
print("FILES SAVED")
print("=" * 90)


print(
    f"\nComparison CSV:\n"
    f"{OUTPUT_CSV}"
)


print(
    f"\nComparison LaTeX:\n"
    f"{OUTPUT_TEX}"
)


print(
    f"\nCommon-sample forecasts:\n"
    f"{COMMON_FORECAST_FILE}"
)


print("\n")
print("=" * 90)
print("TABLE 4.13 COMPLETE")
print("=" * 90)