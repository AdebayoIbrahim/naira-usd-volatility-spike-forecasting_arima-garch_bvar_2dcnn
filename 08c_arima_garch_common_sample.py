# ============================================================
# STEP 08C: ARIMA-GARCH COMMON-SAMPLE WALK-FORWARD FORECASTING
# ============================================================
#
# Purpose:
# Evaluate the fixed ARIMA(0,0,1)-GARCH(2,1)-Student-t model
# on the EXACT same origin dates retained by the BVAR(8)
# walk-forward evaluation.
#
# The common evaluation grid is taken from:
# results/bvar8_spike_walk_forward_forecasts.csv
#
# The target is the NEXT observed squared FX log return:
# target_volatility = r_(t+1)^2
#
# This replaces the old 20-observation rolling-volatility target.
# ============================================================

import os
import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model


# ============================================================
# 1. FILES / SETTINGS
# ============================================================

INPUT_FILE = (
    "data_cleaned/"
    "forecasting_dataset_2006_2025_spike_gap_target_safe.csv"
)

BVAR_GRID_FILE = (
    "results/"
    "bvar8_spike_walk_forward_forecasts.csv"
)

THRESHOLD_FILE = (
    "results/"
    "training_spike_threshold.csv"
)

OUTPUT_FILE = (
    "results/"
    "arima_garch_common_sample_forecasts.csv"
)

METRICS_FILE = (
    "results/"
    "arima_garch_common_sample_metrics.csv"
)

TRAIN_END = "2020-12-31"

ARIMA_ORDER = (0, 0, 1)
GARCH_P = 2
GARCH_Q = 1


# ============================================================
# 2. LOAD DATA
# ============================================================

os.makedirs("results", exist_ok=True)

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])
df["target_date"] = pd.to_datetime(df["target_date"])

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)

bvar_grid = pd.read_csv(BVAR_GRID_FILE)
bvar_grid["origin_date"] = pd.to_datetime(bvar_grid["origin_date"])
bvar_grid["target_date"] = pd.to_datetime(bvar_grid["target_date"])

threshold_df = pd.read_csv(THRESHOLD_FILE)
spike_threshold = float(threshold_df.loc[0, "spike_threshold"])

print("=" * 90)
print("STEP 08C: ARIMA-GARCH COMMON-SAMPLE WALK-FORWARD FORECASTING")
print("=" * 90)

print(f"\nAnalysis-ready rows loaded: {len(df):,}")
print(f"BVAR common-sample origins loaded: {len(bvar_grid):,}")
print(f"Frozen spike threshold: {spike_threshold:.8f}")


# ============================================================
# 3. VALIDATE COMMON EVALUATION GRID
# ============================================================

required_grid_columns = {
    "origin_date",
    "target_date",
}

missing_grid_columns = required_grid_columns - set(bvar_grid.columns)

if missing_grid_columns:
    raise ValueError(
        "BVAR forecast file is missing required columns: "
        f"{sorted(missing_grid_columns)}"
    )

if bvar_grid["origin_date"].duplicated().any():
    raise ValueError("Duplicate origin dates found in BVAR common-sample grid.")

if (bvar_grid["target_date"] <= bvar_grid["origin_date"]).any():
    raise ValueError("Invalid target date found in BVAR common-sample grid.")

# Match each BVAR origin to the corresponding row in the forecasting dataset.
eval_grid = (
    bvar_grid[["origin_date", "target_date"]]
    .merge(
        df,
        left_on="origin_date",
        right_on="date",
        how="left",
        validate="one_to_one",
    )
    .sort_values("origin_date")
    .reset_index(drop=True)
)

if eval_grid["date"].isna().any():
    missing_dates = eval_grid.loc[
        eval_grid["date"].isna(), "origin_date"
    ]
    raise ValueError(
        "Some BVAR forecast origins do not exist in the analysis dataset. "
        f"Examples: {missing_dates.head().tolist()}"
    )

# The target date must agree between the BVAR grid and Step 09H.
target_date_mismatch = (
    eval_grid["target_date_x"] != eval_grid["target_date_y"]
).sum()

if target_date_mismatch > 0:
    raise ValueError(
        f"Found {target_date_mismatch} origin rows with mismatched target dates."
    )

# Rename the single target-date field used below.
eval_grid["target_date"] = eval_grid["target_date_x"]

# Remove merge helper columns.
eval_grid = eval_grid.drop(
    columns=["target_date_x", "target_date_y"]
)

print(
    f"\nCommon evaluation origins confirmed: "
    f"{len(eval_grid):,}"
)
print(
    f"First origin: {eval_grid['origin_date'].min().date()}"
)
print(
    f"Last origin:  {eval_grid['origin_date'].max().date()}"
)


# ============================================================
# 4. TRAINING DATA
# ============================================================

train_df = (
    df[df["date"] <= TRAIN_END]
    .copy()
)

train_returns = (
    train_df["fx_log_return"]
    .dropna()
    .astype(float)
)

train_returns.index = pd.RangeIndex(len(train_returns))

print("\n" + "=" * 90)
print("MODEL TRAINING")
print("=" * 90)
print(f"\nTraining observations: {len(train_returns):,}")
print(f"Training end: {TRAIN_END}")


# ============================================================
# 5. FIT FIXED ARIMA(0,0,1)
# ============================================================

arima_model = ARIMA(
    train_returns,
    order=ARIMA_ORDER,
    trend="c",
)

arima_fit = arima_model.fit()
arima_params = arima_fit.params

arima_const = float(arima_params["const"])
arima_ma1 = float(arima_params["ma.L1"])

print(f"ARIMA order: {ARIMA_ORDER}")
print(f"ARIMA constant: {arima_const:.10f}")
print(f"ARIMA MA(1): {arima_ma1:.10f}")


# ============================================================
# 6. FIT FIXED GARCH(2,1)-STUDENT-t TO ARIMA RESIDUALS
# ============================================================

train_residuals = (
    arima_fit.resid
    .dropna()
    .astype(float)
)

train_residuals.index = pd.RangeIndex(len(train_residuals))

garch_model = arch_model(
    train_residuals,
    mean="Zero",
    vol="GARCH",
    p=GARCH_P,
    q=GARCH_Q,
    dist="t",
    rescale=False,
)

garch_fit = garch_model.fit(disp="off")
garch_params = garch_fit.params

omega = float(garch_params["omega"])
alpha1 = float(garch_params["alpha[1]"])
alpha2 = float(garch_params["alpha[2]"])
beta1 = float(garch_params["beta[1]"])

persistence = alpha1 + alpha2 + beta1

if persistence < 1:
    unconditional_variance = omega / (1 - persistence)
else:
    unconditional_variance = float(
        garch_fit.conditional_volatility.iloc[-1] ** 2
    )

print(f"GARCH order: ({GARCH_P},{GARCH_Q})")
print(f"Distribution: Student-t")
print(f"Omega:  {omega:.10f}")
print(f"Alpha1: {alpha1:.10f}")
print(f"Alpha2: {alpha2:.10f}")
print(f"Beta1:  {beta1:.10f}")
print(f"Persistence: {persistence:.10f}")
print(f"Starting unconditional variance: {unconditional_variance:.10f}")


# ============================================================
# 7. PREPARE GAP-SAFE SEGMENTS
# ============================================================

segments = {}
segment_date_positions = {}

for segment_id, segment in df.groupby("segment_id", sort=False):
    segment = (
        segment
        .sort_values("date")
        .reset_index(drop=True)
    )

    segments[segment_id] = segment
    segment_date_positions[segment_id] = {
        date: i
        for i, date in enumerate(segment["date"])
    }


# ============================================================
# 8. WALK-FORWARD FORECASTING ON EXACT BVAR ORIGINS
# ============================================================

forecast_rows = []
excluded_origins = []

print("\n" + "=" * 90)
print("GENERATING COMMON-SAMPLE WALK-FORWARD FORECASTS")
print("=" * 90)

for counter, origin_row in enumerate(
    eval_grid.itertuples(index=False),
    start=1,
):

    origin_date = pd.Timestamp(origin_row.origin_date)
    target_date = pd.Timestamp(origin_row.target_date)
    segment_id = origin_row.segment_id

    segment = segments[segment_id]
    date_positions = segment_date_positions[segment_id]

    if origin_date not in date_positions:
        raise ValueError(
            f"Could not locate origin {origin_date} in segment {segment_id}."
        )

    position = date_positions[origin_date]

    # All observations up to the origin, but never across a long gap.
    history = segment.iloc[: position + 1]

    segment_returns = (
        history["fx_log_return"]
        .dropna()
        .astype(float)
        .to_numpy()
    )

    if len(segment_returns) == 0:
        excluded_origins.append({
            "origin_date": origin_date,
            "target_date": target_date,
            "segment_id": segment_id,
            "reason": "No valid FX return history in segment",
        })
        continue

    # --------------------------------------------------------
    # A. Reconstruct ARIMA residual state within the segment
    # --------------------------------------------------------

    segment_residuals = []
    previous_residual = 0.0

    for r in segment_returns:
        fitted_mean = (
            arima_const
            + arima_ma1 * previous_residual
        )

        residual = r - fitted_mean
        segment_residuals.append(residual)
        previous_residual = residual

    segment_residuals = np.asarray(segment_residuals, dtype=float)

    eps_t = float(segment_residuals[-1])

    if len(segment_residuals) >= 2:
        eps_t_minus_1 = float(segment_residuals[-2])
    else:
        eps_t_minus_1 = 0.0

    # --------------------------------------------------------
    # B. Reconstruct current GARCH state within the segment
    # --------------------------------------------------------

    sigma2 = unconditional_variance
    previous_eps_1 = 0.0
    previous_eps_2 = 0.0

    for residual in segment_residuals:
        sigma2 = (
            omega
            + alpha1 * (previous_eps_1 ** 2)
            + alpha2 * (previous_eps_2 ** 2)
            + beta1 * sigma2
        )

        previous_eps_2 = previous_eps_1
        previous_eps_1 = residual

    sigma2_t = float(sigma2)

    # --------------------------------------------------------
    # C. One-step ARIMA mean forecast
    # --------------------------------------------------------

    mean_forecast = (
        arima_const
        + arima_ma1 * eps_t
    )

    # --------------------------------------------------------
    # D. One-step GARCH variance forecast
    # --------------------------------------------------------

    variance_forecast = (
        omega
        + alpha1 * (eps_t ** 2)
        + alpha2 * (eps_t_minus_1 ** 2)
        + beta1 * sigma2_t
    )

    variance_forecast = max(
        float(variance_forecast),
        1e-12,
    )

    # --------------------------------------------------------
    # E. Forecast NEXT squared return
    # --------------------------------------------------------
    #
    # For r_(t+1) = mu_(t+1) + epsilon_(t+1):
    # E[r_(t+1)^2 | I_t] = variance_forecast + mean_forecast^2
    #
    # This is the same target definition used in Step 09H.

    predicted_volatility = (
        variance_forecast
        + (mean_forecast ** 2)
    )

    predicted_volatility = max(
        float(predicted_volatility),
        0.0,
    )

    # --------------------------------------------------------
    # F. Actual next-observation target
    # --------------------------------------------------------

    actual_volatility = float(origin_row.target_volatility)
    actual_spike = int(origin_row.target_spike)

    # --------------------------------------------------------
    # G. Spike classification using the FROZEN threshold
    # --------------------------------------------------------

    predicted_spike = int(
        predicted_volatility > spike_threshold
    )

    forecast_rows.append({
        "origin_date": origin_date,
        "target_date": target_date,
        "segment_id": segment_id,
        "actual_volatility": actual_volatility,
        "predicted_volatility": predicted_volatility,
        "actual_spike": actual_spike,
        "predicted_spike": predicted_spike,
        "spike_threshold": spike_threshold,
        "arima_mean_forecast": mean_forecast,
        "garch_variance_forecast": variance_forecast,
        "segment_length": len(segment_returns),
    })

    if counter % 250 == 0:
        print(
            f"Processed: {counter:,}/{len(eval_grid):,} | "
            f"Forecasts retained: {len(forecast_rows):,}"
        )


# ============================================================
# 9. RESULTS DATAFRAME
# ============================================================

forecast_results = pd.DataFrame(forecast_rows)
excluded_df = pd.DataFrame(excluded_origins)

if forecast_results.empty:
    raise ValueError("No ARIMA-GARCH forecasts were generated.")


# ============================================================
# 10. FINAL SAMPLE CHECKS
# ============================================================

print("\n" + "=" * 90)
print("COMMON-SAMPLE CHECKS")
print("=" * 90)

print(f"\nBVAR origins supplied: {len(eval_grid):,}")
print(f"ARIMA-GARCH forecasts retained: {len(forecast_results):,}")
print(f"ARIMA-GARCH origins excluded: {len(excluded_df):,}")

if len(excluded_df) > 0:
    print("\nExcluded origins:")
    print(excluded_df.to_string(index=False))

# The purpose of the step is exact same-origin evaluation.
if set(forecast_results["origin_date"]) != set(eval_grid["origin_date"]):
    missing = sorted(
        set(eval_grid["origin_date"]) -
        set(forecast_results["origin_date"])
    )
    extra = sorted(
        set(forecast_results["origin_date"]) -
        set(eval_grid["origin_date"])
    )

    raise ValueError(
        "Common-sample mismatch detected. "
        f"Missing origins: {len(missing)}; extra origins: {len(extra)}."
    )

if forecast_results["origin_date"].duplicated().any():
    raise ValueError("Duplicate ARIMA-GARCH forecast origins detected.")


# ============================================================
# 11. FORECAST ERRORS
# ============================================================

forecast_results["error"] = (
    forecast_results["actual_volatility"]
    - forecast_results["predicted_volatility"]
)

forecast_results["absolute_error"] = (
    forecast_results["error"].abs()
)

forecast_results["squared_error"] = (
    forecast_results["error"] ** 2
)


# ============================================================
# 12. NUMERICAL FORECAST METRICS
# ============================================================

mae = float(
    forecast_results["absolute_error"].mean()
)

mse = float(
    forecast_results["squared_error"].mean()
)

rmse = float(np.sqrt(mse))


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

actual = forecast_results["actual_spike"]
predicted = forecast_results["predicted_spike"]

tn = int(((predicted == 0) & (actual == 0)).sum())
fp = int(((predicted == 1) & (actual == 0)).sum())
fn = int(((predicted == 0) & (actual == 1)).sum())
tp = int(((predicted == 1) & (actual == 1)).sum())


# ============================================================
# 14. CLASSIFICATION METRICS
# ============================================================

total = tp + tn + fp + fn

accuracy = (
    (tp + tn) / total
    if total > 0
    else np.nan
)

precision = (
    tp / (tp + fp)
    if (tp + fp) > 0
    else np.nan
)

recall = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else np.nan
)

f1 = (
    2 * precision * recall / (precision + recall)
    if (
        pd.notna(precision)
        and pd.notna(recall)
        and (precision + recall) > 0
    )
    else np.nan
)


# ============================================================
# 15. PRINT RESULTS
# ============================================================

print("\n" + "=" * 90)
print("ARIMA-GARCH COMMON-SAMPLE RESULTS")
print("=" * 90)

print(f"\nForecast observations: {len(forecast_results):,}")
print(f"MAE:  {mae:.10f}")
print(f"MSE:  {mse:.10f}")
print(f"RMSE: {rmse:.10f}")

print("\nCONFUSION MATRIX")
print("--------------------------------")
print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TP: {tp}")

print("\nCLASSIFICATION METRICS")
print("--------------------------------")
print(f"Accuracy:  {accuracy:.10f}")
print(f"Precision: {precision:.10f}")
print(f"Recall:    {recall:.10f}")
print(f"F1:        {f1:.10f}")


# ============================================================
# 16. SAVE FORECASTS
# ============================================================

forecast_results.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# 17. SAVE METRICS
# ============================================================

metrics = pd.DataFrame({
    "model": [
        "ARIMA(0,0,1)-GARCH(2,1)"
    ],
    "distribution": [
        "Student-t"
    ],
    "training_end": [
        TRAIN_END
    ],
    "evaluation_sample": [
        "Exact BVAR(8) retained origins"
    ],
    "forecast_observations": [
        len(forecast_results)
    ],
    "MAE": [mae],
    "MSE": [mse],
    "RMSE": [rmse],
    "TN": [tn],
    "FP": [fp],
    "FN": [fn],
    "TP": [tp],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1": [f1],
    "spike_threshold": [spike_threshold],
})

metrics.to_csv(
    METRICS_FILE,
    index=False,
)


# ============================================================
# 18. SHOW FIRST / LAST FORECASTS
# ============================================================

print("\nFIRST 10 FORECASTS")
print("--------------------------------")
print(
    forecast_results.head(10).to_string(index=False)
)

print("\nLAST 10 FORECASTS")
print("--------------------------------")
print(
    forecast_results.tail(10).to_string(index=False)
)

print("\n" + "=" * 90)
print("RESULT FILES SAVED")
print("=" * 90)
print(f"\nForecasts: {OUTPUT_FILE}")
print(f"Metrics:   {METRICS_FILE}")
print("\nCommon-sample ARIMA-GARCH forecasting completed.")
