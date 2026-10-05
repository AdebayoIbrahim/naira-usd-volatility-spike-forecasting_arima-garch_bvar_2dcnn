import pandas as pd
import numpy as np
import os

from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model


# ============================================================
# 1. FILES
# ============================================================

input_file = (
    "data_cleaned/forecasting_dataset_2006_2025.csv"
)

threshold_file = (
    "results/training_spike_threshold.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(
    input_file,
    parse_dates=["date", "target_date"]
)

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)

threshold_df = pd.read_csv(threshold_file)

spike_threshold = float(
    threshold_df.loc[0, "spike_threshold"]
)


# ============================================================
# 3. TRAINING DATA
# ============================================================

train_df = (
    df[df["dataset"] == "train"]
    .copy()
)

train_returns = (
    train_df["fx_log_return"]
    .dropna()
    .astype(float)
)

train_returns.index = pd.RangeIndex(
    len(train_returns)
)


# ============================================================
# 4. FIT ARIMA(0,0,1)
# ============================================================

arima_order = (0, 0, 1)

arima_model = ARIMA(
    train_returns,
    order=arima_order,
    trend="c"
)

arima_fit = arima_model.fit()


# ============================================================
# 5. ARIMA PARAMETERS
# ============================================================

arima_params = arima_fit.params

arima_const = float(
    arima_params["const"]
)

arima_ma1 = float(
    arima_params["ma.L1"]
)


# ============================================================
# 6. TRAINING ARIMA RESIDUALS
# ============================================================

train_residuals = (
    arima_fit.resid
    .dropna()
    .astype(float)
)

train_residuals.index = pd.RangeIndex(
    len(train_residuals)
)


# ============================================================
# 7. FIT GARCH(2,1)
# ============================================================

garch_model = arch_model(
    train_residuals,
    mean="Zero",
    vol="GARCH",
    p=2,
    q=1,
    dist="t",
    rescale=False
)

garch_fit = garch_model.fit(
    disp="off"
)


# ============================================================
# 8. GARCH PARAMETERS
# ============================================================

garch_params = garch_fit.params

omega = float(
    garch_params["omega"]
)

alpha1 = float(
    garch_params["alpha[1]"]
)

alpha2 = float(
    garch_params["alpha[2]"]
)

beta1 = float(
    garch_params["beta[1]"]
)


# ============================================================
# 9. UNCONDITIONAL VARIANCE
# ============================================================
#
# For GARCH(2,1), the long-run variance is approximately:
#
# omega / (1 - alpha1 - alpha2 - beta1)
#
# We use this as the starting variance whenever a new
# volatility segment begins.

persistence = (
    alpha1
    + alpha2
    + beta1
)

if persistence < 1:

    unconditional_variance = (
        omega / (1 - persistence)
    )

else:

    unconditional_variance = float(
        garch_fit.conditional_volatility.iloc[-1] ** 2
    )


# ============================================================
# 10. TEST DATA
# ============================================================

test_df = (
    df[df["dataset"] == "test"]
    .copy()
    .sort_values("date")
    .reset_index(drop=True)
)


# ============================================================
# 11. HELPER FUNCTION:
#     GET SEGMENT HISTORY UP TO FORECAST ORIGIN
# ============================================================

def get_current_segment(data, origin_date):

    history = (
        data[data["date"] <= origin_date]
        .copy()
        .sort_values("date")
    )

    if history.empty:
        return history

    # Locate most recent long-gap boundary
    boundary_positions = np.where(
        history["long_gap"].fillna(0).astype(int).to_numpy() == 1
    )[0]

    if len(boundary_positions) == 0:

        return history

    last_boundary = boundary_positions[-1]

    return history.iloc[last_boundary:].copy()


# ============================================================
# 12. WALK-FORWARD FORECASTING
# ============================================================

forecast_rows = []


for i in range(len(test_df)):

    row = test_df.iloc[i]

    forecast_date = row["date"]
    target_date = row["target_date"]


    # --------------------------------------------------------
    # A. GET ONLY THE CURRENT GAP-SAFE SEGMENT
    # --------------------------------------------------------

    segment_history = get_current_segment(
        df,
        forecast_date
    )


    # --------------------------------------------------------
    # B. GET VALID RETURNS IN CURRENT SEGMENT
    # --------------------------------------------------------

    segment_returns = (
        segment_history["fx_log_return"]
        .dropna()
        .astype(float)
        .to_numpy()
    )


    # Need at least 19 historical returns to construct
    # the 20-observation target window.

    if len(segment_returns) < 19:

        continue


    # --------------------------------------------------------
    # C. CALCULATE ARIMA RESIDUALS FOR CURRENT SEGMENT
    # --------------------------------------------------------
    #
    # ARIMA(0,0,1):
    #
    # r_t = c + theta * epsilon_(t-1) + epsilon_t
    #
    # We calculate the residuals sequentially using the
    # fixed training-estimated parameters.

    segment_residuals = []

    previous_residual = 0.0

    for r in segment_returns:

        fitted_mean = (
            arima_const
            + arima_ma1 * previous_residual
        )

        residual = (
            r - fitted_mean
        )

        segment_residuals.append(
            residual
        )

        previous_residual = residual


    segment_residuals = np.asarray(
        segment_residuals
    )


    # --------------------------------------------------------
    # D. CURRENT ARIMA STATE
    # --------------------------------------------------------

    eps_t = float(
        segment_residuals[-1]
    )


    # Previous residual
    if len(segment_residuals) >= 2:

        eps_t_minus_1 = float(
            segment_residuals[-2]
        )

    else:

        eps_t_minus_1 = 0.0


    # --------------------------------------------------------
    # E. CALCULATE CURRENT GARCH STATE
    # --------------------------------------------------------
    #
    # Start the segment at the unconditional variance and
    # recursively filter only the observations in this
    # segment.

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
    # F. ARIMA ONE-STEP MEAN FORECAST
    # --------------------------------------------------------

    mean_forecast = (
        arima_const
        + arima_ma1 * eps_t
    )


    # --------------------------------------------------------
    # G. GARCH ONE-STEP VARIANCE FORECAST
    # --------------------------------------------------------

    variance_forecast = (
        omega
        + alpha1 * (eps_t ** 2)
        + alpha2 * (eps_t_minus_1 ** 2)
        + beta1 * sigma2_t
    )

    variance_forecast = max(
        float(variance_forecast),
        1e-12
    )


    # --------------------------------------------------------
    # H. GET LAST 19 RETURNS
    # --------------------------------------------------------

    known_returns = (
        segment_returns[-19:]
    )


    # --------------------------------------------------------
    # I. FORECAST NEXT 20-OBSERVATION VOLATILITY
    # --------------------------------------------------------

    n = 20

    A = np.sum(
        known_returns ** 2
    )

    B = np.sum(
        known_returns
    )

    expected_future_squared_return = (
        variance_forecast
        + mean_forecast ** 2
    )

    expected_squared_window_sum = (
        (B + mean_forecast) ** 2
        + variance_forecast
    )

    expected_sample_variance = (
        1 / (n - 1)
    ) * (
        A
        + expected_future_squared_return
        - (
            expected_squared_window_sum / n
        )
    )

    expected_sample_variance = max(
        float(expected_sample_variance),
        0.0
    )

    predicted_volatility = np.sqrt(
        expected_sample_variance
    )


    # --------------------------------------------------------
    # J. ACTUAL TARGET
    # --------------------------------------------------------

    actual_volatility = float(
        row["target_volatility"]
    )

    actual_spike = int(
        row["target_spike"]
    )


    # --------------------------------------------------------
    # K. PREDICT SPIKE
    # --------------------------------------------------------

    predicted_spike = int(
        predicted_volatility > spike_threshold
    )


    # --------------------------------------------------------
    # L. SAVE FORECAST
    # --------------------------------------------------------

    forecast_rows.append({

        "forecast_date":
            forecast_date,

        "target_date":
            target_date,

        "predicted_volatility":
            predicted_volatility,

        "actual_volatility":
            actual_volatility,

        "predicted_spike":
            predicted_spike,

        "actual_spike":
            actual_spike,

        "spike_threshold":
            spike_threshold,

        "arima_mean_forecast":
            mean_forecast,

        "garch_variance_forecast":
            variance_forecast,

        "segment_length":
            len(segment_returns)

    })


# ============================================================
# 13. RESULTS DATAFRAME
# ============================================================

forecast_results = pd.DataFrame(
    forecast_rows
)


# ============================================================
# 14. FORECAST ERRORS
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
# 15. NUMERICAL FORECAST METRICS
# ============================================================

mae = (
    forecast_results["absolute_error"]
    .mean()
)

mse = (
    forecast_results["squared_error"]
    .mean()
)

rmse = np.sqrt(mse)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

actual = (
    forecast_results["actual_spike"]
)

predicted = (
    forecast_results["predicted_spike"]
)


tn = int(
    ((predicted == 0) & (actual == 0)).sum()
)

fp = int(
    ((predicted == 1) & (actual == 0)).sum()
)

fn = int(
    ((predicted == 0) & (actual == 1)).sum()
)

tp = int(
    ((predicted == 1) & (actual == 1)).sum()
)


# ============================================================
# 17. CLASSIFICATION METRICS
# ============================================================

accuracy = (
    (tp + tn)
    / (tp + tn + fp + fn)
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
    2 * precision * recall
    / (precision + recall)
    if (
        pd.notna(precision)
        and pd.notna(recall)
        and (precision + recall) > 0
    )
    else np.nan
)


# ============================================================
# 18. PRINT RESULTS
# ============================================================

print("\nARIMA-GARCH TEST RESULTS")
print("--------------------------------")

print(
    "Forecast observations:",
    len(forecast_results)
)

print(
    "MAE:",
    mae
)

print(
    "MSE:",
    mse
)

print(
    "RMSE:",
    rmse
)


print("\nCONFUSION MATRIX")
print("--------------------------------")

print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)


print("\nCLASSIFICATION METRICS")
print("--------------------------------")

print(
    "Accuracy:",
    accuracy
)

print(
    "Precision:",
    precision
)

print(
    "Recall:",
    recall
)

print(
    "F1:",
    f1
)


# ============================================================
# 19. SAVE FORECASTS
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

forecast_file = (
    "results/"
    "arima_garch_test_forecasts.csv"
)

forecast_results.to_csv(
    forecast_file,
    index=False
)


# ============================================================
# 20. SAVE METRICS
# ============================================================

metrics_file = (
    "results/"
    "arima_garch_test_metrics.csv"
)

pd.DataFrame({

    "model": [
        "ARIMA(0,0,1)-GARCH(2,1)"
    ],

    "distribution": [
        "Student-t"
    ],

    "forecast_observations": [
        len(forecast_results)
    ],

    "MAE": [
        mae
    ],

    "MSE": [
        mse
    ],

    "RMSE": [
        rmse
    ],

    "TN": [
        tn
    ],

    "FP": [
        fp
    ],

    "FN": [
        fn
    ],

    "TP": [
        tp
    ],

    "Accuracy": [
        accuracy
    ],

    "Precision": [
        precision
    ],

    "Recall": [
        recall
    ],

    "F1": [
        f1
    ],

    "spike_threshold": [
        spike_threshold
    ]

}).to_csv(
    metrics_file,
    index=False
)


print("\nSaved forecast results:")
print(forecast_file)

print("\nSaved metrics:")
print(metrics_file)

print("\nGap-aware ARIMA-GARCH forecasting completed.")