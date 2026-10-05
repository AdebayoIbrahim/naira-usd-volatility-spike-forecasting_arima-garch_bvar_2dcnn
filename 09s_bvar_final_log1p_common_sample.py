#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 20:07:35 2026

@author: setup
"""

# ============================================================
# STEP 09S: FINAL BVAR(8) WITH VALIDATED LOG1P TARGET
# ============================================================
#
# Step 09R compared two BVAR target specifications using
# historical validation blocks only:
#
#   1. Raw target:   fx_return_squared
#   2. Log1p target: log1p(fx_return_squared)
#
# The multi-block validation supported carrying the log1p
# specification forward because it reduced pooled MAE/RMSE and
# dramatically reduced the overprediction of spikes seen with
# the raw-target BVAR.
#
# FINAL MODEL:
#   BVAR(8)
#   Variables:
#       fx_log_return
#       fx_return_squared
#       brent_log_return
#       reserves_log_change
#   Target in model: log1p(fx_return_squared)
#   Lambda = 0.10
#   Intercept prior variance = 100
#   Prior df = 6
#   Posterior draws = 3,000
#
# The final model is refit using ALL observations through
# 2020-12-31. The 2021-2025 final test set is then evaluated
# on the exact same 1,083 common-sample origins used by the
# previous BVAR specification.
#
# IMPORTANT:
# - Spike threshold remains defined on the ORIGINAL squared-return
#   scale: 0.10927724.
# - For the log1p model, the threshold is transformed consistently:
#       log1p(0.10927724)
# - Continuous forecasts are transformed back to the original scale
#   using the Normal predictive mean:
#       E[v] = exp(mu + 0.5*variance) - 1
# - The final test set is not used to choose lambda, target transform,
#   cutoff, or any other specification.
# ============================================================

import os

import numpy as np
import pandas as pd
from scipy.stats import invwishart, norm
from sklearn.metrics import roc_auc_score


# ============================================================
# 1. SETTINGS
# ============================================================

INPUT_FILE = (
    "data_cleaned/"
    "forecasting_dataset_2006_2025_spike_gap_target_safe.csv"
)

COMMON_GRID_FILE = (
    "results/"
    "bvar8_spike_walk_forward_forecasts.csv"
)

OUTPUT_FILE = (
    "results/"
    "bvar8_final_log1p_common_sample_forecasts.csv"
)

METRICS_FILE = (
    "results/"
    "bvar8_final_log1p_common_sample_metrics.csv"
)

STATE_FILE = (
    "results/"
    "bvar8_final_log1p_model_state.npz"
)

TRAIN_END = "2020-12-31"

SPIKE_THRESHOLD = 0.10927724

PROBABILITY_CUTOFF = 0.50

LAG_ORDER = 8

LAMBDA = 0.10

INTERCEPT_PRIOR_VARIANCE = 100.0

PRIOR_DF = 6

N_DRAWS = 3000

RANDOM_SEED = 2026

VARIABLES = [
    "fx_log_return",
    "fx_return_squared",
    "brent_log_return",
    "reserves_log_change",
]

TARGET_VARIABLE = "fx_return_squared"

TARGET_INDEX = VARIABLES.index(
    TARGET_VARIABLE
)

K = len(VARIABLES)


# ============================================================
# 2. SETUP
# ============================================================

os.makedirs(
    "results",
    exist_ok=True,
)


# ============================================================
# 3. LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=[
        "date",
        "target_date",
    ],
)

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)

required_columns = [
    "date",
    "target_date",
    "segment_id",
    "fx_log_return",
    "fx_return_squared",
    "brent_log_return",
    "reserves_log_change",
    "target_volatility",
    "target_spike",
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Input file is missing required columns: "
        f"{missing_columns}"
    )


# ============================================================
# 4. LOAD EXACT COMMON-SAMPLE ORIGINS
# ============================================================

common_grid = pd.read_csv(
    COMMON_GRID_FILE,
    parse_dates=[
        "origin_date",
        "target_date",
    ],
)

required_grid_columns = [
    "origin_date",
    "target_date",
]

missing_grid_columns = [
    col
    for col in required_grid_columns
    if col not in common_grid.columns
]

if missing_grid_columns:
    raise ValueError(
        "Common-sample file is missing: "
        f"{missing_grid_columns}"
    )

common_grid = (
    common_grid[
        required_grid_columns
    ]
    .drop_duplicates()
    .sort_values("origin_date")
    .reset_index(drop=True)
)

if len(common_grid) != 1083:
    raise ValueError(
        "Expected exactly 1,083 common-sample origins, "
        f"found {len(common_grid):,}."
    )


# ============================================================
# 5. FINAL TRAINING DATA
# ============================================================

train = df[
    df["date"] <= TRAIN_END
].copy()

if train.empty:
    raise ValueError(
        "No training observations found."
    )

if (train[TARGET_VARIABLE] < 0).any():
    raise ValueError(
        "fx_return_squared contains negative values; "
        "log1p transformation is invalid."
    )

# Only the target is transformed. The other endogenous variables
# stay on their original scales before standardization.
train_model = train.copy()
train_model[TARGET_VARIABLE] = np.log1p(
    train_model[TARGET_VARIABLE]
)

print("=" * 90)
print("STEP 09S: FINAL BVAR(8) WITH VALIDATED LOG1P TARGET")
print("=" * 90)

print(
    f"\nRows loaded: {len(df):,}"
)

print(
    f"Training observations: {len(train):,}"
)

print(
    f"Training end: {TRAIN_END}"
)

print(
    f"Final common-sample origins: "
    f"{len(common_grid):,}"
)

print(
    f"Selected lambda: {LAMBDA:.2f}"
)

print(
    f"Target transform: log1p({TARGET_VARIABLE})"
)

print(
    f"Spike threshold (raw scale): "
    f"{SPIKE_THRESHOLD:.8f}"
)

print(
    f"Spike threshold (log1p scale): "
    f"{np.log1p(SPIKE_THRESHOLD):.8f}"
)

print(
    f"Probability cutoff: {PROBABILITY_CUTOFF:.2f}"
)


# ============================================================
# 6. BUILD GAP-SAFE DESIGN MATRIX
# ============================================================

def build_design_matrix(
    data,
    variables,
    means,
    stds,
    lag_order,
):
    X_rows = []
    Y_rows = []

    for _, segment in data.groupby(
        "segment_id",
        sort=True,
    ):

        segment = (
            segment
            .sort_values("date")
            .reset_index(drop=True)
        )

        values = segment[
            variables
        ].to_numpy(
            dtype=float,
            copy=True,
        )

        scaled = (
            values - means
        ) / stds

        if len(scaled) <= lag_order:
            continue

        for position in range(
            lag_order,
            len(scaled),
        ):

            x = [1.0]

            for lag in range(
                1,
                lag_order + 1,
            ):
                x.extend(
                    scaled[
                        position - lag
                    ]
                )

            X_rows.append(x)

            Y_rows.append(
                scaled[position]
            )

    return (
        np.asarray(
            X_rows,
            dtype=float,
        ),
        np.asarray(
            Y_rows,
            dtype=float,
        ),
    )


# ============================================================
# 7. PREPARE MODEL MOMENTS
# ============================================================

values = train_model[
    VARIABLES
].to_numpy(
    dtype=float,
    copy=True,
)

if np.isnan(values).any():
    raise ValueError(
        "Missing values found in training variables."
    )

means = values.mean(
    axis=0
)

stds = values.std(
    axis=0,
    ddof=0,
)

stds = np.where(
    stds <= 0,
    1.0,
    stds,
)


# ============================================================
# 8. BUILD FINAL DESIGN MATRIX
# ============================================================

X, Y = build_design_matrix(
    train_model,
    VARIABLES,
    means,
    stds,
    LAG_ORDER,
)

if len(X) == 0:
    raise ValueError(
        "No usable BVAR design rows."
    )

T = len(Y)
M = X.shape[1]

print("\n" + "=" * 90)
print("FINAL BVAR FIT")
print("=" * 90)

print(
    f"\nEffective BVAR observations: {T:,}"
)

print(
    f"Regressors per equation: {M}"
)

print(
    f"Lag order: VAR({LAG_ORDER})"
)

print(
    f"Lambda: {LAMBDA:.2f}"
)

print(
    f"Posterior draws: {N_DRAWS:,}"
)


# ============================================================
# 9. PRIOR
# ============================================================

B0 = np.zeros(
    (
        M,
        K,
    )
)

V0_diag = np.empty(
    M
)

V0_diag[0] = INTERCEPT_PRIOR_VARIANCE

index = 1

for lag in range(
    1,
    LAG_ORDER + 1,
):

    coefficient_variance = (
        LAMBDA / lag
    ) ** 2

    V0_diag[
        index:
        index + K
    ] = coefficient_variance

    index += K

V0_inv = np.diag(
    1.0 / V0_diag
)

S0 = np.eye(K)


# ============================================================
# 10. POSTERIOR
# ============================================================

Vn_inv = (
    V0_inv
    + X.T @ X
)

Vn = np.linalg.inv(
    Vn_inv
)

Bn = (
    Vn
    @ (
        V0_inv @ B0
        + X.T @ Y
    )
)

residuals = (
    Y
    - X @ Bn
)

Sn = (
    S0
    + residuals.T @ residuals
    + (
        Bn.T
        @ V0_inv
        @ Bn
    )
)

nu_n = PRIOR_DF + T


# ============================================================
# 11. POSTERIOR DRAWS
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

posterior_B = np.empty(
    (
        N_DRAWS,
        M,
        K,
    )
)

posterior_Sigma = np.empty(
    (
        N_DRAWS,
        K,
        K,
    )
)

chol_V = np.linalg.cholesky(
    Vn
)

for draw in range(
    N_DRAWS
):

    sigma_draw = invwishart.rvs(
        df=nu_n,
        scale=Sn,
        random_state=rng,
    )

    sigma_draw = np.asarray(
        sigma_draw,
        dtype=float,
    )

    sigma_draw = (
        sigma_draw
        + sigma_draw.T
    ) / 2

    posterior_Sigma[draw] = (
        sigma_draw
    )

    chol_sigma = np.linalg.cholesky(
        sigma_draw
    )

    Z = rng.standard_normal(
        (
            M,
            K,
        )
    )

    posterior_B[draw] = (
        Bn
        + chol_V
        @ Z
        @ chol_sigma.T
    )


# ============================================================
# 12. SAVE MODEL STATE
# ============================================================

np.savez(
    STATE_FILE,
    posterior_B=posterior_B,
    posterior_Sigma=posterior_Sigma,
    means=means,
    stds=stds,
    lag_order=LAG_ORDER,
    train_end=TRAIN_END,
    lambda_value=LAMBDA,
    spike_threshold=SPIKE_THRESHOLD,
    target_transform="log1p",
    target_variable=TARGET_VARIABLE,
    probability_cutoff=PROBABILITY_CUTOFF,
)


# ============================================================
# 13. WALK-FORWARD FORECASTING
# ============================================================

forecast_rows = []
processed = 0

for _, grid_row in common_grid.iterrows():

    origin_date = grid_row[
        "origin_date"
    ]

    target_date = grid_row[
        "target_date"
    ]

    origin_match = df[
        df["date"] == origin_date
    ]

    if len(origin_match) != 1:
        raise ValueError(
            "Could not uniquely locate "
            f"origin {origin_date}."
        )

    origin_row = origin_match.iloc[0]

    segment_id = origin_row[
        "segment_id"
    ]

    segment = df[
        (
            df["segment_id"]
            == segment_id
        )
        & (
            df["date"]
            <= origin_date
        )
    ].copy()

    segment = (
        segment
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(segment) <= LAG_ORDER:
        raise ValueError(
            "Common-sample origin does not have "
            "enough within-segment history: "
            f"{origin_date}"
        )

    # Transform only the target before applying the training-scale
    # standardization. This must match the model-fitting pipeline.
    history = segment[
        VARIABLES
    ].to_numpy(
        dtype=float,
        copy=True,
    )

    history[:, TARGET_INDEX] = np.log1p(
        history[:, TARGET_INDEX]
    )

    history_scaled = (
        history
        - means
    ) / stds

    x = [1.0]

    for lag in range(
        1,
        LAG_ORDER + 1,
    ):
        x.extend(
            history_scaled[-lag]
        )

    x = np.asarray(
        x,
        dtype=float,
    )

    if len(x) != M:
        raise ValueError(
            f"Design vector mismatch at "
            f"{origin_date}."
        )

    # --------------------------------------------------------
    # Conditional posterior target means on the standardized
    # log1p scale.
    # --------------------------------------------------------

    mu_draws_scaled = np.einsum(
        "m,nm->n",
        x,
        posterior_B[
            :,
            :,
            TARGET_INDEX,
        ],
    )

    # Conditional posterior variances on the standardized scale.
    var_draws_scaled = np.maximum(
        posterior_Sigma[
            :,
            TARGET_INDEX,
            TARGET_INDEX,
        ],
        1e-12,
    )

    # Convert conditional mean and variance to the log1p target
    # scale, undoing the standardization used for model fitting.
    mu_draws_log = (
        mu_draws_scaled
        * stds[TARGET_INDEX]
        + means[TARGET_INDEX]
    )

    var_draws_log = (
        var_draws_scaled
        * (
            stds[TARGET_INDEX]
            ** 2
        )
    )

    # --------------------------------------------------------
    # Continuous forecast on original squared-return scale.
    # --------------------------------------------------------
    # If Z ~ N(mu, variance), then
    # E[exp(Z) - 1] = exp(mu + variance/2) - 1.
    # --------------------------------------------------------

    predicted_draw_means_original = (
        np.exp(
            np.clip(
                mu_draws_log
                + 0.5 * var_draws_log,
                -700,
                700,
            )
        )
        - 1.0
    )

    predicted_volatility = float(
        np.mean(
            predicted_draw_means_original
        )
    )

    predicted_volatility = max(
        0.0,
        predicted_volatility,
    )

    # --------------------------------------------------------
    # Probability of exceeding the frozen spike threshold.
    # --------------------------------------------------------

    threshold_log = np.log1p(
        SPIKE_THRESHOLD
    )

    threshold_scaled = (
        threshold_log
        - means[TARGET_INDEX]
    ) / stds[TARGET_INDEX]

    z = (
        threshold_scaled
        - mu_draws_scaled
    ) / np.sqrt(
        var_draws_scaled
    )

    probability = float(
        np.mean(
            1.0
            - norm.cdf(z)
        )
    )

    predicted_spike = int(
        probability
        >= PROBABILITY_CUTOFF
    )

    actual_volatility = float(
        origin_row[
            "target_volatility"
        ]
    )

    actual_spike = int(
        origin_row[
            "target_spike"
        ]
    )

    # Verify the supplied target date matches the forecasting dataset.
    stored_target_date = origin_row[
        "target_date"
    ]

    if pd.isna(stored_target_date):
        raise ValueError(
            f"Missing target date at {origin_date}."
        )

    if stored_target_date != target_date:
        raise ValueError(
            f"Target-date mismatch at {origin_date}."
        )

    forecast_rows.append({
        "origin_date": origin_date,
        "target_date": target_date,
        "segment_id": segment_id,
        "actual_volatility": actual_volatility,
        "forecast_volatility": predicted_volatility,
        "actual_spike": actual_spike,
        "predicted_spike_probability": probability,
        "predicted_spike": predicted_spike,
        "probability_cutoff": PROBABILITY_CUTOFF,
        "spike_threshold": SPIKE_THRESHOLD,
        "target_transform": "log1p",
        "lambda": LAMBDA,
    })

    processed += 1

    if processed % 250 == 0:
        print(
            f"Processed: "
            f"{processed:,}/{len(common_grid):,}"
        )


# ============================================================
# 14. FORECAST DATAFRAME
# ============================================================

forecast_df = pd.DataFrame(
    forecast_rows
)

if len(forecast_df) != 1083:
    raise ValueError(
        "Final forecast count is not 1,083: "
        f"{len(forecast_df):,}"
    )


# ============================================================
# 15. METRICS
# ============================================================

error = (
    forecast_df["actual_volatility"]
    - forecast_df["forecast_volatility"]
)

mae = float(
    error.abs().mean()
)

mse = float(
    (error ** 2).mean()
)

rmse = float(
    np.sqrt(mse)
)

actual = forecast_df[
    "actual_spike"
]

predicted = forecast_df[
    "predicted_spike"
]

tn = int(
    (
        (predicted == 0)
        & (actual == 0)
    ).sum()
)

fp = int(
    (
        (predicted == 1)
        & (actual == 0)
    ).sum()
)

fn = int(
    (
        (predicted == 0)
        & (actual == 1)
    ).sum()
)

tp = int(
    (
        (predicted == 1)
        & (actual == 1)
    ).sum()
)

total = (
    tn + fp + fn + tp
)

accuracy = (
    (tp + tn) / total
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

auc_volatility = roc_auc_score(
    actual,
    forecast_df[
        "forecast_volatility"
    ],
)

auc_probability = roc_auc_score(
    actual,
    forecast_df[
        "predicted_spike_probability"
    ],
)


# ============================================================
# 16. METRIC TABLE
# ============================================================

metrics = {
    "model": "BVAR(8)-log1p",
    "forecast_observations": len(forecast_df),
    "actual_spikes": int(actual.sum()),
    "predicted_spikes": int(predicted.sum()),
    "actual_spike_rate": float(actual.mean()),
    "predicted_spike_rate": float(predicted.mean()),
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "AUC_volatility": auc_volatility,
    "AUC_probability": auc_probability,
    "mean_probability": float(
        forecast_df[
            "predicted_spike_probability"
        ].mean()
    ),
    "probability_std": float(
        forecast_df[
            "predicted_spike_probability"
        ].std()
    ),
    "lambda": LAMBDA,
    "lag_order": LAG_ORDER,
    "spike_threshold": SPIKE_THRESHOLD,
    "probability_cutoff": PROBABILITY_CUTOFF,
    "target_transform": "log1p",
}

metrics_df = pd.DataFrame(
    [metrics]
)


# ============================================================
# 17. SAVE RESULTS
# ============================================================

forecast_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

metrics_df.to_csv(
    METRICS_FILE,
    index=False,
)


# ============================================================
# 18. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 90)
print("FINAL BVAR(8)-LOG1P COMMON-SAMPLE RESULTS")
print("=" * 90)

print(
    f"\nForecast observations: "
    f"{len(forecast_df):,}"
)

print(
    f"MAE:  {mae:.10f}"
)

print(
    f"MSE:  {mse:.10f}"
)

print(
    f"RMSE: {rmse:.10f}"
)

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
print(f"AUC (volatility score): {auc_volatility:.10f}")
print(f"AUC (probability score): {auc_probability:.10f}")

print(
    f"\nMean predicted spike probability: "
    f"{metrics['mean_probability']:.6f}"
)

print(
    f"Predicted spikes: {predicted.sum():,}"
)

print(
    f"Actual spikes: {actual.sum():,}"
)

print("\nRESULT FILES SAVED")
print("=" * 90)
print(f"\nForecasts:\n{OUTPUT_FILE}")
print(f"\nMetrics:\n{METRICS_FILE}")
print(f"\nModel state:\n{STATE_FILE}")

print("\nFinal BVAR(8)-log1p evaluation completed.")


