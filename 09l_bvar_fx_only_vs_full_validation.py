#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 19:12:49 2026

@author: setup
"""

# ============================================================
# STEP 09L: BVAR(8) MULTIVARIATE ABLATION VALIDATION
# ============================================================
#
# Compare:
#
#   1. BVAR(8)-FX-only
#      fx_log_return
#      fx_return_squared
#
#   2. BVAR(8)-Full
#      fx_log_return
#      fx_return_squared
#      brent_log_return
#      reserves_log_change
#
# The final 2021-2025 test set is NOT used.
# Model selection is based only on the 2006-2020 development
# period, with the final 20% used for validation.
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

OUTPUT_FILE = (
    "results/"
    "bvar8_fx_only_vs_full_validation.csv"
)

FORECAST_OUTPUT_FILE = (
    "results/"
    "bvar8_fx_only_vs_full_validation_forecasts.csv"
)

FULL_TRAIN_END = "2020-12-31"

SPIKE_THRESHOLD = 0.10927724

LAG_ORDER = 8

LAMBDA = 0.20

INTERCEPT_PRIOR_VARIANCE = 100.0

PRIOR_DF = 6

N_DRAWS = 3000

RANDOM_SEED = 42

VALIDATION_FRACTION = 0.20

FX_ONLY_VARIABLES = [
    "fx_log_return",
    "fx_return_squared",
]

FULL_VARIABLES = [
    "fx_log_return",
    "fx_return_squared",
    "brent_log_return",
    "reserves_log_change",
]


# ============================================================
# 2. SETUP
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

np.set_printoptions(
    suppress=True
)


# ============================================================
# 3. LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["date", "target_date"]
)

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)

print("=" * 90)
print("STEP 09L: BVAR(8) MULTIVARIATE ABLATION VALIDATION")
print("=" * 90)

print(
    f"\nRows loaded: {len(df):,}"
)


# ============================================================
# 4. DEVELOPMENT / VALIDATION SPLIT
# ============================================================

full_train = (
    df[df["date"] <= FULL_TRAIN_END]
    .copy()
    .sort_values("date")
    .reset_index(drop=True)
)

split_position = int(
    np.floor(
        len(full_train)
        * (1 - VALIDATION_FRACTION)
    )
)

validation_start = (
    full_train.loc[
        split_position,
        "date"
    ]
)

model_train = full_train[
    full_train["date"] < validation_start
].copy()

validation = full_train[
    (full_train["date"] >= validation_start)
    & full_train["target_date"].notna()
    & full_train["target_volatility"].notna()
    & full_train["target_spike"].notna()
].copy()

if validation.empty:
    raise ValueError(
        "No valid validation observations found."
    )

print(
    f"\nModel-fit period: "
    f"{model_train['date'].min().date()} "
    f"to "
    f"{model_train['date'].max().date()}"
)

print(
    f"Validation period: "
    f"{validation['date'].min().date()} "
    f"to "
    f"{validation['date'].max().date()}"
)

print(
    f"Model-fit observations: "
    f"{len(model_train):,}"
)

print(
    f"Validation observations: "
    f"{len(validation):,}"
)


# ============================================================
# 5. DESIGN MATRIX
# ============================================================

def build_design_matrix(
    data,
    variables,
    means,
    stds,
    lag_order,
):
    """
    Construct a gap-safe VAR(p) design matrix.

    Each segment is handled independently, so lags never cross
    a long-gap boundary.
    """

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
            dtype=float
        )

        scaled = (
            values - means
        ) / stds

        if len(scaled) <= lag_order:
            continue

        for position in range(
            lag_order,
            len(scaled)
        ):

            x = [1.0]

            for lag in range(
                1,
                lag_order + 1
            ):

                x.extend(
                    scaled[position - lag]
                )

            X_rows.append(x)
            Y_rows.append(
                scaled[position]
            )

    return (
        np.asarray(X_rows, dtype=float),
        np.asarray(Y_rows, dtype=float),
    )


# ============================================================
# 6. FIT BVAR
# ============================================================

def fit_bvar(
    data,
    variables,
    lag_order,
    shrinkage_lambda,
    intercept_variance,
    prior_df,
    n_draws,
    seed,
):
    """
    Conjugate Bayesian VAR with:
      - diagonal coefficient shrinkage prior
      - inverse-Wishart covariance prior
      - standardized variables
    """

    values = data[
        variables
    ].to_numpy(
        dtype=float
    )

    if np.isnan(values).any():
        raise ValueError(
            "Missing values in BVAR model variables."
        )

    means = values.mean(
        axis=0
    )

    stds = values.std(
        axis=0,
        ddof=0
    )

    stds = np.where(
        stds <= 0,
        1.0,
        stds
    )

    X, Y = build_design_matrix(
        data,
        variables,
        means,
        stds,
        lag_order,
    )

    if len(X) == 0:
        raise ValueError(
            "No usable BVAR observations."
        )

    T = len(Y)
    K = len(variables)
    M = X.shape[1]

    # --------------------------------------------------------
    # Prior mean
    # --------------------------------------------------------

    B0 = np.zeros(
        (M, K)
    )

    # --------------------------------------------------------
    # Prior variance
    # --------------------------------------------------------

    V0_diag = np.empty(
        M
    )

    V0_diag[0] = (
        intercept_variance
    )

    index = 1

    for lag in range(
        1,
        lag_order + 1
    ):

        coefficient_variance = (
            shrinkage_lambda
            / lag
        ) ** 2

        V0_diag[
            index:
            index + K
        ] = coefficient_variance

        index += K

    V0_inv = np.diag(
        1.0 / V0_diag
    )

    # --------------------------------------------------------
    # Covariance prior
    # --------------------------------------------------------

    S0 = np.eye(K)

    # --------------------------------------------------------
    # Posterior
    # --------------------------------------------------------

    XtX = (
        X.T @ X
    )

    Vn_inv = (
        V0_inv
        + XtX
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

    nu_n = (
        prior_df
        + T
    )

    # --------------------------------------------------------
    # Posterior draws
    # --------------------------------------------------------

    rng = np.random.default_rng(
        seed
    )

    posterior_B = np.empty(
        (
            n_draws,
            M,
            K,
        )
    )

    posterior_Sigma = np.empty(
        (
            n_draws,
            K,
            K,
        )
    )

    chol_V = np.linalg.cholesky(
        Vn
    )

    for draw in range(
        n_draws
    ):

        sigma_draw = invwishart.rvs(
            df=nu_n,
            scale=Sn,
            random_state=rng,
        )

        sigma_draw = np.asarray(
            sigma_draw,
            dtype=float
        )

        sigma_draw = (
            sigma_draw
            + sigma_draw.T
        ) / 2

        posterior_Sigma[
            draw
        ] = sigma_draw

        chol_sigma = np.linalg.cholesky(
            sigma_draw
        )

        Z = rng.standard_normal(
            (M, K)
        )

        posterior_B[
            draw
        ] = (
            Bn
            + chol_V
            @ Z
            @ chol_sigma.T
        )

    return {
        "posterior_B": posterior_B,
        "posterior_Sigma": posterior_Sigma,
        "means": means,
        "stds": stds,
        "lag_order": lag_order,
        "variables": variables,
        "target_index": variables.index(
            "fx_return_squared"
        ),
        "T": T,
        "M": M,
    }


# ============================================================
# 7. FORECAST ONE ORIGIN
# ============================================================

def forecast_origin(
    origin_row,
    full_data,
    fitted,
):
    variables = fitted["variables"]

    lag_order = fitted["lag_order"]

    K = len(variables)

    target_index = fitted[
        "target_index"
    ]

    origin_date = origin_row[
        "date"
    ]

    segment_id = origin_row[
        "segment_id"
    ]

    segment = full_data[
        (
            full_data["segment_id"]
            == segment_id
        )
        &
        (
            full_data["date"]
            <= origin_date
        )
    ].copy()

    segment = (
        segment
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(segment) <= lag_order:
        return None

    history = segment[
        variables
    ].to_numpy(
        dtype=float
    )

    history_scaled = (
        history
        - fitted["means"]
    ) / fitted["stds"]

    recent = history_scaled[
        -lag_order - 1:
    ]

    if len(recent) != lag_order + 1:
        return None

    x = [1.0]

    for lag in range(
        1,
        lag_order + 1
    ):

        x.extend(
            recent[-lag]
        )

    x = np.asarray(
        x,
        dtype=float
    )

    posterior_B = fitted[
        "posterior_B"
    ]

    posterior_Sigma = fitted[
        "posterior_Sigma"
    ]

    mu_draws = np.einsum(
        "m,nm->n",
        x,
        posterior_B[
            :,
            :,
            target_index
        ],
    )

    var_draws = (
        posterior_Sigma[
            :,
            target_index,
            target_index
        ]
    )

    var_draws = np.maximum(
        var_draws,
        1e-12
    )

    sd_draws = np.sqrt(
        var_draws
    )

    # --------------------------------------------------------
    # Continuous forecast
    # --------------------------------------------------------

    predicted_volatility = float(
        (
            mu_draws.mean()
            * fitted["stds"][target_index]
        )
        + fitted["means"][target_index]
    )

    predicted_volatility = max(
        0.0,
        predicted_volatility
    )

    # --------------------------------------------------------
    # Spike probability
    # --------------------------------------------------------

    threshold_scaled = (
        SPIKE_THRESHOLD
        - fitted["means"][target_index]
    ) / fitted["stds"][target_index]

    z = (
        threshold_scaled
        - mu_draws
    ) / sd_draws

    probability = float(
        np.mean(
            1.0 - norm.cdf(z)
        )
    )

    return {
        "origin_date": origin_date,
        "target_date": origin_row[
            "target_date"
        ],
        "actual_volatility": float(
            origin_row[
                "target_volatility"
            ]
        ),
        "actual_spike": int(
            origin_row[
                "target_spike"
            ]
        ),
        "forecast_volatility": (
            predicted_volatility
        ),
        "predicted_spike_probability": (
            probability
        ),
        "predicted_spike": int(
            probability >= 0.50
        ),
    }


# ============================================================
# 8. RUN SPECIFICATION
# ============================================================

def run_specification(
    name,
    variables,
    seed,
):
    print("\n" + "=" * 90)
    print(f"FITTING {name}")
    print("=" * 90)

    fitted = fit_bvar(
        model_train,
        variables,
        LAG_ORDER,
        LAMBDA,
        INTERCEPT_PRIOR_VARIANCE,
        PRIOR_DF,
        N_DRAWS,
        seed,
    )

    print(
        f"\nVariables: {', '.join(variables)}"
    )

    print(
        f"Effective observations: "
        f"{fitted['T']:,}"
    )

    print(
        f"Regressors per equation: "
        f"{fitted['M']}"
    )

    print(
        f"Posterior draws: "
        f"{N_DRAWS:,}"
    )

    rows = []

    for _, row in validation.iterrows():

        result = forecast_origin(
            row,
            df,
            fitted,
        )

        if result is not None:
            rows.append(result)

    forecasts = pd.DataFrame(
        rows
    )

    if forecasts.empty:
        raise ValueError(
            f"No forecasts produced for {name}."
        )

    error = (
        forecasts["actual_volatility"]
        - forecasts["forecast_volatility"]
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

    actual = forecasts[
        "actual_spike"
    ]

    predicted = forecasts[
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
        2 * precision * recall
        / (precision + recall)
        if (
            pd.notna(precision)
            and pd.notna(recall)
            and (
                precision + recall
            ) > 0
        )
        else np.nan
    )

    auc = roc_auc_score(
        actual,
        forecasts[
            "forecast_volatility"
        ],
    )

    predicted_spikes = int(
        predicted.sum()
    )

    actual_spikes = int(
        actual.sum()
    )

    print(
        f"\nValidation forecasts: "
        f"{len(forecasts):,}"
    )

    print(
        f"MAE:  {mae:.8f}"
    )

    print(
        f"RMSE: {rmse:.8f}"
    )

    print(
        f"Accuracy:  {accuracy:.8f}"
    )

    print(
        f"Precision: {precision:.8f}"
    )

    print(
        f"Recall:    {recall:.8f}"
    )

    print(
        f"F1:        {f1:.8f}"
    )

    print(
        f"ROC-AUC:   {auc:.8f}"
    )

    print(
        f"Actual spikes: "
        f"{actual_spikes}"
    )

    print(
        f"Predicted spikes: "
        f"{predicted_spikes}"
    )

    metrics = {
        "model": name,
        "validation_observations": len(
            forecasts
        ),
        "variables": ", ".join(
            variables
        ),
        "n_variables": len(
            variables
        ),
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
        "AUC": auc,
        "actual_spikes": actual_spikes,
        "predicted_spikes": predicted_spikes,
        "lag_order": LAG_ORDER,
        "lambda": LAMBDA,
        "probability_cutoff": 0.50,
        "spike_threshold": SPIKE_THRESHOLD,
    }

    forecasts["model"] = name

    return (
        metrics,
        forecasts,
    )


# ============================================================
# 9. RUN BOTH MODELS
# ============================================================

fx_metrics, fx_forecasts = run_specification(
    "BVAR(8)-FX-only",
    FX_ONLY_VARIABLES,
    RANDOM_SEED,
)

full_metrics, full_forecasts = run_specification(
    "BVAR(8)-Full",
    FULL_VARIABLES,
    RANDOM_SEED + 1,
)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

metrics_df = pd.DataFrame(
    [
        fx_metrics,
        full_metrics,
    ]
)

forecast_df = pd.concat(
    [
        fx_forecasts,
        full_forecasts,
    ],
    ignore_index=True,
)

metrics_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

forecast_df.to_csv(
    FORECAST_OUTPUT_FILE,
    index=False,
)


# ============================================================
# 11. SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("09L VALIDATION SUMMARY")
print("=" * 90)

print()

print(
    metrics_df[
        [
            "model",
            "n_variables",
            "validation_observations",
            "MAE",
            "RMSE",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "AUC",
            "actual_spikes",
            "predicted_spikes",
        ]
    ].to_string(
        index=False
    )
)

print(
    f"\nMetrics saved:\n{OUTPUT_FILE}"
)

print(
    f"\nForecasts saved:\n"
    f"{FORECAST_OUTPUT_FILE}"
)

print(
    "\nThe final 2021-2025 test set was not used "
    "for this specification comparison."
)
