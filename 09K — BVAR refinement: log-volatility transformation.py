# ============================================================
# STEP 09K: BVAR(8) RAW vs LOG1P VOLATILITY VALIDATION
# ============================================================
#
# Purpose:
#   Test ONE controlled specification change before touching the
#   final 2021-2025 test set:
#
#   Model A: BVAR(8) with raw fx_return_squared
#   Model B: BVAR(8) with log1p(fx_return_squared)
#
# The validation period is the final 20% of the 2006-2020
# training period. The 2021-2025 test set is NOT used here.
#
# IMPORTANT:
#   The spike threshold remains frozen at 0.10927724.
#   We do NOT tune the threshold or probability cutoff.
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
    "bvar8_raw_vs_log1p_validation.csv"
)

FORECAST_OUTPUT_FILE = (
    "results/"
    "bvar8_raw_vs_log1p_validation_forecasts.csv"
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

VARIABLES = [
    "fx_log_return",
    "fx_return_squared",
    "brent_log_return",
    "reserves_log_change",
]

TARGET_INDEX = 1

K = len(VARIABLES)


# ============================================================
# 2. CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
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
print("STEP 09K: BVAR(8) RAW vs LOG1P VOLATILITY VALIDATION")
print("=" * 90)

print(
    f"\nRows loaded: {len(df):,}"
)


# ============================================================
# 4. DEFINE TRAINING / VALIDATION PERIOD
# ============================================================
#
# We split ONLY the original training period.
# The final 2021-2025 test period is untouched.
# ============================================================

full_train = (
    df[df["date"] <= FULL_TRAIN_END]
    .copy()
    .sort_values("date")
    .reset_index(drop=True)
)

if full_train.empty:
    raise ValueError("No observations found in the training period.")

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
        "Validation set contains no rows with valid next-observation targets."
    )

print(
    f"\nFull training period: "
    f"{full_train['date'].min().date()} "
    f"to "
    f"{full_train['date'].max().date()}"
)

print(
    f"Model-fit period: "
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
    f"Validation observations with valid next-observation targets: "
    f"{len(validation):,}"
)


# ============================================================
# 5. TRANSFORMATION FUNCTION
# ============================================================

def build_model_data(data, use_log1p):
    """
    Create the four BVAR variables.

    Only the volatility variable changes:

        raw model:
            fx_return_squared

        log1p model:
            log(1 + fx_return_squared)

    log1p is used instead of log(x + epsilon), so zero returns
    remain exactly well-defined.
    """

    out = data[
        [
            "date",
            "target_date",
            "segment_id",
            *VARIABLES,
            "target_volatility",
            "target_spike",
        ]
    ].copy()

    if use_log1p:

        out["fx_return_squared_model"] = (
            np.log1p(
                out["fx_return_squared"]
            )
        )

    else:

        out["fx_return_squared_model"] = (
            out["fx_return_squared"]
        )

    out["target_model"] = (
        out["fx_return_squared_model"]
    )

    model_variables = [
        "fx_log_return",
        "fx_return_squared_model",
        "brent_log_return",
        "reserves_log_change",
    ]

    return out, model_variables


# ============================================================
# 6. BUILD GAP-SAFE BVAR DESIGN MATRIX
# ============================================================

def build_design_matrix(
    data,
    model_variables,
    lag_order,
    means,
    stds,
):
    """
    Build a VAR(p) design matrix without crossing segment
    boundaries.

    X:
        intercept + lag 1 ... lag p

    Y:
        current four-variable vector
    """

    X_rows = []
    Y_rows = []

    for _, segment in data.groupby(
        "segment_id",
        sort=True
    ):

        segment = (
            segment
            .sort_values("date")
            .reset_index(drop=True)
        )

        values = segment[
            model_variables
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
# 7. FIT BVAR(8)
# ============================================================

def fit_bvar(
    data,
    model_variables,
    lag_order,
    shrinkage_lambda,
    intercept_variance,
    prior_df,
    n_draws,
    seed,
):
    """
    Conjugate Bayesian VAR using a matrix-normal /
    inverse-Wishart posterior with diagonal coefficient
    shrinkage prior.

    Data are standardized using model-fit-period statistics.
    """

    values = data[
        model_variables
    ].to_numpy(dtype=float)

    if np.isnan(values).any():
        raise ValueError(
            "Missing values found in BVAR model variables."
        )

    means = values.mean(axis=0)
    stds = values.std(axis=0, ddof=0)

    stds = np.where(
        stds <= 0,
        1.0,
        stds
    )

    X, Y = build_design_matrix(
        data,
        model_variables,
        lag_order,
        means,
        stds,
    )

    if len(X) == 0:
        raise ValueError(
            "No usable BVAR design rows were created."
        )

    T = len(Y)
    M = X.shape[1]

    # --------------------------------------------------------
    # Prior
    # --------------------------------------------------------

    B0 = np.zeros(
        (M, K)
    )

    V0_diag = np.empty(M)

    # Intercept
    V0_diag[0] = intercept_variance

    # Lag coefficients:
    # variance shrinks as lag length increases.
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

    # Weak inverse-Wishart prior.
    S0 = np.eye(K)

    # --------------------------------------------------------
    # Posterior
    # --------------------------------------------------------

    XtX = X.T @ X

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

        chol_Sigma = np.linalg.cholesky(
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
            @ chol_Sigma.T
        )

    return {
        "posterior_B": posterior_B,
        "posterior_Sigma": posterior_Sigma,
        "means": means,
        "stds": stds,
        "lag_order": lag_order,
        "T": T,
        "M": M,
        "nu_n": nu_n,
    }


# ============================================================
# 8. FORECAST ONE VALIDATION ORIGIN
# ============================================================

def forecast_origin(
    origin_row,
    full_data,
    fitted,
    model_variables,
    use_log1p,
):
    """
    Forecast the next observation from one validation origin,
    using only the current continuous segment.
    """

    origin_date = origin_row["date"]
    segment_id = origin_row["segment_id"]

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

    if len(segment) <= fitted["lag_order"]:
        return None

    history = segment[
        model_variables
    ].to_numpy(
        dtype=float,
        copy=True,
    )

    if use_log1p:
        history[:, 1] = np.log1p(
            history[:, 1]
        )

    means = fitted["means"]
    stds = fitted["stds"]

    history_scaled = (
        history - means
    ) / stds

    # Need lags 1...p
    lag_history = history_scaled[
        -fitted["lag_order"] - 1:
    ]

    if len(lag_history) != (
        fitted["lag_order"] + 1
    ):
        return None

    x = [1.0]

    for lag in range(
        1,
        fitted["lag_order"] + 1
    ):

        x.extend(
            lag_history[-lag]
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

    # Conditional mean for target equation.
    mu_draws = np.einsum(
        "m,nm->n",
        x,
        posterior_B[
            :,
            :,
            TARGET_INDEX
        ]
    )

    # Conditional variance for target equation.
    var_draws = (
        posterior_Sigma[
            :,
            TARGET_INDEX,
            TARGET_INDEX
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

    if use_log1p:

        predicted_volatility = float(
            np.mean(
                np.expm1(
                    mu_draws
                    + 0.5 * var_draws
                )
            )
        )

    else:

        predicted_volatility = float(
            np.mean(mu_draws)
            * stds[TARGET_INDEX]
            + means[TARGET_INDEX]
        )

        predicted_volatility = max(
            0.0,
            predicted_volatility
        )

    # --------------------------------------------------------
    # Spike probability
    # --------------------------------------------------------
    #
    # For log1p model:
    #
    # log(1 + v) > log(1 + threshold)
    #
    # For raw model:
    #
    # v > threshold
    # --------------------------------------------------------

    if use_log1p:

        threshold_model = np.log1p(
            SPIKE_THRESHOLD
        )

    else:

        threshold_model = (
            SPIKE_THRESHOLD
        )

    threshold_scaled = (
        threshold_model
        - means[TARGET_INDEX]
    ) / stds[TARGET_INDEX]

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
        "target_date": origin_row["target_date"],
        "actual_volatility": float(
            origin_row["target_volatility"]
        ),
        "actual_spike": int(
            origin_row["target_spike"]
        ),
        "forecast_volatility": predicted_volatility,
        "predicted_spike_probability": probability,
        "predicted_spike": int(
            probability >= 0.50
        ),
    }


# ============================================================
# 9. RUN ONE MODEL
# ============================================================

def run_validation(
    name,
    use_log1p,
    seed,
):
    print("\n" + "=" * 90)
    print(f"FITTING {name}")
    print("=" * 90)

    model_data, model_variables = (
        build_model_data(
            model_train,
            use_log1p
        )
    )

    fitted = fit_bvar(
        model_data,
        model_variables,
        LAG_ORDER,
        LAMBDA,
        INTERCEPT_PRIOR_VARIANCE,
        PRIOR_DF,
        N_DRAWS,
        seed,
    )

    print(
        f"\nEffective BVAR observations: "
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

    # Transform the FULL dataset in the same way as the fitted
    # model so validation origins can retrieve correctly transformed
    # within-segment histories without changing any future outcomes.
    full_model_data, full_model_variables = (
        build_model_data(
            df,
            use_log1p
        )
    )

    for _, row in validation.iterrows():

        result = forecast_origin(
            row,
            full_model_data,
            fitted,
            full_model_variables,
            use_log1p,
        )

        if result is not None:
            rows.append(result)

    forecasts = pd.DataFrame(rows)

    if forecasts.empty:
        raise ValueError(
            f"No validation forecasts generated for {name}."
        )

    # Continuous metrics
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

    # Classification metrics
    actual = forecasts["actual_spike"]
    predicted = forecasts["predicted_spike"]

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

    total = tn + fp + fn + tp

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
            and (precision + recall) > 0
        )
        else np.nan
    )

    auc = roc_auc_score(
        actual,
        forecasts["forecast_volatility"]
    )

    forecasts["model"] = name

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

    metrics = {
        "model": name,
        "validation_observations": len(forecasts),
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
        "lag_order": LAG_ORDER,
        "lambda": LAMBDA,
        "probability_cutoff": 0.50,
        "spike_threshold": SPIKE_THRESHOLD,
    }

    return (
        metrics,
        forecasts,
    )


# ============================================================
# 10. RUN RAW MODEL
# ============================================================

raw_metrics, raw_forecasts = run_validation(
    "BVAR(8)-Raw",
    use_log1p=False,
    seed=RANDOM_SEED,
)


# ============================================================
# 11. RUN LOG1P MODEL
# ============================================================

log_metrics, log_forecasts = run_validation(
    "BVAR(8)-Log1p",
    use_log1p=True,
    seed=RANDOM_SEED + 1,
)


# ============================================================
# 12. COMBINE RESULTS
# ============================================================

metrics_df = pd.DataFrame([
    raw_metrics,
    log_metrics,
])

forecast_df = pd.concat(
    [
        raw_forecasts,
        log_forecasts,
    ],
    ignore_index=True,
)


# ============================================================
# 13. SAVE RESULTS
# ============================================================

metrics_df.to_csv(
    OUTPUT_FILE,
    index=False
)

forecast_df.to_csv(
    FORECAST_OUTPUT_FILE,
    index=False
)


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("09K VALIDATION SUMMARY")
print("=" * 90)

print("\n")
print(
    metrics_df[
        [
            "model",
            "validation_observations",
            "MAE",
            "RMSE",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "AUC",
        ]
    ].to_string(index=False)
)

print(
    f"\nMetrics saved:\n{OUTPUT_FILE}"
)

print(
    f"\nForecasts saved:\n{FORECAST_OUTPUT_FILE}"
)

print(
    "\nIMPORTANT: this validation does not use the 2021-2025 "
    "test set for model selection."
)
