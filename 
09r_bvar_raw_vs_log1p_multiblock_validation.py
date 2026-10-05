# ============================================================
# STEP 09R: BVAR RAW TARGET VS LOG1P TARGET VALIDATION
# ============================================================
#
# PURPOSE
# -------
# Test one final, pre-specified alternative to the raw BVAR target:
#
#   Model A: raw target = fx_return_squared
#   Model B: transformed target = log1p(fx_return_squared)
#
# The purpose is NOT to tune the final test set. The 2021-2025
# test period remains completely untouched.
#
# We use the same chronological validation blocks as Step 09N:
#   Block 1: 2009-01-01 to 2011-12-31
#   Block 2: 2014-01-01 to 2016-12-31
#   Block 3: 2017-01-01 to 2020-12-31
#
# The BVAR remains:
#   - VAR(8)
#   - four-variable multivariate system
#   - lambda = 0.10
#   - intercept prior variance = 100
#   - prior df = 6
#   - 3,000 posterior draws
#
# IMPORTANT
# ---------
# Only the FX squared-return variable is transformed. The other
# endogenous variables remain on their original scales before
# standardization.
#
# The spike definition remains on the ORIGINAL squared-return scale:
#
#   threshold = 0.10927724
#
# For the log1p model, the threshold is transformed consistently:
#
#   log1p(0.10927724)
#
# The transformed forecast is converted back to the original scale
# using the posterior predictive mean for a Normal predictive draw:
#
#   E[v] = exp(mu + 0.5 * variance) - 1
#
# This makes the continuous forecast directly comparable with the raw
# target forecast.
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

METRICS_OUTPUT_FILE = (
    "results/"
    "bvar8_raw_vs_log1p_multiblock_validation_metrics.csv"
)

FORECAST_OUTPUT_FILE = (
    "results/"
    "bvar8_raw_vs_log1p_multiblock_validation_forecasts.csv"
)

SPIKE_THRESHOLD = 0.10927724

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
# 2. VALIDATION BLOCKS
# ============================================================

VALIDATION_BLOCKS = [
    {
        "block": "Block_1",
        "start": "2009-01-01",
        "end": "2011-12-31",
    },
    {
        "block": "Block_2",
        "start": "2014-01-01",
        "end": "2016-12-31",
    },
    {
        "block": "Block_3",
        "start": "2017-01-01",
        "end": "2020-12-31",
    },
]


# ============================================================
# 3. SETUP
# ============================================================

os.makedirs(
    "results",
    exist_ok=True,
)


# ============================================================
# 4. LOAD DATA
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

print("=" * 90)
print("STEP 09R: BVAR RAW TARGET VS LOG1P TARGET VALIDATION")
print("=" * 90)

print(
    f"\nRows loaded: {len(df):,}"
)

print(
    "\nFinal test period remains untouched: 2021-2025"
)

print(
    f"Spike threshold (raw scale): {SPIKE_THRESHOLD:.8f}"
)

print(
    f"Spike threshold (log1p scale): "
    f"{np.log1p(SPIKE_THRESHOLD):.8f}"
)


# ============================================================
# 5. PREPARE MODEL DATA
# ============================================================

def prepare_model_data(data, model_type):
    """
    Return a copy of the data used by the BVAR.

    raw:
        fx_return_squared remains unchanged.

    log1p:
        only fx_return_squared is transformed to log1p.
    """

    model_data = data.copy()

    if model_type == "raw":
        return model_data

    if model_type == "log1p":
        if (model_data[TARGET_VARIABLE] < 0).any():
            raise ValueError(
                "fx_return_squared contains negative values; "
                "log1p transformation is invalid."
            )

        model_data[TARGET_VARIABLE] = np.log1p(
            model_data[TARGET_VARIABLE]
        )

        return model_data

    raise ValueError(
        f"Unknown model_type: {model_type}"
    )


# ============================================================
# 6. DESIGN MATRIX
# ============================================================

def build_design_matrix(
    data,
    variables,
    means,
    stds,
    lag_order,
):
    """
    Build a gap-safe VAR(p) design matrix.

    Lags are constructed separately inside each continuous segment.
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
# 7. FIT BVAR
# ============================================================

def fit_bvar(
    data,
    model_type,
    variables,
    lag_order,
    shrinkage_lambda,
    intercept_variance,
    prior_df,
    n_draws,
    seed,
):
    """
    Fit the conjugate Bayesian VAR used in the earlier BVAR steps.
    """

    values = data[
        variables
    ].to_numpy(
        dtype=float
    )

    if np.isnan(values).any():
        raise ValueError(
            "Missing values found in BVAR variables."
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
    model_K = len(variables)
    M = X.shape[1]

    # --------------------------------------------------------
    # Prior mean
    # --------------------------------------------------------

    B0 = np.zeros(
        (
            M,
            model_K,
        )
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
        lag_order + 1,
    ):

        coefficient_variance = (
            shrinkage_lambda / lag
        ) ** 2

        V0_diag[
            index:
            index + model_K
        ] = coefficient_variance

        index += model_K

    V0_inv = np.diag(
        1.0 / V0_diag
    )

    # --------------------------------------------------------
    # Covariance prior
    # --------------------------------------------------------

    S0 = np.eye(
        model_K
    )

    # --------------------------------------------------------
    # Posterior
    # --------------------------------------------------------

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
        + Bn.T @ V0_inv @ Bn
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
            model_K,
        )
    )

    posterior_Sigma = np.empty(
        (
            n_draws,
            model_K,
            model_K,
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
                model_K,
            )
        )

        posterior_B[draw] = (
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
            TARGET_VARIABLE
        ),
        "model_type": model_type,
        "T": T,
        "M": M,
    }


# ============================================================
# 8. FORECAST ONE ORIGIN
# ============================================================

def forecast_origin(
    origin_row,
    full_data,
    fitted,
):
    """
    Forecast target t+1 from one origin using only the current
    continuous segment.
    """

    variables = fitted[
        "variables"
    ]

    lag_order = fitted[
        "lag_order"
    ]

    target_index = fitted[
        "target_index"
    ]

    model_type = fitted[
        "model_type"
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
        & (
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
        dtype=float,
        copy=True,
    )

    if model_type == "log1p":
        history[:, target_index] = np.log1p(
            history[:, target_index]
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
        lag_order + 1,
    ):
        x.extend(
            recent[-lag]
        )

    x = np.asarray(
        x,
        dtype=float,
    )

    posterior_B = fitted[
        "posterior_B"
    ]

    posterior_Sigma = fitted[
        "posterior_Sigma"
    ]

    # Conditional target means on the model's standardized scale.
    mu_draws_scaled = np.einsum(
        "m,nm->n",
        x,
        posterior_B[
            :,
            :,
            target_index,
        ],
    )

    # Conditional target variances on the model's standardized scale.
    var_draws_scaled = np.maximum(
        posterior_Sigma[
            :,
            target_index,
            target_index,
        ],
        1e-12,
    )

    sd_draws_scaled = np.sqrt(
        var_draws_scaled
    )

    # Convert predictive mean/variance back to the model target scale.
    mu_draws = (
        mu_draws_scaled
        * fitted["stds"][target_index]
        + fitted["means"][target_index]
    )

    var_draws = (
        var_draws_scaled
        * (
            fitted["stds"][target_index]
            ** 2
        )
    )

    sd_draws = np.sqrt(
        np.maximum(
            var_draws,
            1e-12,
        )
    )

    if model_type == "raw":

        predicted_volatility = float(
            mu_draws.mean()
        )

        predicted_volatility = max(
            0.0,
            predicted_volatility,
        )

        threshold_model_scale = (
            SPIKE_THRESHOLD
        )

    elif model_type == "log1p":

        # For Z ~ Normal(mu, variance):
        # E[exp(Z)-1] = exp(mu + 0.5*variance)-1.
        predicted_draw_means_original = (
            np.exp(
                np.clip(
                    mu_draws + 0.5 * var_draws,
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

        threshold_model_scale = np.log1p(
            SPIKE_THRESHOLD
        )

    else:
        raise ValueError(
            f"Unknown model_type: {model_type}"
        )

    # --------------------------------------------------------
    # Probability of exceeding the spike threshold
    # --------------------------------------------------------
    #
    # Raw model:
    #   P(v > threshold)
    #
    # Log1p model:
    #   P(log1p(v) > log1p(threshold))
    #
    # Both are equivalent exceedance events on the original
    # squared-return scale.
    # --------------------------------------------------------

    threshold_scaled = (
        threshold_model_scale
        - fitted["means"][target_index]
    ) / fitted["stds"][target_index]

    z = (
        threshold_scaled
        - mu_draws_scaled
    ) / sd_draws_scaled

    probability = float(
        np.mean(
            1.0 - norm.cdf(z)
        )
    )

    return {
        "block": origin_row["validation_block"],
        "model_type": model_type,
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
        "forecast_volatility": predicted_volatility,
        "predicted_spike_probability": probability,
        "predicted_spike": int(
            probability >= 0.50
        ),
    }


# ============================================================
# 9. METRICS
# ============================================================

def calculate_metrics(
    forecasts,
    model_type,
):

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
            and (precision + recall) > 0
        )
        else np.nan
    )

    if actual.nunique() >= 2:
        auc_volatility = roc_auc_score(
            actual,
            forecasts[
                "forecast_volatility"
            ],
        )

        auc_probability = roc_auc_score(
            actual,
            forecasts[
                "predicted_spike_probability"
            ],
        )
    else:
        auc_volatility = np.nan
        auc_probability = np.nan

    return {
        "model_type": model_type,
        "validation_observations": len(
            forecasts
        ),
        "actual_spikes": int(
            actual.sum()
        ),
        "predicted_spikes": int(
            predicted.sum()
        ),
        "actual_spike_rate": float(
            actual.mean()
        ),
        "predicted_spike_rate": float(
            predicted.mean()
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
        "AUC_volatility": auc_volatility,
        "AUC_probability": auc_probability,
        "mean_probability": float(
            forecasts[
                "predicted_spike_probability"
            ].mean()
        ),
        "probability_std": float(
            forecasts[
                "predicted_spike_probability"
            ].std()
        ),
    }


# ============================================================
# 10. MAIN VALIDATION LOOP
# ============================================================

MODEL_TYPES = [
    "raw",
    "log1p",
]

all_forecasts = []
all_metrics = []

run_number = 0

total_runs = (
    len(VALIDATION_BLOCKS)
    * len(MODEL_TYPES)
)


for block_info in VALIDATION_BLOCKS:

    block_name = block_info[
        "block"
    ]

    block_start = pd.Timestamp(
        block_info["start"]
    )

    block_end = pd.Timestamp(
        block_info["end"]
    )

    model_train_raw = df[
        df["date"] < block_start
    ].copy()

    validation = df[
        (
            df["date"] >= block_start
        )
        & (
            df["date"] <= block_end
        )
        & df["target_date"].notna()
        & df["target_volatility"].notna()
        & df["target_spike"].notna()
    ].copy()

    if model_train_raw.empty:
        print(
            f"\nSkipping {block_name}: no model-training observations."
        )
        continue

    if validation.empty:
        print(
            f"\nSkipping {block_name}: no valid validation observations."
        )
        continue

    print("\n" + "=" * 90)
    print(
        f"{block_name}: "
        f"{block_start.date()} to {block_end.date()}"
    )
    print("=" * 90)

    print(
        f"\nModel training observations: {len(model_train_raw):,}"
    )

    print(
        f"Validation observations: {len(validation):,}"
    )

    print(
        f"Actual validation spikes: "
        f"{int(validation['target_spike'].sum()):,}"
    )

    for model_type in MODEL_TYPES:

        run_number += 1

        print(
            "\n"
            f"[{run_number}/{total_runs}] "
            f"{block_name} | model = {model_type}"
        )

        model_train = prepare_model_data(
            model_train_raw,
            model_type,
        )

        fitted = fit_bvar(
            model_train,
            model_type,
            VARIABLES,
            LAG_ORDER,
            LAMBDA,
            INTERCEPT_PRIOR_VARIANCE,
            PRIOR_DF,
            N_DRAWS,
            RANDOM_SEED + run_number,
        )

        rows = []

        validation_local = validation.copy()
        validation_local[
            "validation_block"
        ] = block_name

        # The forecast_origin function needs the transformed values
        # for the model history, so pass the model-prepared full data.
        full_model_data = prepare_model_data(
            df,
            model_type,
        )

        for _, row in validation_local.iterrows():

            result = forecast_origin(
                row,
                full_model_data,
                fitted,
            )

            if result is not None:
                rows.append(result)

        forecasts = pd.DataFrame(
            rows
        )

        if forecasts.empty:
            print(
                "No forecasts produced."
            )
            continue

        metrics = calculate_metrics(
            forecasts,
            model_type,
        )

        metrics["block"] = block_name
        metrics["block_start"] = block_start.date()
        metrics["block_end"] = block_end.date()

        all_forecasts.append(
            forecasts
        )

        all_metrics.append(
            metrics
        )

        print(
            f"Forecasts: {len(forecasts):,} | "
            f"MAE: {metrics['MAE']:.4f} | "
            f"RMSE: {metrics['RMSE']:.4f} | "
            f"AUC(vol): {metrics['AUC_volatility']:.4f} | "
            f"AUC(prob): {metrics['AUC_probability']:.4f} | "
            f"Predicted spikes: {metrics['predicted_spikes']:,}"
        )


# ============================================================
# 11. COMBINE RESULTS
# ============================================================

if not all_metrics:
    raise ValueError(
        "No validation results were generated."
    )

metrics_df = pd.DataFrame(
    all_metrics
)

forecasts_df = pd.concat(
    all_forecasts,
    ignore_index=True,
)


# ============================================================
# 12. POOLED SUMMARY
# ============================================================

pooled_rows = []

for model_type in MODEL_TYPES:

    subset = forecasts_df[
        forecasts_df["model_type"]
        == model_type
    ].copy()

    if subset.empty:
        continue

    pooled = calculate_metrics(
        subset,
        model_type,
    )

    pooled[
        "summary_type"
    ] = "Pooled_validation"

    pooled[
        "block"
    ] = "Pooled"

    pooled_rows.append(
        pooled
    )

pooled_df = pd.DataFrame(
    pooled_rows
)


# ============================================================
# 13. SAVE RESULTS
# ============================================================

metrics_df.to_csv(
    METRICS_OUTPUT_FILE,
    index=False,
)

forecasts_df.to_csv(
    FORECAST_OUTPUT_FILE,
    index=False,
)


# ============================================================
# 14. PRINT BLOCK RESULTS
# ============================================================

print("\n" + "=" * 90)
print("BLOCK-LEVEL RESULTS")
print("=" * 90)

print()

print(
    metrics_df[
        [
            "block",
            "model_type",
            "validation_observations",
            "actual_spikes",
            "predicted_spikes",
            "MAE",
            "RMSE",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "AUC_volatility",
            "AUC_probability",
            "mean_probability",
            "probability_std",
        ]
    ]
    .sort_values(
        [
            "block",
            "model_type",
        ]
    )
    .to_string(index=False)
)


# ============================================================
# 15. PRINT POOLED RESULTS
# ============================================================

print("\n" + "=" * 90)
print("POOLED VALIDATION RESULTS")
print("=" * 90)

print()

print(
    pooled_df[
        [
            "model_type",
            "validation_observations",
            "actual_spikes",
            "predicted_spikes",
            "actual_spike_rate",
            "predicted_spike_rate",
            "MAE",
            "RMSE",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "AUC_volatility",
            "AUC_probability",
            "mean_probability",
            "probability_std",
        ]
    ]
    .sort_values("model_type")
    .to_string(index=False)
)


# ============================================================
# 16. CONCLUSION FLAGS
# ============================================================

print("\n" + "=" * 90)
print("STEP 09R INTERPRETATION GUIDE")
print("=" * 90)

print(
    "\nCompare raw vs log1p using the validation results only."
)

print(
    "Do not use the 2021-2025 final test period in this step."
)

print(
    "The spike threshold and 0.50 probability cutoff are fixed."
)

print(
    "If log1p improves the pooled RMSE/AUC pattern consistently "
    "across blocks, it is evidence that transforming the highly "
    "skewed squared-return target may help BVAR."
)

print(
    "If log1p does not improve the validation pattern consistently, "
    "retain the raw BVAR and freeze it."
)

print(
    f"\nMetrics saved: {METRICS_OUTPUT_FILE}"
)

print(
    f"Forecasts saved: {FORECAST_OUTPUT_FILE}"
)

print(
    "\nStep 09R completed."
)
