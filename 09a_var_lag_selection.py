#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 17:57:56 2026

@author: setup
"""

# -*- coding: utf-8 -*-

"""
Step 09A: VAR Lag-Order Selection for the BVAR Framework

Purpose:
    Determine a suitable lag order before estimating the
    Bayesian VAR.

Variables:
    1. fx_return_squared
    2. brent_log_return
    3. reserves_log_change
    4. fx_log_return

Candidate lag orders:
    VAR(1) through VAR(5)

Selection criteria:
    AIC
    BIC

Important:
    Long data gaps are treated as segment boundaries.
    Lagged observations are therefore NOT allowed to cross
    a long-gap boundary.

Training period:
    Up to and including 2020-12-31.

The selected lag order will be used for the BVAR stage.
"""


import os

import numpy as np
import pandas as pd


# ============================================================
# 1. FILE PATHS
# ============================================================

INPUT_FILE = (
    "data_cleaned/forecasting_dataset_2006_2025_spike.csv"
)

OUTPUT_FILE = (
    "results/var_lag_selection_spike.csv"
)


# ============================================================
# 2. SETTINGS
# ============================================================

TRAIN_END = pd.Timestamp(
    "2020-12-31"
)

MAX_LAG = 5

VARIABLES = [
    "fx_return_squared",
    "brent_log_return",
    "reserves_log_change",
    "fx_log_return"
]


# ============================================================
# 3. LOAD DATA
# ============================================================

if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )


df = pd.read_csv(
    INPUT_FILE
)


print("=" * 90)
print("STEP 09A: VAR LAG-ORDER SELECTION")
print("=" * 90)

print(
    f"\nRows loaded: {len(df):,}"
)


# ============================================================
# 4. VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "date",
    "long_gap"
] + VARIABLES


missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# 5. CLEAN DATA
# ============================================================

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


for col in VARIABLES:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


df["long_gap"] = pd.to_numeric(
    df["long_gap"],
    errors="coerce"
)


df = df.dropna(
    subset=[
        "date"
    ] + VARIABLES
).copy()


df = df.sort_values(
    "date"
).reset_index(
    drop=True
)


# ============================================================
# 6. TRAINING PERIOD ONLY
# ============================================================

train = df[
    df["date"] <= TRAIN_END
].copy()


print(
    f"\nTraining period: "
    f"{train['date'].min().date()} "
    f"to "
    f"{train['date'].max().date()}"
)

print(
    f"Training observations before lag construction: "
    f"{len(train):,}"
)


# ============================================================
# 7. CREATE SEGMENT IDENTIFIER
# ============================================================

# A long gap starts a new continuous segment.

train["segment_id"] = (
    train["long_gap"]
    .fillna(0)
    .astype(int)
    .cumsum()
)


print(
    f"Continuous segments: "
    f"{train['segment_id'].nunique():,}"
)


# ============================================================
# 8. STANDARDIZE TRAINING VARIABLES
# ============================================================
#
# Standardization is used ONLY for numerical stability
# during lag-order comparison.
#
# It does not alter the temporal structure.
#
# The same transformation can later be applied consistently
# in the BVAR stage.


means = train[
    VARIABLES
].mean()


stds = train[
    VARIABLES
].std(
    ddof=0
)


if (stds == 0).any():

    zero_variance_variables = (
        stds[stds == 0].index.tolist()
    )

    raise ValueError(
        f"Zero-variance variables found: "
        f"{zero_variance_variables}"
    )


train_scaled = train.copy()


for col in VARIABLES:

    train_scaled[col] = (
        train_scaled[col]
        - means[col]
    ) / stds[col]


# ============================================================
# 9. FUNCTION TO BUILD SEGMENT-SAFE LAGS
# ============================================================

def build_var_data(
    data,
    variables,
    lag_order
):
    """
    Build a VAR regression dataset without allowing lags
    to cross long-gap boundaries.

    For each valid target observation t:

        Y_t =
            intercept
            + A1 Y_(t-1)
            + ...
            + Ap Y_(t-p)
            + error_t

    Returns:
        Y : target matrix
        X : design matrix
    """

    y_blocks = []
    x_blocks = []


    for segment_id, segment in data.groupby(
        "segment_id"
    ):

        segment = (
            segment.sort_values("date")
            .reset_index(drop=True)
        )


        # Need at least p + 1 observations
        if len(segment) <= lag_order:
            continue


        values = segment[
            variables
        ].to_numpy(
            dtype=float
        )


        # ----------------------------------------------------
        # Build observations inside this segment only
        # ----------------------------------------------------

        for t in range(
            lag_order,
            len(values)
        ):

            target = values[t]


            lag_values = []


            # Most recent lag first
            for lag in range(
                1,
                lag_order + 1
            ):

                lag_values.extend(
                    values[t - lag]
                )


            # Intercept is added first
            row = [
                1.0
            ]

            row.extend(
                lag_values
            )


            x_blocks.append(
                row
            )

            y_blocks.append(
                target
            )


    if not y_blocks:

        return None, None


    X = np.asarray(
        x_blocks,
        dtype=float
    )

    Y = np.asarray(
        y_blocks,
        dtype=float
    )


    return Y, X


# ============================================================
# 10. FIT VAR USING MULTIVARIATE OLS
# ============================================================

results = []


print("\n" + "=" * 90)
print("TESTING VAR LAG ORDERS")
print("=" * 90)


for lag_order in range(
    1,
    MAX_LAG + 1
):

    print(
        f"\nTesting VAR({lag_order})..."
    )


    Y, X = build_var_data(
        train_scaled,
        VARIABLES,
        lag_order
    )


    if Y is None:

        print(
            "No usable observations."
        )

        continue


    n_obs = Y.shape[0]

    n_variables = Y.shape[1]

    n_parameters_per_equation = X.shape[1]

    total_parameters = (
        n_variables
        * n_parameters_per_equation
    )


    # --------------------------------------------------------
    # Multivariate OLS
    # --------------------------------------------------------

    coefficients, residuals, rank, singular_values = (
        np.linalg.lstsq(
            X,
            Y,
            rcond=None
        )
    )


    # --------------------------------------------------------
    # Residual matrix
    # --------------------------------------------------------

    fitted_values = (
        X
        @ coefficients
    )


    errors = (
        Y
        - fitted_values
    )


    # --------------------------------------------------------
    # Residual covariance matrix
    # --------------------------------------------------------

    sigma = (
        errors.T
        @ errors
        / n_obs
    )


    # Add tiny numerical protection
    sigma = (
        sigma
        + np.eye(n_variables)
        * 1e-12
    )


    sign, logdet = np.linalg.slogdet(
        sigma
    )


    if sign <= 0:

        print(
            "Residual covariance matrix "
            "is not positive definite."
        )

        continue


    # --------------------------------------------------------
    # Multivariate Gaussian log-likelihood
    # --------------------------------------------------------

    log_likelihood = (
        -0.5
        * n_obs
        * (
            n_variables
            * np.log(
                2 * np.pi
            )
            + logdet
            + n_variables
        )
    )


    # --------------------------------------------------------
    # AIC
    # --------------------------------------------------------

    aic = (
        -2 * log_likelihood
        + 2 * total_parameters
    )


    # --------------------------------------------------------
    # BIC
    # --------------------------------------------------------

    bic = (
        -2 * log_likelihood
        + total_parameters
        * np.log(n_obs)
    )


    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    results.append({

        "lag_order": lag_order,

        "observations": n_obs,

        "variables": n_variables,

        "parameters_per_equation":
            n_parameters_per_equation,

        "total_parameters":
            total_parameters,

        "log_likelihood":
            log_likelihood,

        "AIC":
            aic,

        "BIC":
            bic
    })


    print(
        f"VAR({lag_order}) | "
        f"n={n_obs:,} | "
        f"AIC={aic:.6f} | "
        f"BIC={bic:.6f}"
    )


# ============================================================
# 11. RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)


if results_df.empty:

    raise RuntimeError(
        "No VAR lag-order models could be estimated."
    )


results_df = results_df.sort_values(
    "lag_order"
).reset_index(
    drop=True
)


# Rank by AIC and BIC
results_df["AIC_rank"] = (
    results_df["AIC"]
    .rank(
        method="min",
        ascending=True
    )
)


results_df["BIC_rank"] = (
    results_df["BIC"]
    .rank(
        method="min",
        ascending=True
    )
)


# ============================================================
# 12. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 90)
print("VAR LAG-ORDER COMPARISON")
print("=" * 90)


print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 13. DETERMINE PREFERRED LAGS
# ============================================================

best_aic_row = (
    results_df.loc[
        results_df["AIC"].idxmin()
    ]
)


best_bic_row = (
    results_df.loc[
        results_df["BIC"].idxmin()
    ]
)


best_aic_lag = int(
    best_aic_row["lag_order"]
)


best_bic_lag = int(
    best_bic_row["lag_order"]
)


print("\n" + "=" * 90)
print("LAG-ORDER SELECTION")
print("=" * 90)


print(
    f"Lowest AIC: "
    f"VAR({best_aic_lag})"
)


print(
    f"Lowest BIC: "
    f"VAR({best_bic_lag})"
)


# ============================================================
# 14. SIMPLE SELECTION RULE
# ============================================================

if best_aic_lag == best_bic_lag:

    selected_lag = best_aic_lag

    selection_reason = (
        "AIC and BIC selected the same lag order."
    )

else:

    # When criteria disagree, use BIC as the
    # more conservative criterion for the initial
    # BVAR specification.

    selected_lag = best_bic_lag

    selection_reason = (
        "AIC and BIC disagreed; BIC was used as the "
        "more conservative lag-order criterion."
    )


print(
    f"\nSelected lag order for BVAR: "
    f"VAR({selected_lag})"
)


print(
    f"Reason: {selection_reason}"
)


# ============================================================
# 15. SAVE SELECTED LAG INFORMATION
# ============================================================

results_df["selected_for_bvar"] = (
    results_df["lag_order"]
    == selected_lag
)


# ============================================================
# 16. SAVE OUTPUT
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 17. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 90)
print("RESULT SAVED")
print("=" * 90)


print(
    f"File: {OUTPUT_FILE}"
)


print("=" * 90)