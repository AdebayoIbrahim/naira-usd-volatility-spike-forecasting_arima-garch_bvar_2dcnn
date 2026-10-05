#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 18:28:31 2026

@author: setup
"""

# ============================================================
# STEP 09G: BAYESIAN VAR(8) ESTIMATION
# ============================================================

import os
import numpy as np
import pandas as pd

from scipy.stats import invwishart
from statsmodels.stats.diagnostic import acorr_ljungbox


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "data_cleaned/"
    "forecasting_dataset_2006_2025_spike_gap_corrected.csv"
)

COEF_OUTPUT = (
    "results/bvar8_posterior_coefficients_spike.csv"
)

DIAG_OUTPUT = (
    "results/bvar8_diagnostics_spike.csv"
)

STATE_OUTPUT = (
    "results/bvar8_model_state_spike.npz"
)

TRAIN_END = "2020-12-31"

LAG_ORDER = 8

N_DRAWS = 3000

RANDOM_SEED = 42

LAMBDA = 0.20

INTERCEPT_VARIANCE = 100.0


# ============================================================
# VARIABLES
# ============================================================

VARIABLES = [
    "fx_log_return",
    "fx_return_squared",
    "brent_log_return",
    "reserves_log_change"
]

K = len(VARIABLES)


# ============================================================
# OUTPUT FOLDER
# ============================================================

os.makedirs("results", exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)


print("=" * 90)
print("STEP 09G: BAYESIAN VAR(8) ESTIMATION")
print("=" * 90)

print(f"\nRows loaded: {len(df):,}")


# ============================================================
# TRAINING DATA
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
    f"Training observations: "
    f"{len(train):,}"
)

print(
    f"Training continuous segments: "
    f"{train['segment_id'].nunique():,}"
)

print(
    f"Training long gaps: "
    f"{train['long_gap'].sum():,}"
)


# ============================================================
# CHECK VARIABLES
# ============================================================

required_columns = [
    "date",
    "segment_id",
    "long_gap"
] + VARIABLES

missing_columns = [
    col
    for col in required_columns
    if col not in train.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

train = train.dropna(
    subset=VARIABLES
).copy()

print(
    f"\nTraining observations after NA removal: "
    f"{len(train):,}"
)


# ============================================================
# STANDARDIZE
# ============================================================

means = train[VARIABLES].mean()

stds = train[VARIABLES].std(ddof=0)

if (stds <= 0).any():
    bad = stds[stds <= 0].index.tolist()

    raise ValueError(
        f"Zero-variance variables found: {bad}"
    )


train_scaled = train.copy()

for col in VARIABLES:

    train_scaled[col] = (
        train[col] - means[col]
    ) / stds[col]


# ============================================================
# BUILD SEGMENT-SAFE VAR DESIGN
# ============================================================

def build_var_design(data, variables, p):

    X_parts = []
    Y_parts = []

    for segment_id, segment in data.groupby(
        "segment_id",
        sort=True
    ):

        segment = (
            segment
            .sort_values("date")
            .reset_index(drop=True)
        )

        values = segment[
            variables
        ].to_numpy(dtype=float)

        n = len(values)

        if n <= p:
            continue

        X_segment = []
        Y_segment = []

        for t in range(p, n):

            row = [1.0]

            for lag in range(1, p + 1):

                row.extend(
                    values[t - lag]
                )

            X_segment.append(row)

            Y_segment.append(
                values[t]
            )

        X_parts.append(
            np.asarray(X_segment)
        )

        Y_parts.append(
            np.asarray(Y_segment)
        )

    if not X_parts:
        raise ValueError(
            "No valid VAR observations created."
        )

    X = np.vstack(X_parts)

    Y = np.vstack(Y_parts)

    return X, Y


# ============================================================
# CREATE VAR(8) DESIGN
# ============================================================

X, Y = build_var_design(
    train_scaled,
    VARIABLES,
    LAG_ORDER
)

n = Y.shape[0]

M = X.shape[1]


print("\n" + "=" * 90)
print("VAR(8) DESIGN")
print("=" * 90)

print(
    f"\nEffective observations: {n:,}"
)

print(
    f"Number of endogenous variables: {K}"
)

print(
    f"Regressors per equation: {M}"
)

print(
    f"Number of coefficient parameters: "
    f"{M * K:,}"
)


# ============================================================
# BAYESIAN PRIOR
# ============================================================

print("\n" + "=" * 90)
print("BAYESIAN PRIOR")
print("=" * 90)

print(
    f"\nShrinkage lambda: {LAMBDA}"
)

print(
    f"Intercept prior variance: "
    f"{INTERCEPT_VARIANCE}"
)


# Prior mean
B0 = np.zeros(
    (M, K)
)


# Prior covariance
V0 = np.zeros(
    (M, M)
)


# Intercept
V0[0, 0] = INTERCEPT_VARIANCE


# Lag coefficients
row = 1

for lag in range(1, LAG_ORDER + 1):

    lag_variance = (
        LAMBDA ** 2
        / (lag ** 2)
    )

    for _ in range(K):

        V0[row, row] = (
            lag_variance
        )

        row += 1


# Numerical stabilization
V0 += (
    np.eye(M) * 1e-10
)


# Prior covariance matrix
nu0 = K + 2

S0 = np.eye(K)


# ============================================================
# POSTERIOR
# ============================================================

print("\n" + "=" * 90)
print("CALCULATING POSTERIOR")
print("=" * 90)


V0_inv = np.linalg.inv(V0)

XtX = X.T @ X

XtY = X.T @ Y

Vn = np.linalg.inv(
    V0_inv + XtX
)

Bn = Vn @ (
    V0_inv @ B0
    + XtY
)

nun = nu0 + n

Sn = (
    S0
    + Y.T @ Y
    + B0.T @ V0_inv @ B0
    - Bn.T @ np.linalg.inv(Vn) @ Bn
)

Sn = (
    Sn + Sn.T
) / 2

Sn += (
    np.eye(K) * 1e-10
)


print(
    f"\nPrior degrees of freedom: {nu0}"
)

print(
    f"Posterior degrees of freedom: {nun}"
)


# ============================================================
# POSITIVE DEFINITENESS
# ============================================================

try:

    np.linalg.cholesky(Vn)

    np.linalg.cholesky(Sn)

    print(
        "\nPosterior covariance matrices "
        "are positive definite."
    )

except np.linalg.LinAlgError:

    raise ValueError(
        "Posterior covariance matrix "
        "is not positive definite."
    )


# ============================================================
# POSTERIOR DRAWS
# ============================================================

print("\n" + "=" * 90)
print("DRAWING FROM POSTERIOR")
print("=" * 90)


rng = np.random.default_rng(
    RANDOM_SEED
)

L_V = np.linalg.cholesky(Vn)

posterior_B = np.zeros(
    (N_DRAWS, M, K)
)

posterior_Sigma = np.zeros(
    (N_DRAWS, K, K)
)


for draw in range(N_DRAWS):

    Sigma_draw = invwishart.rvs(
        df=nun,
        scale=Sn,
        random_state=rng
    )

    Sigma_draw = (
        Sigma_draw
        + Sigma_draw.T
    ) / 2

    L_S = np.linalg.cholesky(
        Sigma_draw
    )

    Z = rng.normal(
        size=(M, K)
    )

    B_draw = (
        Bn
        + L_V @ Z @ L_S.T
    )

    posterior_B[draw] = B_draw

    posterior_Sigma[draw] = (
        Sigma_draw
    )


print(
    f"\nPosterior draws generated: "
    f"{N_DRAWS:,}"
)


# ============================================================
# COEFFICIENT SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("POSTERIOR COEFFICIENT SUMMARY")
print("=" * 90)


parameter_names = [
    "intercept"
]

for lag in range(
    1,
    LAG_ORDER + 1
):

    for variable in VARIABLES:

        parameter_names.append(
            f"{variable}_lag{lag}"
        )


rows = []

for i, parameter in enumerate(
    parameter_names
):

    for j, equation in enumerate(
        VARIABLES
    ):

        draws = (
            posterior_B[
                :,
                i,
                j
            ]
        )

        rows.append({

            "parameter": parameter,

            "equation": equation,

            "posterior_mean":
                draws.mean(),

            "posterior_sd":
                draws.std(ddof=1),

            "ci_lower_95":
                np.quantile(
                    draws,
                    0.025
                ),

            "ci_upper_95":
                np.quantile(
                    draws,
                    0.975
                )
        })


coef_df = pd.DataFrame(
    rows
)

coef_df.to_csv(
    COEF_OUTPUT,
    index=False
)


# ============================================================
# POSTERIOR MEAN RESIDUALS
# ============================================================

B_mean = Bn

Y_hat = X @ B_mean

residuals = Y - Y_hat


# ============================================================
# LJUNG-BOX DIAGNOSTICS
# ============================================================

print("\n" + "=" * 90)
print("POSTERIOR MEAN RESIDUAL DIAGNOSTICS")
print("=" * 90)


diagnostic_rows = []


for j, variable in enumerate(
    VARIABLES
):

    residual_series = (
        residuals[:, j]
    )

    # Ordinary residuals
    lb10 = acorr_ljungbox(
        residual_series,
        lags=[10],
        return_df=True
    )

    lb20 = acorr_ljungbox(
        residual_series,
        lags=[20],
        return_df=True
    )

    # Squared residuals
    squared = (
        residual_series ** 2
    )

    lb_sq10 = acorr_ljungbox(
        squared,
        lags=[10],
        return_df=True
    )

    lb_sq20 = acorr_ljungbox(
        squared,
        lags=[20],
        return_df=True
    )

    diagnostic_rows.append({

        "variable": variable,

        "residual_mean":
            residual_series.mean(),

        "residual_std":
            residual_series.std(
                ddof=1
            ),

        "ljung_box_10_pvalue":
            float(
                lb10[
                    "lb_pvalue"
                ].iloc[0]
            ),

        "ljung_box_20_pvalue":
            float(
                lb20[
                    "lb_pvalue"
                ].iloc[0]
            ),

        "squared_residual_ljung_box_10_pvalue":
            float(
                lb_sq10[
                    "lb_pvalue"
                ].iloc[0]
            ),

        "squared_residual_ljung_box_20_pvalue":
            float(
                lb_sq20[
                    "lb_pvalue"
                ].iloc[0]
            )
    })


diagnostics_df = pd.DataFrame(
    diagnostic_rows
)


# ============================================================
# SAVE DIAGNOSTICS
# ============================================================

diagnostics_df.to_csv(
    DIAG_OUTPUT,
    index=False
)


# ============================================================
# SAVE MODEL STATE
# ============================================================

np.savez(
    STATE_OUTPUT,

    Bn=Bn,

    Vn=Vn,

    Sn=Sn,

    posterior_B=posterior_B,

    posterior_Sigma=posterior_Sigma,

    means=means.to_numpy(),

    stds=stds.to_numpy(),

    variable_names=np.array(
        VARIABLES,
        dtype=object
    ),

    target_variable=TARGET_VARIABLE
        if "TARGET_VARIABLE" in globals()
        else "fx_return_squared",

    lag_order=LAG_ORDER,

    train_end=TRAIN_END,

    lambda_value=LAMBDA,

    nu0=nu0,

    nun=nun
)


# ============================================================
# DISPLAY DIAGNOSTICS
# ============================================================

print()

for _, row in diagnostics_df.iterrows():

    print(
        f"{row['variable']}: "
        f"LB(10) p="
        f"{row['ljung_box_10_pvalue']:.6f}, "
        f"LB(20) p="
        f"{row['ljung_box_20_pvalue']:.6f}, "
        f"Squared LB(10) p="
        f"{row['squared_residual_ljung_box_10_pvalue']:.6f}, "
        f"Squared LB(20) p="
        f"{row['squared_residual_ljung_box_20_pvalue']:.6f}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("STEP 09G COMPLETE")
print("=" * 90)

print(
    f"\nModel: Bayesian VAR({LAG_ORDER})"
)

print(
    f"Effective training observations: "
    f"{n:,}"
)

print(
    f"Posterior draws: "
    f"{N_DRAWS:,}"
)

print("\nFiles saved:")

print(
    f"  {COEF_OUTPUT}"
)

print(
    f"  {DIAG_OUTPUT}"
)

print(
    f"  {STATE_OUTPUT}"
)

print("=" * 90)