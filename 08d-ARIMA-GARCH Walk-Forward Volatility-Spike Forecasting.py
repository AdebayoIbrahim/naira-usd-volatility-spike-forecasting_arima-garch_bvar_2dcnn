#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 17:11:55 2026

@author: setup
"""

# -*- coding: utf-8 -*-

"""
Step 08D: ARIMA-GARCH Walk-Forward Volatility-Spike Forecasting

Final model:
    ARIMA(0,0,1) + GARCH(2,1) with Student-t innovations

Purpose:
    Produce genuine one-step-ahead out-of-sample forecasts for
    the volatility-spike target.

Target:
    Next observation's squared FX log return.

Spike rule:
    target_spike = 1 if next squared return > frozen threshold.

Important:
    The spike threshold was estimated from training data only.

    The model parameters are estimated from the training period
    and then kept fixed throughout the test period.

    Model states are updated recursively as each new observation
    becomes available.

    Long gaps are treated as boundaries:
        - forecasts across a long-gap target are skipped
        - the model state is reset when entering a new segment
"""


import os
import warnings

import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model


# ============================================================
# 1. FILE PATHS
# ============================================================

INPUT_FILE = (
    "data_cleaned/forecasting_dataset_2006_2025_spike.csv"
)

THRESHOLD_FILE = (
    "results/training_spike_threshold_new.csv"
)

FORECAST_OUTPUT = (
    "results/arima_garch_spike_walk_forward_forecasts.csv"
)

SUMMARY_OUTPUT = (
    "results/arima_garch_spike_forecast_summary.csv"
)


# ============================================================
# 2. SETTINGS
# ============================================================

TRAIN_END = pd.Timestamp("2020-12-31")

ARIMA_ORDER = (0, 0, 1)

GARCH_P = 2
GARCH_Q = 1

GARCH_DISTRIBUTION = "t"


# ============================================================
# 3. LOAD FORECASTING DATASET
# ============================================================

if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"Forecasting dataset not found:\n{INPUT_FILE}"
    )


df = pd.read_csv(
    INPUT_FILE
)


# ============================================================
# 4. VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "date",
    "target_date",
    "dataset",
    "fx_log_return",
    "target_volatility",
    "target_spike",
    "gap_days",
    "long_gap"
]


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

df["target_date"] = pd.to_datetime(
    df["target_date"],
    errors="coerce"
)

numeric_columns = [
    "fx_log_return",
    "target_volatility",
    "target_spike",
    "gap_days",
    "long_gap"
]


for col in numeric_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


df = df.dropna(
    subset=[
        "date",
        "target_date",
        "fx_log_return"
    ]
).copy()


df = df.sort_values(
    "date"
).reset_index(
    drop=True
)


# ============================================================
# 6. LOAD FROZEN SPIKE THRESHOLD
# ============================================================

if not os.path.exists(THRESHOLD_FILE):

    raise FileNotFoundError(
        f"Threshold file not found:\n{THRESHOLD_FILE}"
    )


threshold_df = pd.read_csv(
    THRESHOLD_FILE
)


if "threshold_r_squared" not in threshold_df.columns:

    raise ValueError(
        "Threshold file does not contain "
        "'threshold_r_squared'."
    )


SPIKE_THRESHOLD = float(
    threshold_df.loc[
        0,
        "threshold_r_squared"
    ]
)


print("=" * 90)
print("STEP 08D: ARIMA-GARCH WALK-FORWARD FORECASTING")
print("=" * 90)

print(
    f"\nFrozen spike threshold: "
    f"{SPIKE_THRESHOLD:.8f}"
)


# ============================================================
# 7. TRAINING DATA
# ============================================================

train = df[
    df["date"] <= TRAIN_END
].copy()


train_returns = (
    train["fx_log_return"]
    .astype(float)
    .reset_index(drop=True)
)


if len(train_returns) < 100:

    raise ValueError(
        "Too few training observations."
    )


print(
    f"Training observations: "
    f"{len(train_returns):,}"
)


# ============================================================
# 8. FIT ARIMA ON TRAINING DATA
# ============================================================

print("\n" + "=" * 90)
print("FITTING TRAINING ARIMA MODEL")
print("=" * 90)


with warnings.catch_warnings():

    warnings.simplefilter(
        "ignore"
    )

    arima_model = ARIMA(
        train_returns,
        order=ARIMA_ORDER,
        trend="c"
    )

    arima_fit = arima_model.fit()


print(
    f"ARIMA specification: "
    f"ARIMA{ARIMA_ORDER}"
)

print(
    f"AIC: {arima_fit.aic:.6f}"
)

print(
    f"BIC: {arima_fit.bic:.6f}"
)


# ============================================================
# 9. EXTRACT TRAINING ARIMA RESIDUALS
# ============================================================

arima_residuals = np.asarray(
    arima_fit.resid,
    dtype=float
)


if not np.isfinite(
    arima_residuals
).all():

    raise ValueError(
        "Non-finite ARIMA residuals detected."
    )


# ARIMA(0,0,1):
#
#   r_t = c + epsilon_t + theta * epsilon_(t-1)
#
# Therefore the next-period forecast is:
#
#   E(r_(t+1)|F_t) = c + theta * epsilon_t
#

arima_params = arima_fit.params


# Get parameter names safely
arima_param_names = list(
    arima_fit.param_names
)


param_dict = dict(
    zip(
        arima_param_names,
        arima_params
    )
)


if "const" in param_dict:

    arima_constant = float(
        param_dict["const"]
    )

elif "intercept" in param_dict:

    arima_constant = float(
        param_dict["intercept"]
    )

else:

    raise ValueError(
        "Could not identify the ARIMA constant parameter."
    )


if "ma.L1" not in param_dict:

    raise ValueError(
        "Could not identify the MA(1) coefficient."
    )


arima_theta = float(
    param_dict["ma.L1"]
)


print(
    f"ARIMA constant: "
    f"{arima_constant:.10f}"
)

print(
    f"ARIMA MA(1):    "
    f"{arima_theta:.10f}"
)


# ============================================================
# 10. FIT GARCH ON ARIMA RESIDUALS
# ============================================================

print("\n" + "=" * 90)
print("FITTING TRAINING GARCH MODEL")
print("=" * 90)


with warnings.catch_warnings():

    warnings.simplefilter(
        "ignore"
    )

    garch_model = arch_model(
        arima_residuals,
        mean="Zero",
        vol="GARCH",
        p=GARCH_P,
        q=GARCH_Q,
        dist=GARCH_DISTRIBUTION,
        rescale=False
    )

    garch_fit = garch_model.fit(
        update_freq=0,
        disp="off",
        options={
            "maxiter": 5000,
            "ftol": 1e-10
        }
    )


if garch_fit.convergence_flag != 0:

    raise RuntimeError(
        "Training GARCH model did not converge successfully."
    )


print(
    f"GARCH specification: "
    f"GARCH({GARCH_P},{GARCH_Q})-{GARCH_DISTRIBUTION}"
)

print(
    f"AIC: {garch_fit.aic:.6f}"
)

print(
    f"BIC: {garch_fit.bic:.6f}"
)

print(
    f"Log-likelihood: "
    f"{garch_fit.loglikelihood:.6f}"
)


# ============================================================
# 11. EXTRACT GARCH PARAMETERS
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


print(
    f"omega:    {omega:.10f}"
)

print(
    f"alpha[1]: {alpha1:.10f}"
)

print(
    f"alpha[2]: {alpha2:.10f}"
)

print(
    f"beta[1]:  {beta1:.10f}"
)


# ============================================================
# 12. INITIAL GARCH STATE
# ============================================================

training_conditional_variance = (
    np.asarray(
        garch_fit.conditional_volatility,
        dtype=float
    ) ** 2
)


if len(training_conditional_variance) != len(
    arima_residuals
):

    raise ValueError(
        "GARCH conditional-volatility length "
        "does not match ARIMA residual length."
    )


# Latest training states

current_variance = float(
    training_conditional_variance[-1]
)

current_residual = float(
    arima_residuals[-1]
)

previous_residual = float(
    arima_residuals[-2]
)


# ============================================================
# 13. UNCONDITIONAL VARIANCE
# ============================================================

persistence = (
    alpha1
    + alpha2
    + beta1
)


if persistence >= 1:

    print(
        "\nWarning: estimated GARCH persistence "
        "is >= 1."
    )

    unconditional_variance = (
        current_variance
    )

else:

    unconditional_variance = (
        omega
        / (
            1.0
            - persistence
        )
    )


print(
    f"\nGARCH persistence: "
    f"{persistence:.10f}"
)

print(
    f"Initial unconditional variance: "
    f"{unconditional_variance:.10f}"
)


# ============================================================
# 14. TEST FORECAST LOOP
# ============================================================

print("\n" + "=" * 90)
print("STARTING WALK-FORWARD TEST")
print("=" * 90)


forecast_records = []


# We need the row immediately after each forecast origin
# because target_date is the next observed date.

for i in range(
    len(df) - 1
):

    origin_row = df.iloc[i]
    target_row = df.iloc[i + 1]


    origin_date = origin_row["date"]
    target_date = target_row["date"]


    # --------------------------------------------------------
    # Only evaluate forecasts whose target is in TEST period.
    # --------------------------------------------------------

    if target_date <= TRAIN_END:

        continue


    # --------------------------------------------------------
    # If the target is the first observation after a LONG GAP,
    # do not pretend that this was a normal one-step daily
    # forecast. We will reset the state when that target becomes
    # the forecast origin instead.
    # --------------------------------------------------------

    target_is_long_gap_boundary = (
        int(
            target_row["long_gap"]
        ) == 1
    )


    # --------------------------------------------------------
    # If this origin is the first observation after a long gap,
    # reset the model state so the new segment does not inherit
    # the previous segment's state.
    # --------------------------------------------------------

    origin_is_new_segment = (
        int(
            origin_row["long_gap"]
        ) == 1
    )


    if origin_is_new_segment:

        # Reset previous residual information
        previous_residual = 0.0
        current_residual = 0.0

        # Reset current conditional variance
        current_variance = (
            unconditional_variance
        )

        # Use current observed return to reconstruct
        # its ARIMA residual within the new segment.

        current_return = float(
            origin_row["fx_log_return"]
        )

        current_mean = (
            arima_constant
            + arima_theta * previous_residual
        )

        current_residual = (
            current_return
            - current_mean
        )


        # After observing the first return of the segment,
        # compute the variance for the next observation.

        next_variance = (
            omega
            + alpha1 * current_residual ** 2
            + alpha2 * previous_residual ** 2
            + beta1 * current_variance
        )

    else:

        # ----------------------------------------------------
        # Normal forecasting step
        # ----------------------------------------------------

        next_mean = (
            arima_constant
            + arima_theta * current_residual
        )


        next_variance = (
            omega
            + alpha1 * current_residual ** 2
            + alpha2 * previous_residual ** 2
            + beta1 * current_variance
        )


    # --------------------------------------------------------
    # Conditional variance cannot be negative.
    # --------------------------------------------------------

    next_variance = max(
        float(next_variance),
        0.0
    )


    forecast_mean = (
        next_mean
        if not origin_is_new_segment
        else (
            arima_constant
            + arima_theta * current_residual
        )
    )


    # --------------------------------------------------------
    # Forecast conditional volatility
    # --------------------------------------------------------

    forecast_volatility = np.sqrt(
        next_variance
    )


    # --------------------------------------------------------
    # Since:
    #
    # E(R² | information)
    #     = Var(R | information)
    #       + E(R | information)²
    #
    # calculate the expected squared return.
    # --------------------------------------------------------

    forecast_squared_return = (
        next_variance
        + forecast_mean ** 2
    )


    # --------------------------------------------------------
    # If the target is immediately after a long gap, skip it.
    # --------------------------------------------------------

    if not target_is_long_gap_boundary:

        actual_return = float(
            target_row["fx_log_return"]
        )

        actual_squared_return = (
            actual_return ** 2
        )

        actual_spike = int(
            actual_squared_return
            > SPIKE_THRESHOLD
        )


        predicted_spike = int(
            forecast_squared_return
            > SPIKE_THRESHOLD
        )


        stored_target_spike = int(
            origin_row["target_spike"]
        )


        forecast_records.append({

            "forecast_origin":
                origin_date,

            "target_date":
                target_date,

            "forecast_mean":
                forecast_mean,

            "forecast_variance":
                next_variance,

            "forecast_volatility":
                forecast_volatility,

            "forecast_squared_return":
                forecast_squared_return,

            "actual_return":
                actual_return,

            "actual_squared_return":
                actual_squared_return,

            "actual_spike":
                actual_spike,

            "stored_target_spike":
                stored_target_spike,

            "predicted_spike":
                predicted_spike,

            "gap_days":
                target_row["gap_days"],

            "target_long_gap":
                target_row["long_gap"]
        })


    # --------------------------------------------------------
    # Once the target becomes known, update the model state.
    #
    # IMPORTANT:
    # We only update states for a normal within-segment
    # observation. A long-gap target is handled when it becomes
    # the next origin.
    # --------------------------------------------------------

    if not target_is_long_gap_boundary:

        target_return = float(
            target_row["fx_log_return"]
        )


        target_residual = (
            target_return
            - forecast_mean
        )


        # Shift residual states
        previous_residual = (
            current_residual
        )

        current_residual = (
            target_residual
        )


        # The forecast variance for the target becomes
        # the current conditional variance after observing it.

        current_variance = (
            next_variance
        )


# ============================================================
# 15. CREATE FORECAST DATAFRAME
# ============================================================

forecast_df = pd.DataFrame(
    forecast_records
)


if forecast_df.empty:

    raise RuntimeError(
        "No test forecasts were produced."
    )


forecast_df["forecast_origin"] = pd.to_datetime(
    forecast_df["forecast_origin"]
)

forecast_df["target_date"] = pd.to_datetime(
    forecast_df["target_date"]
)


forecast_df = forecast_df.sort_values(
    "target_date"
).reset_index(
    drop=True
)


# ============================================================
# 16. VERIFY STORED TARGET SPIKE
# ============================================================

target_mismatch = (
    forecast_df["actual_spike"]
    != forecast_df["stored_target_spike"]
).sum()


print("\n" + "=" * 90)
print("TARGET CONSISTENCY CHECK")
print("=" * 90)

print(
    f"Target spike mismatches: "
    f"{target_mismatch:,}"
)


if target_mismatch > 0:

    raise RuntimeError(
        "Recomputed spike labels do not match "
        "the stored target_spike values."
    )


# ============================================================
# 17. BASIC SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("FORECAST SUMMARY")
print("=" * 90)

print(
    f"Forecast observations: "
    f"{len(forecast_df):,}"
)

print(
    f"Forecast period: "
    f"{forecast_df['target_date'].min().date()} "
    f"to "
    f"{forecast_df['target_date'].max().date()}"
)

print(
    f"Actual spikes: "
    f"{forecast_df['actual_spike'].sum():,}"
)

print(
    f"Actual spike rate: "
    f"{forecast_df['actual_spike'].mean() * 100:.2f}%"
)

print(
    f"Predicted spikes: "
    f"{forecast_df['predicted_spike'].sum():,}"
)

print(
    f"Predicted spike rate: "
    f"{forecast_df['predicted_spike'].mean() * 100:.2f}%"
)


# ============================================================
# 18. SAVE FORECASTS
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


forecast_df.to_csv(
    FORECAST_OUTPUT,
    index=False
)


# ============================================================
# 19. SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame({

    "metric": [

        "training_observations",

        "forecast_observations",

        "forecast_start",

        "forecast_end",

        "spike_threshold",

        "actual_spikes",

        "actual_spike_rate",

        "predicted_spikes",

        "predicted_spike_rate"
    ],

    "value": [

        len(train_returns),

        len(forecast_df),

        forecast_df[
            "target_date"
        ].min(),

        forecast_df[
            "target_date"
        ].max(),

        SPIKE_THRESHOLD,

        forecast_df[
            "actual_spike"
        ].sum(),

        forecast_df[
            "actual_spike"
        ].mean() * 100,

        forecast_df[
            "predicted_spike"
        ].sum(),

        forecast_df[
            "predicted_spike"
        ].mean() * 100
    ]
})


summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False
)


# ============================================================
# 20. SHOW FIRST FEW FORECASTS
# ============================================================

print("\n" + "=" * 90)
print("FIRST 10 OUT-OF-SAMPLE FORECASTS")
print("=" * 90)


print(
    forecast_df[
        [
            "forecast_origin",
            "target_date",
            "forecast_mean",
            "forecast_variance",
            "forecast_squared_return",
            "actual_squared_return",
            "predicted_spike",
            "actual_spike"
        ]
    ].head(10).to_string(
        index=False
    )
)


# ============================================================
# 21. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 90)
print("FILES SAVED")
print("=" * 90)

print(
    f"Forecasts: "
    f"{FORECAST_OUTPUT}"
)

print(
    f"Summary:   "
    f"{SUMMARY_OUTPUT}"
)

print("=" * 90)