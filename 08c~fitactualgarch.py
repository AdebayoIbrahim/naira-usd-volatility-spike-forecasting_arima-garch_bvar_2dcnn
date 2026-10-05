#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 13:26:25 2026

@author: setup
"""

# -*- coding: utf-8 -*-
"""
Step 08C: Final GARCH(2,1)-Student-t Fit and Diagnostics

Selected model from Step 08B:
    GARCH(2,1) with Student-t innovations

Input:
    ARIMA selected residuals

Purpose:
    1. Refit the selected GARCH model.
    2. Extract parameter estimates.
    3. Calculate volatility persistence.
    4. Check standardized residuals.
    5. Check squared standardized residuals.
    6. Test for remaining ARCH effects.
    7. Save final training conditional volatility.
"""

import os
import warnings

import numpy as np
import pandas as pd

from arch import arch_model
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_arch
)


# ============================================================
# 1. FILE PATHS
# ============================================================

INPUT_FILE = (
    "results/arima_selected_residuals_spike.csv"
)

MODEL_OUTPUT = (
    "results/final_garch_model_spike.csv"
)

DIAGNOSTIC_OUTPUT = (
    "results/final_garch_diagnostics_spike.csv"
)

VOLATILITY_OUTPUT = (
    "results/final_garch_conditional_volatility_spike.csv"
)


# ============================================================
# 2. SETTINGS
# ============================================================

P = 2
Q = 1

DISTRIBUTION = "t"

ARCH_LAGS = 10


# ============================================================
# 3. LOAD ARIMA RESIDUALS
# ============================================================

if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )


residual_df = pd.read_csv(
    INPUT_FILE
)


# ============================================================
# 4. VALIDATE DATA
# ============================================================

required_columns = [
    "date",
    "arima_residual"
]

missing_columns = [
    col
    for col in required_columns
    if col not in residual_df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# 5. CLEAN DATA
# ============================================================

residual_df["date"] = pd.to_datetime(
    residual_df["date"],
    errors="coerce"
)

residual_df["arima_residual"] = pd.to_numeric(
    residual_df["arima_residual"],
    errors="coerce"
)

residual_df = residual_df.dropna(
    subset=[
        "date",
        "arima_residual"
    ]
).copy()

residual_df = residual_df.sort_values(
    "date"
).reset_index(drop=True)


# ============================================================
# 6. KEEP ORIGINAL PROJECT RETURN SCALE
# ============================================================

residuals = (
    residual_df["arima_residual"]
    .astype(float)
    .reset_index(drop=True)
)


# ============================================================
# 7. BASIC CHECKS
# ============================================================

if len(residuals) < 100:

    raise ValueError(
        "Too few residual observations."
    )


if not np.isfinite(residuals).all():

    raise ValueError(
        "Non-finite residual values detected."
    )


print("=" * 80)
print("STEP 08C: FINAL GARCH FIT AND DIAGNOSTICS")
print("=" * 80)

print(
    f"\nModel: GARCH({P},{Q})-{DISTRIBUTION}"
)

print(
    f"Observations: {len(residuals):,}"
)

print(
    f"Start: {residual_df['date'].min().date()}"
)

print(
    f"End:   {residual_df['date'].max().date()}"
)


# ============================================================
# 8. FIT FINAL GARCH MODEL
# ============================================================

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    model = arch_model(
        residuals,
        mean="Zero",
        vol="GARCH",
        p=P,
        q=Q,
        dist=DISTRIBUTION,
        rescale=False
    )

    fitted = model.fit(
        update_freq=0,
        disp="off",
        options={
            "maxiter": 5000,
            "ftol": 1e-10
        }
    )


# ============================================================
# 9. CONVERGENCE CHECK
# ============================================================

print("\n" + "=" * 80)
print("CONVERGENCE CHECK")
print("=" * 80)

print(
    f"Convergence flag: "
    f"{fitted.convergence_flag}"
)

print(
    f"Optimizer message: "
    f"{fitted.optimization_result.message}"
)


if fitted.convergence_flag != 0:

    raise RuntimeError(
        "Final GARCH model did not converge."
    )


# ============================================================
# 10. MODEL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL GARCH MODEL SUMMARY")
print("=" * 80)

print(
    fitted.summary()
)


# ============================================================
# 11. EXTRACT PARAMETERS
# ============================================================

params = fitted.params


omega = float(
    params.get(
        "omega",
        np.nan
    )
)

alpha1 = float(
    params.get(
        "alpha[1]",
        np.nan
    )
)

alpha2 = float(
    params.get(
        "alpha[2]",
        np.nan
    )
)

beta1 = float(
    params.get(
        "beta[1]",
        np.nan
    )
)

nu = float(
    params.get(
        "nu",
        np.nan
    )
)


# ============================================================
# 12. CALCULATE VOLATILITY PERSISTENCE
# ============================================================

persistence = (
    alpha1
    + alpha2
    + beta1
)


print("\n" + "=" * 80)
print("GARCH PARAMETERS")
print("=" * 80)

print(
    f"omega:     {omega:.10f}"
)

print(
    f"alpha[1]:  {alpha1:.10f}"
)

print(
    f"alpha[2]:  {alpha2:.10f}"
)

print(
    f"beta[1]:   {beta1:.10f}"
)

print(
    f"nu:        {nu:.10f}"
)

print(
    f"\nVolatility persistence "
    f"(alpha1 + alpha2 + beta1): "
    f"{persistence:.10f}"
)


# ============================================================
# 13. STANDARDIZED RESIDUALS
# ============================================================

std_resid_values = np.asarray(
    fitted.std_resid,
    dtype=float
)

valid_mask = np.isfinite(
    std_resid_values
)

std_resid_values = (
    std_resid_values[valid_mask]
)

std_dates = (
    residual_df["date"]
    .to_numpy()
)

std_dates = (
    std_dates[valid_mask]
)


standardized_residuals = pd.Series(
    std_resid_values,
    index=std_dates,
    name="standardized_residual"
)


if len(standardized_residuals) < 21:

    raise RuntimeError(
        "Too few valid standardized residuals "
        "for diagnostics."
    )


squared_standardized_residuals = (
    standardized_residuals ** 2
)


print("\n" + "=" * 80)
print("STANDARDIZED RESIDUAL CHECK")
print("=" * 80)

print(
    f"Valid standardized residuals: "
    f"{len(standardized_residuals):,}"
)


# ============================================================
# 14. LJUNG-BOX: STANDARDIZED RESIDUALS
# ============================================================

lb_standardized = acorr_ljungbox(
    standardized_residuals,
    lags=[10, 20],
    return_df=True
)


print("\n" + "=" * 80)
print("LJUNG-BOX: STANDARDIZED RESIDUALS")
print("=" * 80)

print(
    lb_standardized.to_string()
)


# ============================================================
# 15. LJUNG-BOX: SQUARED STANDARDIZED RESIDUALS
# ============================================================

lb_squared = acorr_ljungbox(
    squared_standardized_residuals,
    lags=[10, 20],
    return_df=True
)


print("\n" + "=" * 80)
print("LJUNG-BOX: SQUARED STANDARDIZED RESIDUALS")
print("=" * 80)

print(
    lb_squared.to_string()
)


# ============================================================
# 16. ARCH-LM AFTER GARCH
# ============================================================

post_arch_test = het_arch(
    standardized_residuals,
    nlags=ARCH_LAGS
)


post_arch_lm_stat = post_arch_test[0]
post_arch_lm_pvalue = post_arch_test[1]

post_arch_f_stat = post_arch_test[2]
post_arch_f_pvalue = post_arch_test[3]


print("\n" + "=" * 80)
print("ARCH-LM TEST AFTER GARCH")
print("=" * 80)

print(
    f"ARCH-LM statistic: "
    f"{post_arch_lm_stat:.6f}"
)

print(
    f"ARCH-LM p-value:   "
    f"{post_arch_lm_pvalue:.6f}"
)

print(
    f"F-statistic:       "
    f"{post_arch_f_stat:.6f}"
)

print(
    f"F-test p-value:    "
    f"{post_arch_f_pvalue:.6f}"
)


# ============================================================
# 17. INTERPRETATION
# ============================================================

lb_std_p10 = float(
    lb_standardized.loc[
        10,
        "lb_pvalue"
    ]
)

lb_std_p20 = float(
    lb_standardized.loc[
        20,
        "lb_pvalue"
    ]
)

lb_sq_p10 = float(
    lb_squared.loc[
        10,
        "lb_pvalue"
    ]
)

lb_sq_p20 = float(
    lb_squared.loc[
        20,
        "lb_pvalue"
    ]
)


print("\n" + "=" * 80)
print("DIAGNOSTIC INTERPRETATION")
print("=" * 80)


if (
    lb_std_p10 >= 0.05
    and lb_std_p20 >= 0.05
):

    print(
        "Standardized residuals: "
        "no significant serial correlation "
        "at lags 10 and 20."
    )

else:

    print(
        "Standardized residuals: "
        "some serial correlation remains."
    )


if (
    lb_sq_p10 >= 0.05
    and lb_sq_p20 >= 0.05
):

    print(
        "Squared standardized residuals: "
        "no significant remaining volatility "
        "dependence at lags 10 and 20."
    )

else:

    print(
        "Squared standardized residuals: "
        "some volatility dependence remains."
    )


if post_arch_lm_pvalue >= 0.05:

    print(
        "Post-GARCH ARCH-LM: "
        "no significant remaining ARCH effect."
    )

else:

    print(
        "Post-GARCH ARCH-LM: "
        "significant ARCH effect remains."
    )


# ============================================================
# 18. CONDITIONAL VOLATILITY
# ============================================================

conditional_volatility = np.asarray(
    fitted.conditional_volatility,
    dtype=float
)


conditional_volatility_df = pd.DataFrame({
    "date": residual_df["date"],
    "conditional_volatility": conditional_volatility
})


# ============================================================
# 19. SAVE MODEL INFORMATION
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


model_data = {
    "model": f"GARCH({P},{Q})",
    "distribution": DISTRIBUTION,
    "aic": fitted.aic,
    "bic": fitted.bic,
    "loglik": fitted.loglikelihood,
    "convergence_flag": fitted.convergence_flag,

    "omega": omega,
    "alpha1": alpha1,
    "alpha2": alpha2,
    "beta1": beta1,
    "nu": nu,

    "volatility_persistence":
        persistence,

    "arch_lm_pvalue_after":
        post_arch_lm_pvalue,

    "ljung_box_std_pvalue_10":
        lb_std_p10,

    "ljung_box_std_pvalue_20":
        lb_std_p20,

    "ljung_box_squared_std_pvalue_10":
        lb_sq_p10,

    "ljung_box_squared_std_pvalue_20":
        lb_sq_p20,

    "training_observations":
        len(residuals),

    "training_start":
        residual_df["date"].min(),

    "training_end":
        residual_df["date"].max()
}


model_df = pd.DataFrame(
    [model_data]
)


model_df.to_csv(
    MODEL_OUTPUT,
    index=False
)


# ============================================================
# 20. SAVE DIAGNOSTICS
# ============================================================

diagnostic_df = pd.DataFrame({

    "diagnostic": [
        "Ljung-Box standardized residuals lag 10",
        "Ljung-Box standardized residuals lag 20",
        "Ljung-Box squared standardized residuals lag 10",
        "Ljung-Box squared standardized residuals lag 20",
        "ARCH-LM after GARCH"
    ],

    "statistic": [
        lb_standardized.loc[
            10,
            "lb_stat"
        ],

        lb_standardized.loc[
            20,
            "lb_stat"
        ],

        lb_squared.loc[
            10,
            "lb_stat"
        ],

        lb_squared.loc[
            20,
            "lb_stat"
        ],

        post_arch_lm_stat
    ],

    "p_value": [
        lb_std_p10,
        lb_std_p20,
        lb_sq_p10,
        lb_sq_p20,
        post_arch_lm_pvalue
    ]
})


diagnostic_df.to_csv(
    DIAGNOSTIC_OUTPUT,
    index=False
)


# ============================================================
# 21. SAVE CONDITIONAL VOLATILITY
# ============================================================

conditional_volatility_df.to_csv(
    VOLATILITY_OUTPUT,
    index=False
)


# ============================================================
# 22. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    f"Final model: "
    f"{MODEL_OUTPUT}"
)

print(
    f"Diagnostics: "
    f"{DIAGNOSTIC_OUTPUT}"
)

print(
    f"Conditional volatility: "
    f"{VOLATILITY_OUTPUT}"
)

print("=" * 80)