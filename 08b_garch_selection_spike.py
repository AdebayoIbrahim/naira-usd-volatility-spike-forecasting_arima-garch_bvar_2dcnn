# -*- coding: utf-8 -*-
"""
Step 08B: GARCH Model Selection for the New Volatility-Spike Framework

Purpose:
    Select the conditional variance specification after the
    ARIMA mean model.

Input:
    Residuals from the selected ARIMA(0,0,1) model.

Candidate variance models:
    GARCH(1,1)
    GARCH(1,2)
    GARCH(2,1)
    GARCH(2,2)

Innovation distributions:
    Normal
    Student-t

Selection:
    Primary: AIC
    Secondary: BIC
    Diagnostics:
        - ARCH-LM before GARCH
        - Ljung-Box standardized residuals
        - Ljung-Box squared standardized residuals
        - ARCH-LM after GARCH
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

CANDIDATE_OUTPUT = (
    "results/garch_candidate_models_spike.csv"
)

SELECTED_OUTPUT = (
    "results/selected_garch_model_spike.csv"
)

DIAGNOSTIC_OUTPUT = (
    "results/garch_selected_diagnostics_spike.csv"
)

VOLATILITY_OUTPUT = (
    "results/garch_selected_conditional_volatility_spike.csv"
)


# ============================================================
# 2. SETTINGS
# ============================================================

ARCH_LAGS = 10

GARCH_ORDERS = [
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2)
]

DISTRIBUTIONS = [
    "normal",
    "t"
]


# ============================================================
# 3. LOAD ARIMA RESIDUALS
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"ARIMA residual file not found:\n{INPUT_FILE}"
    )

residual_df = pd.read_csv(
    INPUT_FILE
)

print("=" * 80)
print("STEP 08B: GARCH MODEL SELECTION")
print("=" * 80)

print(
    f"\nRows loaded: {len(residual_df):,}"
)


# ============================================================
# 4. VALIDATE REQUIRED COLUMNS
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
# 5. CLEAN RESIDUALS
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


residuals = residual_df[
    "arima_residual"
].astype(float)


# ============================================================
# 6. BASIC CHECKS
# ============================================================

if len(residuals) < 100:
    raise ValueError(
        "Too few residual observations for GARCH estimation."
    )

if not np.isfinite(residuals).all():
    raise ValueError(
        "Non-finite values found in ARIMA residuals."
    )


print("\nARIMA residual information:")
print(
    f"Start: {residual_df['date'].min().date()}"
)

print(
    f"End:   {residual_df['date'].max().date()}"
)

print(
    f"Observations: {len(residuals):,}"
)

print(
    f"Mean: {residuals.mean():.6f}"
)

print(
    f"Std:  {residuals.std():.6f}"
)

print(
    f"Min:  {residuals.min():.6f}"
)

print(
    f"Max:  {residuals.max():.6f}"
)


# ============================================================
# 7. ARCH-LM TEST BEFORE GARCH
# ============================================================

print("\n" + "=" * 80)
print("ARCH-LM TEST BEFORE GARCH")
print("=" * 80)

arch_test = het_arch(
    residuals,
    nlags=ARCH_LAGS
)

arch_lm_stat = arch_test[0]
arch_lm_pvalue = arch_test[1]
arch_f_stat = arch_test[2]
arch_f_pvalue = arch_test[3]

print(
    f"ARCH-LM statistic: {arch_lm_stat:.6f}"
)

print(
    f"ARCH-LM p-value:   {arch_lm_pvalue:.6f}"
)

print(
    f"F-statistic:       {arch_f_stat:.6f}"
)

print(
    f"F-test p-value:    {arch_f_pvalue:.6f}"
)

if arch_lm_pvalue < 0.05:

    print(
        "\nDecision: reject the null hypothesis of "
        "no ARCH effects at the 5% level."
    )

    print(
        "There is evidence of conditional "
        "heteroskedasticity."
    )

else:

    print(
        "\nDecision: fail to reject the null hypothesis "
        "of no ARCH effects at the 5% level."
    )

    print(
        "There is limited evidence of ARCH effects."
    )


# ============================================================
# 8. FIT CANDIDATE GARCH MODELS
# ============================================================

results = []

print("\n" + "=" * 80)
print("FITTING CANDIDATE GARCH MODELS")
print("=" * 80)


for p, q in GARCH_ORDERS:

    for distribution in DISTRIBUTIONS:

        model_name = (
            f"GARCH({p},{q})-{distribution}"
        )

        print(
            f"\nFitting {model_name}..."
        )

        try:

            # ------------------------------------------------
            # rescale=True allows the arch package to
            # automatically rescale the input if numerical
            # optimization is sensitive to the data scale.
            # ------------------------------------------------

            with warnings.catch_warnings(record=True) as caught_warnings:

                warnings.simplefilter("always")

                model = arch_model(
                    residuals,
                    mean="Zero",
                    vol="GARCH",
                    p=p,
                    q=q,
                    dist=distribution,
                    rescale=True
                )

                fitted = model.fit(
                    update_freq=0,
                    disp="off",
                    show_warning=True,
                    options={
                        "maxiter": 5000
                    }
                )


            # ------------------------------------------------
            # Convergence information
            # ------------------------------------------------

            convergence_flag = (
                fitted.convergence_flag
            )

            converged = (
                convergence_flag == 0
            )

            optimization_message = (
                fitted.optimization_result.message
            )


            warning_messages = [
                str(w.message)
                for w in caught_warnings
            ]


            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

            results.append({
                "model": model_name,
                "p": p,
                "q": q,
                "distribution": distribution,
                "aic": fitted.aic,
                "bic": fitted.bic,
                "loglik": fitted.loglikelihood,
                "convergence_flag": convergence_flag,
                "converged": converged,
                "optimization_message": str(
                    optimization_message
                ),
                "warning": " | ".join(
                    warning_messages
                )
            })


            # ------------------------------------------------
            # Print result
            # ------------------------------------------------

            print(
                f"{model_name}"
            )

            print(
                f"  AIC: {fitted.aic:.6f}"
            )

            print(
                f"  BIC: {fitted.bic:.6f}"
            )

            print(
                f"  LogLik: {fitted.loglikelihood:.6f}"
            )

            print(
                f"  Convergence flag: "
                f"{convergence_flag}"
            )

            print(
                f"  Converged: "
                f"{converged}"
            )

            print(
                f"  Optimizer message: "
                f"{optimization_message}"
            )

            if warning_messages:

                print(
                    "  Warnings:"
                )

                for message in warning_messages:

                    print(
                        f"    - {message}"
                    )


        except Exception as e:

            error_message = str(e)

            print(
                f"{model_name} FAILED"
            )

            print(
                f"  Error: {error_message}"
            )

            results.append({
                "model": model_name,
                "p": p,
                "q": q,
                "distribution": distribution,
                "aic": np.nan,
                "bic": np.nan,
                "loglik": np.nan,
                "convergence_flag": np.nan,
                "converged": False,
                "optimization_message": "",
                "warning": error_message
            })


# ============================================================
# 9. CANDIDATE RESULTS
# ============================================================

candidate_df = pd.DataFrame(
    results
)

candidate_df = candidate_df.sort_values(
    ["aic", "bic"],
    na_position="last"
).reset_index(drop=True)


candidate_df["rank_by_aic"] = (
    candidate_df["aic"].rank(
        method="min",
        ascending=True
    )
)


print("\n" + "=" * 80)
print("GARCH CANDIDATE COMPARISON")
print("=" * 80)

print(
    candidate_df[
        [
            "model",
            "p",
            "q",
            "distribution",
            "aic",
            "bic",
            "loglik",
            "convergence_flag",
            "converged",
            "rank_by_aic"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 10. SELECT SUCCESSFULLY CONVERGED MODEL
# ============================================================

valid_candidates = candidate_df[
    (candidate_df["converged"] == True)
    & candidate_df["aic"].notna()
].copy()


if valid_candidates.empty:

    print("\n" + "=" * 80)
    print("NO SUCCESSFUL GARCH MODEL")
    print("=" * 80)

    print(
        "\nEvery candidate model failed to report "
        "successful convergence."
    )

    print(
        "\nOptimizer details:"
    )

    print(
        candidate_df[
            [
                "model",
                "convergence_flag",
                "optimization_message",
                "warning"
            ]
        ].to_string(
            index=False
        )
    )

    raise RuntimeError(
        "No GARCH candidate model converged successfully. "
        "Review the optimizer messages printed above."
    )


# ============================================================
# 11. SELECT LOWEST AIC
# ============================================================

best = valid_candidates.iloc[0]

best_p = int(
    best["p"]
)

best_q = int(
    best["q"]
)

best_distribution = (
    best["distribution"]
)


print("\n" + "=" * 80)
print("SELECTED GARCH MODEL")
print("=" * 80)

print(
    f"Selected specification: "
    f"GARCH({best_p},{best_q})"
)

print(
    f"Innovation distribution: "
    f"{best_distribution}"
)

print(
    f"AIC: {best['aic']:.6f}"
)

print(
    f"BIC: {best['bic']:.6f}"
)

print(
    f"Log-likelihood: "
    f"{best['loglik']:.6f}"
)


# ============================================================
# 12. REFIT SELECTED MODEL
# ============================================================

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    selected_model = arch_model(
        residuals,
        mean="Zero",
        vol="GARCH",
        p=best_p,
        q=best_q,
        dist=best_distribution,
        rescale=True
    )

    selected_fit = selected_model.fit(
        update_freq=0,
        disp="off",
        options={
            "maxiter": 5000
        }
    )


# ============================================================
# 13. SELECTED MODEL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("SELECTED GARCH MODEL SUMMARY")
print("=" * 80)

print(
    selected_fit.summary()
)


# ============================================================
# 14. STANDARDIZED RESIDUALS
# ============================================================

standardized_residuals = pd.Series(
    selected_fit.std_resid,
    index=residual_df["date"].values
).dropna()

squared_standardized_residuals = (
    standardized_residuals ** 2
)


# ============================================================
# 15. LJUNG-BOX TESTS
# ============================================================

print("\n" + "=" * 80)
print("POST-GARCH RESIDUAL DIAGNOSTICS")
print("=" * 80)

lb_standardized = acorr_ljungbox(
    standardized_residuals,
    lags=[10, 20],
    return_df=True
)

lb_squared = acorr_ljungbox(
    squared_standardized_residuals,
    lags=[10, 20],
    return_df=True
)


print(
    "\nLjung-Box test: standardized residuals"
)

print(
    lb_standardized.to_string()
)


print(
    "\nLjung-Box test: squared standardized residuals"
)

print(
    lb_squared.to_string()
)


# ============================================================
# 16. POST-GARCH ARCH-LM
# ============================================================

post_arch_test = het_arch(
    standardized_residuals,
    nlags=ARCH_LAGS
)

post_arch_lm_stat = post_arch_test[0]
post_arch_lm_pvalue = post_arch_test[1]
post_arch_f_stat = post_arch_test[2]
post_arch_f_pvalue = post_arch_test[3]


print(
    "\nARCH-LM test after GARCH"
)

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
# 17. DIAGNOSTIC INTERPRETATION
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
        "no significant remaining serial correlation "
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
        "no significant remaining volatility dependence "
        "at lags 10 and 20."
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

conditional_volatility = pd.Series(
    selected_fit.conditional_volatility,
    index=residual_df["date"].values,
    name="conditional_volatility"
)


volatility_df = pd.DataFrame({
    "date": conditional_volatility.index,
    "conditional_volatility": conditional_volatility.values
})


# ============================================================
# 19. SAVE CANDIDATE MODELS
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

candidate_df.to_csv(
    CANDIDATE_OUTPUT,
    index=False
)


# ============================================================
# 20. SAVE SELECTED MODEL
# ============================================================

selected_params = selected_fit.params


selected_model_df = pd.DataFrame({
    "model": [
        f"GARCH({best_p},{best_q})"
    ],

    "distribution": [
        best_distribution
    ],

    "p": [
        best_p
    ],

    "q": [
        best_q
    ],

    "aic": [
        selected_fit.aic
    ],

    "bic": [
        selected_fit.bic
    ],

    "loglik": [
        selected_fit.loglikelihood
    ],

    "convergence_flag": [
        selected_fit.convergence_flag
    ],

    "arch_lm_pvalue_before": [
        arch_lm_pvalue
    ],

    "ljung_box_std_pvalue_10": [
        lb_std_p10
    ],

    "ljung_box_std_pvalue_20": [
        lb_std_p20
    ],

    "ljung_box_squared_std_pvalue_10": [
        lb_sq_p10
    ],

    "ljung_box_squared_std_pvalue_20": [
        lb_sq_p20
    ],

    "arch_lm_pvalue_after": [
        post_arch_lm_pvalue
    ]
})


# Add model parameters
for parameter_name, parameter_value in selected_params.items():

    selected_model_df[
        f"parameter_{parameter_name}"
    ] = parameter_value


selected_model_df.to_csv(
    SELECTED_OUTPUT,
    index=False
)


# ============================================================
# 21. SAVE DIAGNOSTICS
# ============================================================

diagnostic_df = pd.DataFrame({
    "diagnostic": [
        "ARCH-LM before GARCH",
        "Ljung-Box standardized residuals lag 10",
        "Ljung-Box standardized residuals lag 20",
        "Ljung-Box squared standardized residuals lag 10",
        "Ljung-Box squared standardized residuals lag 20",
        "ARCH-LM after GARCH"
    ],

    "statistic": [
        arch_lm_stat,
        lb_standardized.loc[10, "lb_stat"],
        lb_standardized.loc[20, "lb_stat"],
        lb_squared.loc[10, "lb_stat"],
        lb_squared.loc[20, "lb_stat"],
        post_arch_lm_stat
    ],

    "p_value": [
        arch_lm_pvalue,
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
# 22. SAVE CONDITIONAL VOLATILITY
# ============================================================

volatility_df.to_csv(
    VOLATILITY_OUTPUT,
    index=False
)


# ============================================================
# 23. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    f"Candidate models: "
    f"{CANDIDATE_OUTPUT}"
)

print(
    f"Selected model:   "
    f"{SELECTED_OUTPUT}"
)

print(
    f"Diagnostics:      "
    f"{DIAGNOSTIC_OUTPUT}"
)

print(
    f"Conditional vol:  "
    f"{VOLATILITY_OUTPUT}"
)

print("=" * 80)