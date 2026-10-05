#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 18:56:12 2026

@author: setup
"""

import pandas as pd
import numpy as np
import os

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_arch
)

from arch import arch_model


# ============================================================
# 1. LOAD FORECASTING DATA
# ============================================================

input_file = (
    "data_cleaned/forecasting_dataset_2006_2025.csv"
)

df = pd.read_csv(
    input_file,
    parse_dates=["date", "target_date"]
)


# ============================================================
# 2. TRAINING DATA ONLY
# ============================================================

train_df = df[
    df["dataset"] == "train"
].copy()

returns = (
    train_df["fx_log_return"]
    .dropna()
    .astype(float)
)

returns.index = pd.RangeIndex(len(returns))


print("\nTraining observations:")
print(len(returns))


# ============================================================
# 3. FIT SELECTED ARIMA MODEL
# ============================================================
# Step 08A selected ARIMA(0,0,1)

arima_order = (0, 0, 1)

arima_model = ARIMA(
    returns,
    order=arima_order,
    trend="c"
)

arima_fit = arima_model.fit()


# ============================================================
# 4. EXTRACT ARIMA RESIDUALS
# ============================================================

residuals = (
    arima_fit.resid
    .dropna()
    .astype(float)
)

residuals.index = pd.RangeIndex(len(residuals))


print("\nARIMA residual observations:")
print(len(residuals))


# ============================================================
# 5. PRE-GARCH ARCH EFFECT TEST
# ============================================================

arch_test = het_arch(
    residuals,
    nlags=10
)

print("\nARCH EFFECT TEST BEFORE GARCH")
print("--------------------------------")
print("LM statistic:", arch_test[0])
print("LM p-value:", arch_test[1])


# ============================================================
# 6. TEST GARCH SPECIFICATIONS
# ============================================================
#
# We compare small parsimonious models:
#
# GARCH(1,1)
# GARCH(1,2)
# GARCH(2,1)
# GARCH(2,2)
#
# Student-t innovations are used.

garch_results = []

candidate_orders = [
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2)
]


print("\nGARCH MODEL SEARCH")
print("--------------------------------")


for p, q in candidate_orders:

    try:

        model = arch_model(
            residuals,
            mean="Zero",
            vol="GARCH",
            p=p,
            q=q,
            dist="t",
            rescale=False
        )

        fitted = model.fit(
            disp="off"
        )

        garch_results.append({
            "p": p,
            "q": q,
            "distribution": "Student-t",
            "AIC": fitted.aic,
            "BIC": fitted.bic,
            "LogLik": fitted.loglikelihood
        })

        print(
            f"GARCH({p},{q}) "
            f"AIC={fitted.aic:.4f} "
            f"BIC={fitted.bic:.4f}"
        )

    except Exception as e:

        print(
            f"GARCH({p},{q}) failed: {e}"
        )


# ============================================================
# 7. RANK GARCH MODELS
# ============================================================

garch_results_df = pd.DataFrame(
    garch_results
)

garch_results_df = (
    garch_results_df
    .sort_values("AIC")
    .reset_index(drop=True)
)


print("\nGARCH MODELS RANKED BY AIC")
print("--------------------------------")

print(
    garch_results_df.to_string(index=False)
)


# ============================================================
# 8. SELECT LOWEST-AIC GARCH
# ============================================================

best_garch = garch_results_df.iloc[0]

best_p = int(best_garch["p"])
best_q = int(best_garch["q"])


print("\nSELECTED GARCH MODEL")
print("--------------------------------")

print(
    f"GARCH({best_p},{best_q})"
)

print(
    "AIC:",
    best_garch["AIC"]
)

print(
    "BIC:",
    best_garch["BIC"]
)


# ============================================================
# 9. FIT SELECTED GARCH
# ============================================================

selected_garch = arch_model(
    residuals,
    mean="Zero",
    vol="GARCH",
    p=best_p,
    q=best_q,
    dist="t",
    rescale=False
)

garch_fit = selected_garch.fit(
    disp="off"
)


print("\nSELECTED GARCH SUMMARY")
print("--------------------------------")

print(
    garch_fit.summary()
)


# ============================================================
# 10. CONDITIONAL VOLATILITY
# ============================================================

conditional_volatility = (
    garch_fit.conditional_volatility
)

print("\nConditional volatility summary:")
print(
    conditional_volatility.describe()
)


# ============================================================
# 11. STANDARDIZED RESIDUALS
# ============================================================

standardized_residuals = (
    residuals / conditional_volatility
).dropna()


# ============================================================
# 12. LJUNG-BOX ON STANDARDIZED RESIDUALS
# ============================================================

lb_standardized = acorr_ljungbox(
    standardized_residuals,
    lags=[10, 20],
    return_df=True
)


print("\nLJUNG-BOX ON STANDARDIZED RESIDUALS")
print("--------------------------------")

print(
    lb_standardized
)


# ============================================================
# 13. LJUNG-BOX ON SQUARED STANDARDIZED RESIDUALS
# ============================================================

lb_squared = acorr_ljungbox(
    standardized_residuals ** 2,
    lags=[10, 20],
    return_df=True
)


print("\nLJUNG-BOX ON SQUARED STANDARDIZED RESIDUALS")
print("--------------------------------")

print(
    lb_squared
)


# ============================================================
# 14. ARCH EFFECT TEST AFTER GARCH
# ============================================================

arch_test_after = het_arch(
    standardized_residuals,
    nlags=10
)


print("\nARCH EFFECT TEST AFTER GARCH")
print("--------------------------------")

print(
    "LM statistic:",
    arch_test_after[0]
)

print(
    "LM p-value:",
    arch_test_after[1]
)


# ============================================================
# 15. SAVE GARCH MODEL SELECTION
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


garch_selection_file = (
    "results/garch_model_selection.csv"
)

garch_results_df.to_csv(
    garch_selection_file,
    index=False
)


# ============================================================
# 16. SAVE SELECTED MODEL INFORMATION
# ============================================================

selected_model_file = (
    "results/selected_garch_model.csv"
)

pd.DataFrame({
    "p": [best_p],
    "q": [best_q],
    "distribution": ["Student-t"],
    "AIC": [best_garch["AIC"]],
    "BIC": [best_garch["BIC"]],
    "LogLik": [best_garch["LogLik"]],
    "pre_garch_arch_pvalue": [arch_test[1]],
    "post_garch_arch_pvalue": [arch_test_after[1]]
}).to_csv(
    selected_model_file,
    index=False
)


print("\nSaved:")
print(garch_selection_file)
print(selected_model_file)


print("\nStep 08B completed.")