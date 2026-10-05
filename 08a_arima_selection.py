#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 18:20:16 2026

@author: setup
"""

import pandas as pd
import numpy as np
import os

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox


# ============================================================
# 1. LOAD FORECASTING DATASET
# ============================================================

input_file = (
    "data_cleaned/forecasting_dataset_2006_2025.csv"
)

df = pd.read_csv(
    input_file,
    parse_dates=["date", "target_date"]
)


# ============================================================
# 2. SELECT TRAINING DATA ONLY
# ============================================================

train_df = df[
    df["dataset"] == "train"
].copy()


# FX return series
returns = (
    train_df["fx_log_return"]
    .dropna()
    .astype(float)
)

returns.index = pd.RangeIndex(len(returns))


print("\nTraining observations:")
print(len(returns))

print("\nTraining return period:")
print(
    train_df["date"].min().date(),
    "→",
    train_df["date"].max().date()
)


# ============================================================
# 3. STATIONARITY CHECK
# ============================================================

adf_result = adfuller(
    returns,
    autolag="AIC"
)

adf_statistic = adf_result[0]
adf_pvalue = adf_result[1]

print("\nADF TEST")
print("--------------------------------")
print("ADF statistic:", adf_statistic)
print("p-value:", adf_pvalue)

if adf_pvalue < 0.05:
    print(
        "Interpretation: reject the unit-root null; "
        "the return series is consistent with stationarity."
    )
else:
    print(
        "Interpretation: fail to reject the unit-root null; "
        "further differencing may need consideration."
    )


# ============================================================
# 4. SEARCH ARIMA(p,0,q) MODELS
# ============================================================

results = []

p_values = range(0, 4)
q_values = range(0, 4)

print("\nARIMA MODEL SEARCH")
print("--------------------------------")

for p in p_values:

    for q in q_values:

        try:

            model = ARIMA(
                returns,
                order=(p, 0, q),
                trend="c"
            )

            fitted = model.fit()

            results.append({
                "p": p,
                "d": 0,
                "q": q,
                "AIC": fitted.aic,
                "BIC": fitted.bic,
                "LogLik": fitted.llf
            })

            print(
                f"ARIMA({p},0,{q}) "
                f"AIC={fitted.aic:.4f} "
                f"BIC={fitted.bic:.4f}"
            )

        except Exception as e:

            print(
                f"ARIMA({p},0,{q}) failed: {e}"
            )


# ============================================================
# 5. RANK MODELS BY AIC
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "AIC"
).reset_index(drop=True)


print("\nARIMA MODELS RANKED BY AIC")
print("--------------------------------")

print(
    results_df.to_string(index=False)
)


# ============================================================
# 6. SELECT BEST MODEL
# ============================================================

best_row = results_df.iloc[0]

best_p = int(best_row["p"])
best_d = int(best_row["d"])
best_q = int(best_row["q"])

best_order = (
    best_p,
    best_d,
    best_q
)


print("\nSELECTED ARIMA MODEL")
print("--------------------------------")

print(
    f"ARIMA{best_order}"
)

print(
    "AIC:",
    best_row["AIC"]
)

print(
    "BIC:",
    best_row["BIC"]
)


# ============================================================
# 7. FIT SELECTED ARIMA
# ============================================================

selected_model = ARIMA(
    returns,
    order=best_order,
    trend="c"
)

selected_fit = selected_model.fit()


print("\nSELECTED ARIMA SUMMARY")
print("--------------------------------")

print(
    selected_fit.summary()
)


# ============================================================
# 8. RESIDUAL DIAGNOSTICS
# ============================================================

residuals = selected_fit.resid.dropna()


# Ljung-Box test
ljung_box = acorr_ljungbox(
    residuals,
    lags=[10, 20],
    return_df=True
)


print("\nLJUNG-BOX TEST ON ARIMA RESIDUALS")
print("--------------------------------")

print(ljung_box)


# ============================================================
# 9. SAVE RESULTS
# ============================================================

os.makedirs("results", exist_ok=True)

results_file = (
    "results/arima_model_selection.csv"
)

results_df.to_csv(
    results_file,
    index=False
)


selected_info_file = (
    "results/selected_arima_order.csv"
)

pd.DataFrame({
    "p": [best_p],
    "d": [best_d],
    "q": [best_q],
    "AIC": [best_row["AIC"]],
    "BIC": [best_row["BIC"]],
    "LogLik": [best_row["LogLik"]],
    "ADF_statistic": [adf_statistic],
    "ADF_pvalue": [adf_pvalue]
}).to_csv(
    selected_info_file,
    index=False
)


print("\nSaved:")
print(results_file)
print(selected_info_file)