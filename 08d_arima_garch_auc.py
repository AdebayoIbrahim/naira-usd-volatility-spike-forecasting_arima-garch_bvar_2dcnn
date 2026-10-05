#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 20:25:11 2026

@author: setup
"""
import pandas as pd
from sklearn.metrics import roc_auc_score


# ============================================================
# 1. LOAD ARIMA-GARCH FORECASTS
# ============================================================

input_file = (
    "results/arima_garch_test_forecasts.csv"
)

df = pd.read_csv(input_file)


# ============================================================
# 2. CALCULATE AUC
# ============================================================
#
# predicted_volatility is used as the continuous score.
# Higher predicted volatility = stronger evidence of a spike.

auc = roc_auc_score(
    df["actual_spike"],
    df["predicted_volatility"]
)


print("\nARIMA-GARCH AUC")
print("--------------------------------")
print("ROC-AUC:", auc)


# ============================================================
# 3. ADD AUC TO EXISTING METRICS
# ============================================================

metrics_file = (
    "results/arima_garch_test_metrics.csv"
)

metrics = pd.read_csv(metrics_file)

metrics["AUC"] = auc

metrics.to_csv(
    metrics_file,
    index=False
)


print("\nUpdated metrics saved:")
print(metrics_file)
