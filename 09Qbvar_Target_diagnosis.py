#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 19:45:26 2026

@author: setup
"""

import pandas as pd
import numpy as np

# ============================================================
# STEP 09Q: BVAR TARGET VS FORECAST DIAGNOSTIC
# ============================================================

INPUT_FILE = "results/bvar8_final_lambda01_common_sample_forecasts.csv"

df = pd.read_csv(INPUT_FILE)

print("=" * 90)
print("STEP 09Q: BVAR TARGET VS FORECAST DIAGNOSTIC")
print("=" * 90)

# ------------------------------------------------------------
# 1. BASIC INFORMATION
# ------------------------------------------------------------

print(f"\nForecast observations: {len(df)}")

required_cols = [
    "actual_volatility",
    "forecast_volatility",
    "actual_spike"
]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")


# ------------------------------------------------------------
# 2. OVERALL TARGET VS FORECAST
# ------------------------------------------------------------

actual = df["actual_volatility"].astype(float)
forecast = df["forecast_volatility"].astype(float)

print("\n" + "=" * 90)
print("1. OVERALL TARGET VS FORECAST")
print("=" * 90)

print(f"Mean actual volatility:    {actual.mean():.6f}")
print(f"Mean forecast volatility:  {forecast.mean():.6f}")

print(f"\nMedian actual volatility:  {actual.median():.6f}")
print(f"Median forecast volatility:{forecast.median():.6f}")

print(f"\nStd actual volatility:      {actual.std():.6f}")
print(f"Std forecast volatility:   {forecast.std():.6f}")

print(f"\nMinimum actual volatility:  {actual.min():.6f}")
print(f"Minimum forecast volatility:{forecast.min():.6f}")

print(f"\nMaximum actual volatility:  {actual.max():.6f}")
print(f"Maximum forecast volatility:{forecast.max():.6f}")


# ------------------------------------------------------------
# 3. FORECAST BIAS
# ------------------------------------------------------------

errors = forecast - actual

print("\n" + "=" * 90)
print("2. FORECAST BIAS")
print("=" * 90)

print(f"Mean error (Forecast - Actual): {errors.mean():.6f}")
print(f"Median error:                    {errors.median():.6f}")

positive_errors = (errors > 0).sum()
negative_errors = (errors < 0).sum()
zero_errors = (errors == 0).sum()

print(f"\nForecast > Actual: {positive_errors}")
print(f"Forecast < Actual: {negative_errors}")
print(f"Forecast = Actual: {zero_errors}")

print(
    f"\nPercentage overpredicted: "
    f"{positive_errors / len(df):.4%}"
)

print(
    f"Percentage underpredicted: "
    f"{negative_errors / len(df):.4%}"
)


# ------------------------------------------------------------
# 4. CORRELATION
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("3. ACTUAL VS FORECAST CORRELATION")
print("=" * 90)

pearson_corr = actual.corr(forecast, method="pearson")
spearman_corr = actual.corr(forecast, method="spearman")

print(f"Pearson correlation:  {pearson_corr:.6f}")
print(f"Spearman correlation: {spearman_corr:.6f}")


# ------------------------------------------------------------
# 5. PERFORMANCE BY ACTUAL SPIKE STATUS
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("4. ERROR BY ACTUAL SPIKE STATUS")
print("=" * 90)

df["error"] = df["forecast_volatility"] - df["actual_volatility"]
df["absolute_error"] = df["error"].abs()
df["squared_error"] = df["error"] ** 2

grouped = df.groupby("actual_spike").agg(
    observations=("actual_spike", "size"),
    mean_actual=("actual_volatility", "mean"),
    median_actual=("actual_volatility", "median"),
    mean_forecast=("forecast_volatility", "mean"),
    median_forecast=("forecast_volatility", "median"),
    mean_error=("error", "mean"),
    mean_absolute_error=("absolute_error", "mean"),
    rmse=("squared_error", lambda x: np.sqrt(x.mean()))
)

print(grouped)


# ------------------------------------------------------------
# 6. FORECAST/ACTUAL RATIO
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("5. FORECAST / ACTUAL RATIO")
print("=" * 90)

# Only calculate where actual > 0
positive_actual = actual > 0

ratio = forecast[positive_actual] / actual[positive_actual]

print(f"Mean ratio:   {ratio.mean():.6f}")
print(f"Median ratio: {ratio.median():.6f}")
print(f"25th pct:     {ratio.quantile(0.25):.6f}")
print(f"75th pct:     {ratio.quantile(0.75):.6f}")
print(f"95th pct:     {ratio.quantile(0.95):.6f}")


# ------------------------------------------------------------
# 7. CHECK ZERO / LOW ACTUAL VOLATILITY DAYS
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("6. LOW-VOLATILITY OBSERVATIONS")
print("=" * 90)

zero_actual = actual == 0
low_actual = actual <= 0.01

print(f"Actual volatility = 0:     {zero_actual.sum()}")
print(f"Actual volatility <= 0.01: {low_actual.sum()}")

if zero_actual.sum() > 0:
    print(
        f"\nMean forecast when actual = 0: "
        f"{forecast[zero_actual].mean():.6f}"
    )

if low_actual.sum() > 0:
    print(
        f"Mean forecast when actual <= 0.01: "
        f"{forecast[low_actual].mean():.6f}"
    )


# ------------------------------------------------------------
# 8. SPIKE THRESHOLD CHECK
# ------------------------------------------------------------

threshold = df["spike_threshold"].iloc[0]

print("\n" + "=" * 90)
print("7. THRESHOLD CHECK")
print("=" * 90)

print(f"Spike threshold: {threshold:.8f}")

print(
    f"Mean actual volatility / threshold: "
    f"{actual.mean() / threshold:.6f}"
)

print(
    f"Mean forecast volatility / threshold: "
    f"{forecast.mean() / threshold:.6f}"
)

print(
    f"Median actual volatility / threshold: "
    f"{actual.median() / threshold:.6f}"
)

print(
    f"Median forecast volatility / threshold: "
    f"{forecast.median() / threshold:.6f}"
)


# ------------------------------------------------------------
# 9. HOW OFTEN DOES FORECAST EXCEED THRESHOLD?
# ------------------------------------------------------------

forecast_above_threshold = forecast > threshold
actual_above_threshold = actual > threshold

print("\n" + "=" * 90)
print("8. THRESHOLD EXCEEDANCE")
print("=" * 90)

print(
    f"Actual above threshold:   "
    f"{actual_above_threshold.sum()} "
    f"({actual_above_threshold.mean():.4%})"
)

print(
    f"Forecast above threshold: "
    f"{forecast_above_threshold.sum()} "
    f"({forecast_above_threshold.mean():.4%})"
)


# ------------------------------------------------------------
# 10. CHECK WHETHER HIGH FORECASTS CORRESPOND TO HIGH ACTUALS
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("9. TOP 20 FORECASTS")
print("=" * 90)

top_forecasts = df[
    [
        "origin_date",
        "target_date",
        "actual_volatility",
        "forecast_volatility",
        "actual_spike"
    ]
].sort_values(
    "forecast_volatility",
    ascending=False
).head(20)

print(top_forecasts.to_string(index=False))


# ------------------------------------------------------------
# 11. TOP 20 ACTUAL VOLATILITY VALUES
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("10. TOP 20 ACTUAL VOLATILITY VALUES")
print("=" * 90)

top_actual = df[
    [
        "origin_date",
        "target_date",
        "actual_volatility",
        "forecast_volatility",
        "actual_spike"
    ]
].sort_values(
    "actual_volatility",
    ascending=False
).head(20)

print(top_actual.to_string(index=False))


# ------------------------------------------------------------
# 12. FINAL INTERPRETATION FLAGS
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("11. AUTOMATIC DIAGNOSTIC FLAGS")
print("=" * 90)

if forecast.mean() > actual.mean():
    print("⚠ Mean forecast is higher than mean actual → possible upward bias.")
else:
    print("✓ Mean forecast is not higher than mean actual.")

if pearson_corr < 0.20:
    print("⚠ Very weak linear association between forecast and actual.")
elif pearson_corr < 0.50:
    print("⚠ Moderate/weak association between forecast and actual.")
else:
    print("✓ Meaningful linear association detected.")

if forecast_above_threshold.mean() > actual_above_threshold.mean():
    print("⚠ Forecast exceeds spike threshold much more often than actual target.")
else:
    print("✓ Forecast threshold exceedance is not substantially higher than actual.")

print("\n" + "=" * 90)
print("STEP 09Q COMPLETE")
print("=" * 90)