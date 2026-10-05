#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 22:46:27 2026

@author: setup
"""
# ============================================================
# 02_daily_data_diagnostics.py
#
# Project:
# Forecasting Volatility Spikes in the Nigerian Naira–US Dollar
# Exchange Rate Using ARIMA-GARCH, Bayesian VAR, and
# Multivariate 2D Convolutional Neural Network Models
#
# Purpose:
# Diagnose the merged daily dataset before transformations
# or modelling.
# ============================================================


# ============================================================
# 1. Imports
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 2. Project directories
# ============================================================

BASE_DIR = Path(".")

CLEANED_DIR = BASE_DIR / "data_cleaned"
RESULTS_DIR = BASE_DIR / "results"

CLEANED_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# 3. Load merged daily dataset
# ============================================================

merged_file = (
    CLEANED_DIR /
    "merged_daily_macro_dataset_2006_2025.csv"
)

df = pd.read_csv(merged_file)

df["date"] = pd.to_datetime(df["date"])

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)


# ============================================================
# 4. Basic dataset information
# ============================================================

print("\n" + "=" * 75)
print("DAILY DATASET DIAGNOSTICS")
print("=" * 75)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nDate range:")
print(df["date"].min(), "to", df["date"].max())

print("\nNumber of observations:")
print(len(df))


# ============================================================
# 5. Missing values
# ============================================================

print("\n" + "=" * 75)
print("MISSING VALUE SUMMARY")
print("=" * 75)

missing_summary = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percentage": (
        df.isna().mean() * 100
    )
})

missing_summary["missing_percentage"] = (
    missing_summary["missing_percentage"]
    .round(4)
)

print(missing_summary)


# Save missing-value report
missing_summary_path = (
    RESULTS_DIR /
    "daily_missing_value_summary.csv"
)

missing_summary.to_csv(
    missing_summary_path
)


# ============================================================
# 6. Duplicate-date check
# ============================================================

duplicate_dates = df["date"].duplicated().sum()

print("\nDuplicate dates:")
print(duplicate_dates)


# ============================================================
# 7. Observation-gap diagnostics
# ============================================================

df["gap_days"] = (
    df["date"]
    .diff()
    .dt.days
)

print("\n" + "=" * 75)
print("OBSERVATION GAP DIAGNOSTICS")
print("=" * 75)

print("\nGap distribution:")
print(
    df["gap_days"]
    .value_counts()
    .sort_index()
)

print("\nNumber of gaps > 1 day:")
print(
    (df["gap_days"] > 1).sum()
)

print("\nNumber of gaps > 3 days:")
print(
    (df["gap_days"] > 3).sum()
)

print("\nNumber of gaps > 5 days:")
print(
    (df["gap_days"] > 5).sum()
)

print("\nLargest observation gap:")
print(
    df["gap_days"].max(),
    "days"
)

large_gap_df = df.loc[
    df["gap_days"] > 5,
    ["date", "gap_days"]
].copy()

print("\nGaps greater than 5 days:")
print(
    large_gap_df.to_string(index=False)
)

large_gap_path = (
    RESULTS_DIR /
    "daily_large_observation_gaps.csv"
)

large_gap_df.to_csv(
    large_gap_path,
    index=False
)


# ============================================================
# 8. Exchange-rate comparison
# ============================================================

print("\n" + "=" * 75)
print("EXCHANGE-RATE MEASURES")
print("=" * 75)

exchange_cols = [
    "buying_rate",
    "central_rate",
    "selling_rate"
]

exchange_description = (
    df[exchange_cols]
    .describe()
    .T
)

print(
    exchange_description
)

# Pairwise correlations
exchange_correlation = (
    df[exchange_cols]
    .corr()
)

print("\nExchange-rate correlation matrix:")
print(exchange_correlation)


# ============================================================
# 9. Check differences between exchange-rate measures
# ============================================================

df["central_minus_buying"] = (
    df["central_rate"] -
    df["buying_rate"]
)

df["selling_minus_central"] = (
    df["selling_rate"] -
    df["central_rate"]
)

df["selling_minus_buying"] = (
    df["selling_rate"] -
    df["buying_rate"]
)

print("\nExchange-rate spread summary:")
print(
    df[
        [
            "central_minus_buying",
            "selling_minus_central",
            "selling_minus_buying"
        ]
    ].describe()
)


# ============================================================
# 10. External reserves comparison
# ============================================================

print("\n" + "=" * 75)
print("EXTERNAL RESERVE MEASURES")
print("=" * 75)

reserve_cols = [
    "gross_reserves",
    "liquid_reserves",
    "blocked_reserves",
    "blocked_percent"
]

reserve_description = (
    df[reserve_cols]
    .describe()
    .T
)

print(
    reserve_description
)

reserve_correlation = (
    df[reserve_cols]
    .corr()
)

print("\nExternal-reserve correlation matrix:")
print(reserve_correlation)


# ============================================================
# 11. Check relationship between Gross and Liquid reserves
# ============================================================

df["liquid_to_gross_ratio"] = (
    df["liquid_reserves"] /
    df["gross_reserves"]
)

df["blocked_to_gross_ratio"] = (
    df["blocked_reserves"] /
    df["gross_reserves"]
)

print("\nReserve composition summary:")
print(
    df[
        [
            "liquid_to_gross_ratio",
            "blocked_to_gross_ratio",
            "blocked_percent"
        ]
    ].describe()
)


# ============================================================
# 12. Main modelling-variable descriptive statistics
# ============================================================

main_variables = [
    "buying_rate",
    "central_rate",
    "selling_rate",
    "brent_price",
    "gross_reserves",
    "liquid_reserves",
    "blocked_reserves"
]

main_descriptive_stats = (
    df[main_variables]
    .describe()
    .T
)

print("\n" + "=" * 75)
print("DESCRIPTIVE STATISTICS")
print("=" * 75)

print(main_descriptive_stats)


# ============================================================
# 13. Quantile-based extreme value check
# ============================================================

print("\n" + "=" * 75)
print("EXTREME VALUE CHECK")
print("=" * 75)

for column in [
    "buying_rate",
    "brent_price",
    "gross_reserves"
]:

    q1 = df[column].quantile(0.01)
    q99 = df[column].quantile(0.99)

    print(f"\n{column}")
    print("1st percentile :", q1)
    print("99th percentile:", q99)

    extreme_count = (
        (df[column] < q1) |
        (df[column] > q99)
    ).sum()

    print(
        "Observations outside 1%-99% range:",
        extreme_count
    )


# ============================================================
# 14. Correlation matrix for preliminary variables
# ============================================================

correlation_cols = [
    "buying_rate",
    "brent_price",
    "gross_reserves"
]

preliminary_correlation = (
    df[correlation_cols]
    .corr()
)

print("\n" + "=" * 75)
print("PRELIMINARY CORRELATION MATRIX")
print("=" * 75)

print(
    preliminary_correlation
)


# ============================================================
# 15. Check Brent missing dates
# ============================================================

missing_brent_dates = df.loc[
    df["brent_price"].isna(),
    "date"
].copy()

print("\n" + "=" * 75)
print("BRENT MISSING-DATE DIAGNOSTICS")
print("=" * 75)

print("\nNumber of missing Brent observations:")
print(len(missing_brent_dates))

if len(missing_brent_dates) > 0:

    print("\nMissing Brent by day of week:")
    print(
        missing_brent_dates
        .dt.day_name()
        .value_counts()
    )

    print("\nMissing Brent by year:")
    print(
        missing_brent_dates
        .dt.year
        .value_counts()
        .sort_index()
    )


# ============================================================
# 16. Save detailed diagnostic tables
# ============================================================

exchange_stats_path = (
    RESULTS_DIR /
    "exchange_rate_diagnostics.csv"
)

reserve_stats_path = (
    RESULTS_DIR /
    "external_reserves_diagnostics.csv"
)

descriptive_stats_path = (
    RESULTS_DIR /
    "daily_descriptive_statistics.csv"
)

correlation_path = (
    RESULTS_DIR /
    "daily_preliminary_correlation_matrix.csv"
)

exchange_description.to_csv(
    exchange_stats_path
)

reserve_description.to_csv(
    reserve_stats_path
)

main_descriptive_stats.to_csv(
    descriptive_stats_path
)

preliminary_correlation.to_csv(
    correlation_path
)


# ============================================================
# 17. Print preliminary modelling decision information
# ============================================================

print("\n" + "=" * 75)
print("PRELIMINARY MODELLING CANDIDATES")
print("=" * 75)

print("""
Exchange-rate candidate:
    buying_rate

Primary reserve candidate:
    gross_reserves

Oil-price variable:
    brent_price

These are diagnostic candidates only.
Final transformations and modelling variables will be
decided after stationarity and volatility analysis.
""")


# ============================================================
# 18. Save a diagnostic copy
# ============================================================

diagnostic_copy_path = (
    RESULTS_DIR /
    "daily_dataset_diagnostic_copy.csv"
)

df.to_csv(
    diagnostic_copy_path,
    index=False
)


# ============================================================
# 19. Final output
# ============================================================

print("\n" + "=" * 75)
print("DIAGNOSTIC FILES SAVED")
print("=" * 75)

print("\nMissing-value report:")
print(missing_summary_path)

print("\nLarge-gap report:")
print(large_gap_path)

print("\nExchange-rate diagnostics:")
print(exchange_stats_path)

print("\nReserve diagnostics:")
print(reserve_stats_path)

print("\nDescriptive statistics:")
print(descriptive_stats_path)

print("\nPreliminary correlation matrix:")
print(correlation_path)

print("\nDiagnostic dataset copy:")
print(diagnostic_copy_path)

print("\nDaily data diagnostics completed successfully.")




# ============================================================
# 20. Investigate extreme reserve observations
# ============================================================

print("\n" + "=" * 75)
print("EXTREME EXTERNAL RESERVE OBSERVATIONS")
print("=" * 75)

reserve_extremes = (
    df[
        [
            "date",
            "gross_reserves",
            "liquid_reserves",
            "blocked_reserves",
            "blocked_percent"
        ]
    ]
    .sort_values(
        "gross_reserves",
        ascending=False
    )
    .head(20)
)

print(
    reserve_extremes.to_string(index=False)
)


# ============================================================
# 21. Investigate extreme liquid-reserve observations
# ============================================================

print("\n" + "=" * 75)
print("EXTREME LIQUID RESERVE OBSERVATIONS")
print("=" * 75)

liquid_extremes = (
    df[
        [
            "date",
            "gross_reserves",
            "liquid_reserves",
            "blocked_reserves",
            "blocked_percent"
        ]
    ]
    .sort_values(
        "liquid_reserves",
        ascending=False
    )
    .head(20)
)

print(
    liquid_extremes.to_string(index=False)
)


# ============================================================
# 22. Investigate exchange-rate spread anomalies
# ============================================================

print("\n" + "=" * 75)
print("EXCHANGE-RATE SPREAD ANOMALIES")
print("=" * 75)

spread_extremes = (
    df[
        [
            "date",
            "buying_rate",
            "central_rate",
            "selling_rate",
            "central_minus_buying",
            "selling_minus_central",
            "selling_minus_buying"
        ]
    ]
    .sort_values(
        "selling_minus_buying",
        ascending=False
    )
    .head(20)
)

print(
    spread_extremes.to_string(index=False)
)


# ============================================================
# 23. Check reserve component consistency
# ============================================================

# In principle:
#
# Gross Reserves ≈ Liquid Reserves + Blocked Reserves
#
# We check how large the difference is.

df["reserve_reconstruction_difference"] = (
    df["gross_reserves"]
    - (
        df["liquid_reserves"]
        + df["blocked_reserves"]
    )
)

print("\n" + "=" * 75)
print("RESERVE RECONSTRUCTION CHECK")
print("=" * 75)

print("\nSummary of:")
print("Gross - (Liquid + Blocked)")

print(
    df["reserve_reconstruction_difference"]
    .describe()
)

print("\nLargest absolute differences:")

reserve_balance_check = (
    df[
        [
            "date",
            "gross_reserves",
            "liquid_reserves",
            "blocked_reserves",
            "reserve_reconstruction_difference"
        ]
    ]
    .copy()
)

reserve_balance_check["absolute_difference"] = (
    reserve_balance_check[
        "reserve_reconstruction_difference"
    ].abs()
)

print(
    reserve_balance_check
    .sort_values(
        "absolute_difference",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 24. Check for impossible / suspicious values
# ============================================================

print("\n" + "=" * 75)
print("BASIC LOGICAL CONSISTENCY CHECKS")
print("=" * 75)

print(
    "\nGross reserves <= 0:",
    (df["gross_reserves"] <= 0).sum()
)

print(
    "Liquid reserves <= 0:",
    (df["liquid_reserves"] <= 0).sum()
)

print(
    "Blocked reserves < 0:",
    (df["blocked_reserves"] < 0).sum()
)

print(
    "Liquid reserves > Gross reserves:",
    (
        df["liquid_reserves"]
        > df["gross_reserves"]
    ).sum()
)

print(
    "Blocked reserves > Gross reserves:",
    (
        df["blocked_reserves"]
        > df["gross_reserves"]
    ).sum()
)

print(
    "Blocked percentage outside 0-100:",
    (
        (df["blocked_percent"] < 0) |
        (df["blocked_percent"] > 100)
    ).sum()
)


# ============================================================
# 25. Save anomaly reports
# ============================================================

reserve_extreme_path = (
    RESULTS_DIR /
    "reserve_extreme_observations.csv"
)

spread_extreme_path = (
    RESULTS_DIR /
    "exchange_rate_spread_anomalies.csv"
)

reserve_balance_path = (
    RESULTS_DIR /
    "reserve_reconstruction_check.csv"
)

reserve_extremes.to_csv(
    reserve_extreme_path,
    index=False
)

spread_extremes.to_csv(
    spread_extreme_path,
    index=False
)

reserve_balance_check.to_csv(
    reserve_balance_path,
    index=False
)

print("\nAnomaly reports saved:")
print(reserve_extreme_path)
print(spread_extreme_path)
print(reserve_balance_path)



# ============================================================
# Investigate suspicious reserve values
# ============================================================

print("\n" + "=" * 75)
print("SUSPICIOUS EXTERNAL RESERVE VALUES")
print("=" * 75)

# Values above $100 billion
suspicious_reserves = df[
    df["gross_reserves"] > 100_000_000_000
][
    [
        "date",
        "gross_reserves",
        "liquid_reserves",
        "blocked_reserves",
        "blocked_percent"
    ]
].copy()

print("\nObservations with Gross Reserves > $100 billion:")
print(
    suspicious_reserves.to_string(index=False)
)

print("\nNumber of suspicious observations:")
print(len(suspicious_reserves))


# ============================================================
# Investigate liquid reserves separately
# ============================================================

suspicious_liquid = df[
    df["liquid_reserves"] > 100_000_000_000
][
    [
        "date",
        "gross_reserves",
        "liquid_reserves",
        "blocked_reserves",
        "blocked_percent"
    ]
].copy()

print("\n" + "=" * 75)
print("SUSPICIOUS LIQUID RESERVE VALUES")
print("=" * 75)

print(
    suspicious_liquid.to_string(index=False)
)

print("\nNumber of suspicious liquid-reserve observations:")
print(len(suspicious_liquid))



# ============================================================
# Investigate the two reserve anomalies in detail
# ============================================================

anomaly_dates = [
    pd.Timestamp("2007-11-20"),
    pd.Timestamp("2006-11-07")
]

for anomaly_date in anomaly_dates:

    print("\n" + "=" * 75)
    print(f"RESERVE ANOMALY INVESTIGATION: {anomaly_date.date()}")
    print("=" * 75)

    idx = df.index[df["date"] == anomaly_date][0]

    start = max(0, idx - 5)
    end = min(len(df), idx + 6)

    print(
        df.loc[
            start:end,
            [
                "date",
                "gross_reserves",
                "liquid_reserves",
                "blocked_reserves",
                "blocked_percent"
            ]
        ].to_string(index=False)
    )