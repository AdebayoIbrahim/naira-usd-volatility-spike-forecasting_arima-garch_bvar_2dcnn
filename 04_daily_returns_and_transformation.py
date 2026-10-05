#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 18:14:12 2026

@author: setup
"""

# ============================================================
# 04_daily_returns_and_transformations.py
# Daily Returns and Transformations
# ============================================================

import os
import numpy as np
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

INPUT_FILE = "data_cleaned/analysis_ready_daily_dataset_2006_2025_corrected.csv"

OUTPUT_FILE = (
    "data_cleaned/daily_transformed_dataset_2006_2025.csv"
)

SUMMARY_FILE = (
    "results/daily_transformation_summary.csv"
)

GAP_RETURN_FILE = (
    "results/daily_return_gap_diagnostics.csv"
)

os.makedirs("data_cleaned", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ------------------------------------------------------------
# 2. Load analysis-ready dataset
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

print("=" * 70)
print("DAILY RETURNS AND TRANSFORMATIONS")
print("=" * 70)

print("\nInput shape:", df.shape)
print("Date range:", df["date"].min().date(), "to", df["date"].max().date())


# ------------------------------------------------------------
# 3. Check required columns
# ------------------------------------------------------------

required_columns = [
    "date",
    "buying_rate",
    "brent_price",
    "gross_reserves",
    "gap_days",
    "long_gap"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ------------------------------------------------------------
# 4. Create a clean numeric copy
# ------------------------------------------------------------

numeric_columns = [
    "buying_rate",
    "brent_price",
    "gross_reserves",
    "liquid_reserves",
    "blocked_reserves",
    "blocked_percent",
    "gap_days"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# ------------------------------------------------------------
# 5. Identify structural/long gaps
# ------------------------------------------------------------

# We already defined long_gap during the previous stage.
# For safety, reconstruct it from gap_days.

df["long_gap"] = (
    df["gap_days"].notna() &
    (df["gap_days"] > 5)
).astype(int)


# ------------------------------------------------------------
# 6. Gap-aware log return function
# ------------------------------------------------------------

def gap_aware_log_return(series, gap_days, max_allowed_gap=5):
    """
    Calculate percentage log returns only when the previous
    observation is sufficiently close.

    Return:
        100 * ln(P_t / P_{t-1})

    A return is NOT calculated across a structural/long gap.
    """

    returns = pd.Series(
        np.nan,
        index=series.index,
        dtype="float64"
    )

    for i in range(1, len(series)):

        current_value = series.iloc[i]
        previous_value = series.iloc[i - 1]

        current_gap = gap_days.iloc[i]

        # Cannot calculate if values are missing
        if pd.isna(current_value) or pd.isna(previous_value):
            continue

        # Do not cross long/structural gaps
        if pd.isna(current_gap) or current_gap > max_allowed_gap:
            continue

        # Log requires strictly positive values
        if current_value <= 0 or previous_value <= 0:
            continue

        returns.iloc[i] = (
            100 * np.log(current_value / previous_value)
        )

    return returns


# ------------------------------------------------------------
# 7. Calculate exchange-rate log returns
# ------------------------------------------------------------

df["fx_log_return"] = gap_aware_log_return(
    df["buying_rate"],
    df["gap_days"]
)


# ------------------------------------------------------------
# 8. Calculate Brent oil log returns
# ------------------------------------------------------------

df["brent_log_return"] = gap_aware_log_return(
    df["brent_price"],
    df["gap_days"]
)


# ------------------------------------------------------------
# 9. Calculate external-reserve log changes
# ------------------------------------------------------------

df["reserves_log_change"] = gap_aware_log_return(
    df["gross_reserves"],
    df["gap_days"]
)


# ------------------------------------------------------------
# 10. Create simple percentage changes as supplementary
# ------------------------------------------------------------

def gap_aware_pct_change(series, gap_days, max_allowed_gap=5):

    changes = pd.Series(
        np.nan,
        index=series.index,
        dtype="float64"
    )

    for i in range(1, len(series)):

        current_value = series.iloc[i]
        previous_value = series.iloc[i - 1]

        current_gap = gap_days.iloc[i]

        if pd.isna(current_value) or pd.isna(previous_value):
            continue

        if pd.isna(current_gap) or current_gap > max_allowed_gap:
            continue

        if previous_value == 0:
            continue

        changes.iloc[i] = (
            ((current_value - previous_value) / previous_value)
            * 100
        )

    return changes


df["fx_pct_change"] = gap_aware_pct_change(
    df["buying_rate"],
    df["gap_days"]
)

df["brent_pct_change"] = gap_aware_pct_change(
    df["brent_price"],
    df["gap_days"]
)

df["reserves_pct_change"] = gap_aware_pct_change(
    df["gross_reserves"],
    df["gap_days"]
)


# ------------------------------------------------------------
# 11. Diagnose returns around long gaps
# ------------------------------------------------------------

long_gap_indices = df.index[df["long_gap"] == 1].tolist()

gap_diagnostics = []

for idx in long_gap_indices:

    gap_date = df.loc[idx, "date"]

    row = {
        "gap_date": gap_date,
        "gap_days": df.loc[idx, "gap_days"],
        "buying_rate": df.loc[idx, "buying_rate"],
        "fx_log_return": df.loc[idx, "fx_log_return"],
        "brent_price": df.loc[idx, "brent_price"],
        "brent_log_return": df.loc[idx, "brent_log_return"],
        "gross_reserves": df.loc[idx, "gross_reserves"],
        "reserves_log_change": df.loc[idx, "reserves_log_change"]
    }

    gap_diagnostics.append(row)

gap_diagnostics = pd.DataFrame(gap_diagnostics)

gap_diagnostics.to_csv(
    GAP_RETURN_FILE,
    index=False
)


# ------------------------------------------------------------
# 12. Basic transformation summary
# ------------------------------------------------------------

transformation_columns = [
    "fx_log_return",
    "brent_log_return",
    "reserves_log_change",
    "fx_pct_change",
    "brent_pct_change",
    "reserves_pct_change"
]

summary_rows = []

for col in transformation_columns:

    series = df[col]

    summary_rows.append({
        "variable": col,
        "observations": series.notna().sum(),
        "missing": series.isna().sum(),
        "mean": series.mean(),
        "std": series.std(),
        "minimum": series.min(),
        "maximum": series.max(),
        "median": series.median()
    })

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ------------------------------------------------------------
# 13. Check extreme transformed observations
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("RETURN SUMMARY")
print("-" * 70)

print(summary.to_string(index=False))


# ------------------------------------------------------------
# 14. Check long gaps
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("LONG GAP RETURN CHECK")
print("-" * 70)

if len(gap_diagnostics) > 0:

    print(
        gap_diagnostics[
            [
                "gap_date",
                "gap_days",
                "fx_log_return",
                "brent_log_return",
                "reserves_log_change"
            ]
        ].to_string(index=False)
    )

else:

    print("No long gaps found.")


# ------------------------------------------------------------
# 15. Confirm no returns cross long gaps
# ------------------------------------------------------------

invalid_fx_returns = df.loc[
    (df["long_gap"] == 1) &
    (df["fx_log_return"].notna())
]

invalid_brent_returns = df.loc[
    (df["long_gap"] == 1) &
    (df["brent_log_return"].notna())
]

invalid_reserve_changes = df.loc[
    (df["long_gap"] == 1) &
    (df["reserves_log_change"].notna())
]

print("\n" + "-" * 70)
print("GAP SAFETY CHECK")
print("-" * 70)

print(
    "FX returns crossing long gaps:",
    len(invalid_fx_returns)
)

print(
    "Brent returns crossing long gaps:",
    len(invalid_brent_returns)
)

print(
    "Reserve changes crossing long gaps:",
    len(invalid_reserve_changes)
)

if (
    len(invalid_fx_returns) > 0
    or len(invalid_brent_returns) > 0
    or len(invalid_reserve_changes) > 0
):
    raise ValueError(
        "ERROR: A return/change was calculated across a long gap."
    )

print("Gap safety check PASSED.")


# ------------------------------------------------------------
# 16. Save transformed dataset
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# 17. Final output
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TRANSFORMATION COMPLETED")
print("=" * 70)

print("\nOutput file:")
print(OUTPUT_FILE)

print("\nTransformation summary:")
print(SUMMARY_FILE)

print("\nGap diagnostics:")
print(GAP_RETURN_FILE)

print("\nFinal shape:", df.shape)

print("\nNew transformation columns:")

for col in transformation_columns:
    print(
        f"  {col}: {df[col].notna().sum()} valid observations"
    )


# show in data frame     
showtransformed = pd.read_csv(OUTPUT_FILE)
print(len(showtransformed))    


# 0  log dominance in exchnage rate  check 
zero_count = (df["fx_log_return"] == 0).sum()
valid_count = df["fx_log_return"].notna().sum()

zero_percentage = (zero_count / valid_count) * 100

print("Zero FX returns:", zero_count)
print("Valid FX returns:", valid_count)
print("Percentage of zero returns:", zero_percentage)



unchanged_count = (
    df["buying_rate"].diff() == 0
).sum()

valid_changes = df["buying_rate"].diff().notna().sum()

print("Unchanged exchange-rate observations:", unchanged_count)
print(
    "Percentage unchanged:",
    unchanged_count / valid_changes * 100
)


# 1. How many non-zero returns do we actually have?

nonzero_returns = (
    df["fx_log_return"]
    .notna() &
    (df["fx_log_return"] != 0)
).sum()

print("Non-zero FX returns:", nonzero_returns)
print(
    "Percentage non-zero:",
    nonzero_returns / df["fx_log_return"].notna().sum() * 100
)



# 2. How many different exchange-rate values are there?

print(
    "Unique buying rates:",
    df["buying_rate"].nunique()
)

print(
    "Total observations:",
    len(df)
)


# 3. How many consecutive days does the rate remain unchanged?

rate_changed = df["buying_rate"].diff().ne(0)

groups = rate_changed.cumsum()

unchanged_runs = (
    df.groupby(groups)
      .size()
)

print("\nLongest periods with the same exchange rate:")
print(
    unchanged_runs
    .sort_values(ascending=False)
    .head(15)
)




# Create groups of consecutive identical exchange rates

temp = df[["date", "buying_rate"]].copy()

temp["rate_changed"] = temp["buying_rate"].ne(
    temp["buying_rate"].shift()
)

temp["group"] = temp["rate_changed"].cumsum()

rate_runs = (
    temp.groupby("group")
    .agg(
        buying_rate=("buying_rate", "first"),
        start_date=("date", "first"),
        end_date=("date", "last"),
        observations=("date", "size")
    )
    .reset_index(drop=True)
)

rate_runs["calendar_days"] = (
    rate_runs["end_date"] - rate_runs["start_date"]
).dt.days + 1

print(
    rate_runs
    .sort_values("observations", ascending=False)
    .head(20)
    .to_string(index=False)
)