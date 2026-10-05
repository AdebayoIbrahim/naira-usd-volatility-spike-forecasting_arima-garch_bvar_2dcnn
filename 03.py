#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 23:48:16 2026

@author: setup
"""

# ============================================================
# 03_daily_modeling_dataset.py
#
# Purpose:
# Create the analysis-ready daily dataset by:
# 1. Removing observations where Brent is unavailable
# 2. Preserving observation gaps
# 3. Marking long gaps before return calculation
# 4. Creating the main modelling variables
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. Paths
# ============================================================

BASE_DIR = Path(".")

CLEANED_DIR = BASE_DIR / "data_cleaned"
RESULTS_DIR = BASE_DIR / "results"

CLEANED_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. Load merged dataset
# ============================================================

input_file = (
    CLEANED_DIR /
    "merged_daily_macro_dataset_2006_2025.csv"
)

df = pd.read_csv(input_file)

df["date"] = pd.to_datetime(df["date"])

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)


# ============================================================
# 3. Recalculate observation gaps
# ============================================================

df["gap_days"] = (
    df["date"]
    .diff()
    .dt.days
)


# ============================================================
# 4. Remove rows where Brent is unavailable
# ============================================================

missing_brent_count = df["brent_price"].isna().sum()

print("\n" + "=" * 75)
print("BRENT MISSING-VALUE HANDLING")
print("=" * 75)

print("\nMissing Brent observations before removal:")
print(missing_brent_count)


# Keep only dates where all three core variables exist
analysis_df = df.dropna(
    subset=[
        "buying_rate",
        "brent_price",
        "gross_reserves"
    ]
).copy()

analysis_df = (
    analysis_df
    .sort_values("date")
    .reset_index(drop=True)
)


print("\nRows before removing missing Brent:")
print(len(df))

print("\nRows after removing missing Brent:")
print(len(analysis_df))

print("\nRows removed:")
print(len(df) - len(analysis_df))


# ============================================================
# 5. Recalculate gaps AFTER removing Brent holidays
# ============================================================

analysis_df["gap_days"] = (
    analysis_df["date"]
    .diff()
    .dt.days
)


# ============================================================
# 6. Identify long observation gaps
# ============================================================

# We treat gaps greater than 5 calendar days as structural breaks.
analysis_df["long_gap"] = (
    analysis_df["gap_days"] > 5
).astype(int)


print("\n" + "=" * 75)
print("GAP DIAGNOSTICS AFTER BRENT CLEANING")
print("=" * 75)

print("\nGap distribution:")
print(
    analysis_df["gap_days"]
    .value_counts()
    .sort_index()
)

print("\nNumber of gaps > 5 days:")
print(
    (analysis_df["gap_days"] > 5).sum()
)

print("\nLong-gap dates:")
print(
    analysis_df.loc[
        analysis_df["long_gap"] == 1,
        ["date", "gap_days"]
    ].to_string(index=False)
)


# ============================================================
# 7. Select core modelling variables
# ============================================================

# Primary variables for the revised daily study:
#
# buying_rate      -> Nigerian Naira/USD exchange rate
# brent_price      -> international Brent crude oil price
# gross_reserves   -> Nigeria's gross external reserves

analysis_df = analysis_df[
    [
        "date",
        "buying_rate",
        "brent_price",
        "gross_reserves",
        "liquid_reserves",
        "blocked_reserves",
        "blocked_percent",
        "gap_days",
        "long_gap"
    ]
].copy()


# ============================================================
# 8. Check the resulting dataset
# ============================================================

print("\n" + "=" * 75)
print("ANALYSIS-READY DATASET")
print("=" * 75)

print("\nShape:")
print(analysis_df.shape)

print("\nDate range:")
print(
    analysis_df["date"].min(),
    "to",
    analysis_df["date"].max()
)

print("\nMissing values:")
print(
    analysis_df.isna().sum()
)

print("\nDuplicate dates:")
print(
    analysis_df["date"].duplicated().sum()
)


# ============================================================
# 9. Save analysis-ready dataset
# ============================================================

output_file = (
    CLEANED_DIR /
    "analysis_ready_daily_dataset_2006_2025.csv"
)

analysis_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# 10. Save dataset summary
# ============================================================

summary = pd.DataFrame({
    "item": [
        "Original merged observations",
        "Missing Brent observations removed",
        "Final analysis-ready observations",
        "Number of long gaps (>5 days)",
        "Study start",
        "Study end"
    ],
    "value": [
        len(df),
        missing_brent_count,
        len(analysis_df),
        int((analysis_df["long_gap"] == 1).sum()),
        analysis_df["date"].min(),
        analysis_df["date"].max()
    ]
})

summary_file = (
    RESULTS_DIR /
    "analysis_ready_daily_dataset_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# 11. Preview
# ============================================================

print("\nFirst 10 observations:")
print(
    analysis_df
    .head(10)
    .to_string(index=False)
)

print("\nLast 10 observations:")
print(
    analysis_df
    .tail(10)
    .to_string(index=False)
)

print("\nSaved analysis-ready dataset:")
print(output_file)

print("\nSaved dataset summary:")
print(summary_file)

print("\nAnalysis-ready daily dataset created successfully.")

showf = pd.read_csv( CLEANED_DIR / "analysis_ready_daily_dataset_2006_2025.csv" );
print(len(showf))

#checking nan brent aggain
upd_missing_brent_count = showf["brent_price"].isna().sum()
print(upd_missing_brent_count)