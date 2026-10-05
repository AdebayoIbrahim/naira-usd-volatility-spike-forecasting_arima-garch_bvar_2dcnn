#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 19:41:29 2026

@author: setup
"""

# ============================================================
# 03b_verified_exchange_anomaly_corrections.py
# Verified corrections to exchange-rate observations
# ============================================================

import os
import pandas as pd

# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

INPUT_FILE = (
    "data_cleaned/analysis_ready_daily_dataset_2006_2025.csv"
)

OUTPUT_FILE = (
    "data_cleaned/analysis_ready_daily_dataset_2006_2025_corrected.csv"
)

AUDIT_FILE = (
    "results/verified_exchange_rate_corrections.csv"
)

os.makedirs("data_cleaned", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ------------------------------------------------------------
# 2. Load analysis-ready dataset
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)


# ------------------------------------------------------------
# 3. Define verified corrections
# ------------------------------------------------------------

corrections = [
    {
        "date": "2008-08-01",
        "variable": "buying_rate",
        "original_value": 15.59,
        "corrected_value": 115.59,
        "reason": (
            "Value inconsistent with immediately surrounding CBN "
            "observations. The CBN row shows buying=15.59, "
            "central=66.09 and selling=116.59, while adjacent "
            "observations are approximately 115.59, 116.09 and "
            "116.59. Corrected based on surrounding official "
            "series and verification."
        )
    },
    {
        "date": "2007-08-28",
        "variable": "buying_rate",
        "original_value": 24.02,
        "corrected_value": 124.02,
        "reason": (
            "Value inconsistent with immediately surrounding CBN "
            "observations. Adjacent observations are approximately "
            "124.05 and 123.95, with corresponding central and "
            "selling rates around 124.5 and 125.0."
        )
    }
]


# ------------------------------------------------------------
# 4. Apply corrections with validation
# ------------------------------------------------------------

audit_rows = []

for correction in corrections:

    date = pd.to_datetime(correction["date"])

    mask = df["date"] == date

    if mask.sum() != 1:
        raise ValueError(
            f"Expected exactly one row for {date.date()}, "
            f"found {mask.sum()}."
        )

    current_value = df.loc[mask, correction["variable"]].iloc[0]

    # Make sure the dataset contains the value we expect
    # before modifying it.
    if current_value != correction["original_value"]:
        raise ValueError(
            f"Unexpected value for {date.date()}.\n"
            f"Expected: {correction['original_value']}\n"
            f"Found: {current_value}\n"
            f"Correction not applied."
        )

    # Apply correction
    df.loc[
        mask,
        correction["variable"]
    ] = correction["corrected_value"]

    audit_rows.append({
        "date": date,
        "variable": correction["variable"],
        "original_value": current_value,
        "corrected_value": correction["corrected_value"],
        "reason": correction["reason"]
    })


# ------------------------------------------------------------
# 5. Save audit trail
# ------------------------------------------------------------

audit = pd.DataFrame(audit_rows)

audit.to_csv(
    AUDIT_FILE,
    index=False
)


# ------------------------------------------------------------
# 6. Save corrected dataset
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# 7. Verify corrected values
# ------------------------------------------------------------

print("=" * 70)
print("VERIFIED EXCHANGE-RATE ANOMALY CORRECTIONS")
print("=" * 70)

print("\nCorrections applied:")

for correction in corrections:

    date = pd.to_datetime(correction["date"])

    value = df.loc[
        df["date"] == date,
        correction["variable"]
    ].iloc[0]

    print(
        f"{date.date()} | "
        f"{correction['variable']} | "
        f"{correction['original_value']} -> {value}"
    )


print("\nCorrected dataset:")
print(OUTPUT_FILE)

print("\nAudit file:")
print(AUDIT_FILE)

print("\nCorrection process completed successfully.")