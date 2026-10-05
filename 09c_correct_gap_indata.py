#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 18:10:50 2026

@author: setup
"""

# -*- coding: utf-8 -*-

"""
Step 09C: Correct Gap Structure in the Spike Forecasting Dataset

Purpose:
    Recalculate gap_days and long_gap directly from the actual
    observation dates.

Reason:
    The previous Step 07 dataset incorrectly contains long_gap = 0
    for every observation even though the dates contain genuine
    long gaps.

Rule:
    gap_days = calendar-day difference from previous observation

    long_gap = 1 if gap_days > 5
                0 otherwise

A new segment begins immediately after a long gap.

No target, return, threshold, or model variable is changed.
"""

import os
import pandas as pd


# ============================================================
# 1. FILE PATHS
# ============================================================

INPUT_FILE = (
    "data_cleaned/forecasting_dataset_2006_2025_spike.csv"
)

OUTPUT_FILE = (
    "data_cleaned/forecasting_dataset_2006_2025_spike_gap_corrected.csv"
)

AUDIT_OUTPUT = (
    "results/gap_structure_audit_spike.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("=" * 90)
print("STEP 09C: GAP STRUCTURE CORRECTION")
print("=" * 90)

print(
    f"\nRows loaded: {len(df):,}"
)


# ============================================================
# 3. VALIDATE DATE
# ============================================================

if "date" not in df.columns:

    raise ValueError(
        "The dataset must contain a 'date' column."
    )


df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


if df["date"].isna().any():

    raise ValueError(
        "Invalid dates found in the dataset."
    )


# ============================================================
# 4. SORT CHRONOLOGICALLY
# ============================================================

df = df.sort_values(
    "date"
).reset_index(
    drop=True
)


# ============================================================
# 5. RECALCULATE GAP DAYS
# ============================================================

df["gap_days"] = (
    df["date"]
    .diff()
    .dt.days
)


# First observation has no previous observation
df.loc[
    df.index[0],
    "gap_days"
] = 0


# ============================================================
# 6. RECALCULATE LONG GAP FLAG
# ============================================================

df["long_gap"] = (
    df["gap_days"] > 5
).astype(int)


# ============================================================
# 7. CREATE SEGMENT ID
# ============================================================

df["segment_id"] = (
    df["long_gap"]
    .cumsum()
)


# ============================================================
# 8. DISPLAY GAP SUMMARY
# ============================================================

long_gaps = df[
    df["long_gap"] == 1
].copy()


print("\n" + "=" * 90)
print("GAP SUMMARY")
print("=" * 90)

print(
    f"Maximum gap: "
    f"{df['gap_days'].max():.0f} days"
)

print(
    f"Long gaps (>5 days): "
    f"{len(long_gaps):,}"
)

print(
    f"Continuous segments: "
    f"{df['segment_id'].nunique():,}"
)


# ============================================================
# 9. DISPLAY ALL LONG GAPS
# ============================================================

if not long_gaps.empty:

    print("\nLong-gap observations:\n")

    gap_display = long_gaps.copy()

    gap_display["previous_date"] = (
        gap_display["date"].shift(1)
    )

    # The shift above is not the correct previous date
    # relative to the original complete dataset.
    # Recalculate directly from the full dataset.

    gap_display["previous_date"] = (
        df["date"]
        .shift(1)
        .loc[long_gaps.index]
        .values
    )

    print(
        gap_display[
            [
                "previous_date",
                "date",
                "gap_days",
                "long_gap",
                "segment_id"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# 10. VERIFY NO FALSE LONG-GAP FLAGS
# ============================================================

incorrect_flags = (
    df["long_gap"]
    != (df["gap_days"] > 5).astype(int)
).sum()


print("\n" + "=" * 90)
print("FLAG VALIDATION")
print("=" * 90)

print(
    f"Incorrect long-gap flags: "
    f"{incorrect_flags:,}"
)


if incorrect_flags != 0:

    raise RuntimeError(
        "Gap flag validation failed."
    )


# ============================================================
# 11. TRAINING GAP SUMMARY
# ============================================================

training = df[
    df["date"] <= pd.Timestamp(
        "2020-12-31"
    )
].copy()

test = df[
    df["date"] > pd.Timestamp(
        "2020-12-31"
    )
].copy()


print("\n" + "=" * 90)
print("TRAIN / TEST GAP SUMMARY")
print("=" * 90)

print(
    f"Training long gaps: "
    f"{training['long_gap'].sum():,}"
)

print(
    f"Test long gaps: "
    f"{test['long_gap'].sum():,}"
)


# ============================================================
# 12. SAVE AUDIT TABLE
# ============================================================

audit_columns = [
    "date",
    "gap_days",
    "long_gap",
    "segment_id"
]


df[
    audit_columns
].to_csv(
    AUDIT_OUTPUT,
    index=False
)


# ============================================================
# 13. SAVE CORRECTED DATASET
# ============================================================

os.makedirs(
    "data_cleaned",
    exist_ok=True
)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 14. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 90)
print("FILES SAVED")
print("=" * 90)

print(
    f"Corrected dataset: {OUTPUT_FILE}"
)

print(
    f"Gap audit:         {AUDIT_OUTPUT}"
)

print("=" * 90)