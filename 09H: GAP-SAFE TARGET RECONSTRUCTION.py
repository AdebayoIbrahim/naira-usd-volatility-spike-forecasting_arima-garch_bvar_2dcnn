#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 18:30:33 2026

@author: setup
"""

# ============================================================
# STEP 09H: GAP-SAFE TARGET RECONSTRUCTION
# ============================================================

import os
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "data_cleaned/"
    "forecasting_dataset_2006_2025_spike_gap_corrected.csv"
)

OUTPUT_FILE = (
    "data_cleaned/"
    "forecasting_dataset_2006_2025_spike_gap_target_safe.csv"
)

THRESHOLD_FILE = (
    "results/training_spike_threshold_new.csv"
)

os.makedirs(
    "data_cleaned",
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)


print("=" * 90)
print("STEP 09H: GAP-SAFE TARGET RECONSTRUCTION")
print("=" * 90)

print(
    f"\nRows loaded: {len(df):,}"
)


# ============================================================
# LOAD FROZEN TRAINING THRESHOLD
# ============================================================

# ============================================================
# FROZEN TRAINING SPIKE THRESHOLD
# ============================================================
#
# 90th percentile of NON-ZERO training squared FX returns.
# Frozen before any test-period evaluation.
# ============================================================

threshold = 0.10927724

print(
    f"\nFrozen spike threshold: "
    f"{threshold:.8f}"
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required = [
    "date",
    "segment_id",
    "fx_return_squared"
]

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# RECONSTRUCT TARGET
# ============================================================
#
# target_volatility at time t =
#     squared FX return at t+1
#
# BUT ONLY when t and t+1 belong to the same
# continuous time-series segment.
#
# Therefore, no target is created across a long gap.
# ============================================================

df["next_date"] = df["date"].shift(-1)

df["next_segment_id"] = (
    df["segment_id"].shift(-1)
)

df["next_fx_return_squared"] = (
    df["fx_return_squared"].shift(-1)
)


# Valid next observation
df["valid_next_observation"] = (
    df["segment_id"]
    == df["next_segment_id"]
)


# Target volatility
df["target_volatility"] = np.where(
    df["valid_next_observation"],
    df["next_fx_return_squared"],
    np.nan
)


# Target spike
df["target_spike"] = np.where(
    df["valid_next_observation"],
    (
        df["target_volatility"]
        > threshold
    ).astype(int),
    np.nan
)


# ============================================================
# TARGET DATE
# ============================================================

df["target_date"] = np.where(
    df["valid_next_observation"],
    df["next_date"],
    pd.NaT
)

df["target_date"] = pd.to_datetime(
    df["target_date"]
)


# ============================================================
# AUDIT
# ============================================================

invalid_targets = (
    ~df["valid_next_observation"]
).sum()

valid_targets = (
    df["valid_next_observation"]
).sum()


print("\n" + "=" * 90)
print("TARGET AUDIT")
print("=" * 90)

print(
    f"\nValid next-observation targets: "
    f"{valid_targets:,}"
)

print(
    f"Targets removed at segment boundaries: "
    f"{invalid_targets:,}"
)


# ============================================================
# SHOW REMOVED TARGET BOUNDARIES
# ============================================================

boundary_rows = df[
    ~df["valid_next_observation"]
][
    [
        "date",
        "next_date",
        "segment_id",
        "next_segment_id"
    ]
].copy()


print("\nSegment-boundary targets removed:")

print(
    boundary_rows.to_string(
        index=False
    )
)


# ============================================================
# DROP TEMPORARY COLUMNS
# ============================================================

df = df.drop(
    columns=[
        "next_date",
        "next_segment_id",
        "next_fx_return_squared",
        "valid_next_observation"
    ]
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 90)
print("RESULT SAVED")
print("=" * 90)

print(
    f"\nFile: {OUTPUT_FILE}"
)

print("=" * 90)