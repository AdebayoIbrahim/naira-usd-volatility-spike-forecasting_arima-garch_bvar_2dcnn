#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 12:22:49 2026

@author: setup
"""

# -*- coding: utf-8 -*-
"""
Step 06E: Audit observations classified as volatility spikes.

Spike definition:
    90th percentile of squared FX returns
    among NON-ZERO training returns only.

Training period:
    2006-04-04 to 2020-12-31

Important:
    fx_log_return in our dataset is stored as:
        100 * ln(X_t / X_{t-1})

Therefore:
    actual_log_return = fx_log_return / 100
"""

import os
import pandas as pd
import numpy as np


# ============================================================
# 1. FILE PATH
# ============================================================

INPUT_FILE = "data_cleaned/daily_transformed_dataset_2006_2025.csv"

OUTPUT_FILE = "results/proposed_spike_observations_audit.csv"
YEARLY_OUTPUT_FILE = "results/proposed_spike_counts_by_year.csv"

TRAIN_END = "2020-12-31"


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 90)
print("STEP 06E: PROPOSED SPIKE OBSERVATION AUDIT")
print("=" * 90)

print(f"\nRows loaded: {len(df):,}")


# ============================================================
# 3. VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "date",
    "buying_rate",
    "fx_log_return",
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


# ============================================================
# 4. CLEAN DATA
# ============================================================

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df["buying_rate"] = pd.to_numeric(
    df["buying_rate"],
    errors="coerce"
)

df["fx_log_return"] = pd.to_numeric(
    df["fx_log_return"],
    errors="coerce"
)

df["gap_days"] = pd.to_numeric(
    df["gap_days"],
    errors="coerce"
)

df["long_gap"] = pd.to_numeric(
    df["long_gap"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "date",
        "buying_rate",
        "fx_log_return"
    ]
).copy()

df = df.sort_values("date").reset_index(drop=True)


# ============================================================
# 5. CREATE RETURN VARIABLES
# ============================================================

# Squared stored return
df["fx_return_squared"] = (
    df["fx_log_return"] ** 2
)

# Absolute stored return
df["abs_fx_log_return"] = (
    df["fx_log_return"].abs()
)

# Convert stored return back to actual log-return scale
df["actual_log_return"] = (
    df["fx_log_return"] / 100
)

# Convert to simple percentage movement
df["simple_return_percent"] = (
    (np.exp(df["actual_log_return"]) - 1) * 100
)

# Absolute simple percentage movement
df["abs_simple_return_percent"] = (
    df["simple_return_percent"].abs()
)


# ============================================================
# 6. CREATE PREVIOUS/NEXT RATE INFORMATION
# ============================================================

df["previous_buying_rate"] = (
    df["buying_rate"].shift(1)
)

df["next_buying_rate"] = (
    df["buying_rate"].shift(-1)
)

df["previous_date"] = (
    df["date"].shift(1)
)

df["next_date"] = (
    df["date"].shift(-1)
)


# ============================================================
# 7. TRAINING DATA ONLY
# ============================================================

train = df[
    df["date"] <= pd.Timestamp(TRAIN_END)
].copy()

if train.empty:
    raise ValueError(
        "Training dataset is empty."
    )


# ============================================================
# 8. NON-ZERO TRAINING RETURNS
# ============================================================

nonzero_train = train[
    train["fx_log_return"] != 0
].copy()

if nonzero_train.empty:
    raise ValueError(
        "No non-zero returns found in training data."
    )


# ============================================================
# 9. CALCULATE PROPOSED THRESHOLD
# ============================================================

threshold = nonzero_train[
    "fx_return_squared"
].quantile(0.90)

abs_stored_threshold = np.sqrt(
    threshold
)

actual_log_threshold = (
    abs_stored_threshold / 100
)

approx_percent_threshold = (
    np.exp(actual_log_threshold) - 1
) * 100


print("\nProposed spike definition:")
print(
    "90th percentile of squared returns "
    "among non-zero training returns"
)

print(
    f"Threshold (stored return²): "
    f"{threshold:.8f}"
)

print(
    f"Absolute stored log-return threshold: "
    f"{abs_stored_threshold:.6f}"
)

print(
    f"Equivalent actual log-return threshold: "
    f"{actual_log_threshold:.8f}"
)

print(
    f"Approximate absolute price movement: "
    f"{approx_percent_threshold:.4f}%"
)


# ============================================================
# 10. IDENTIFY SPIKES
# ============================================================

spikes = train[
    train["fx_return_squared"] > threshold
].copy()

spikes = spikes.sort_values(
    "abs_fx_log_return",
    ascending=False
).reset_index(drop=True)


# ============================================================
# 11. ADD SPIKE RANK
# ============================================================

spikes.insert(
    0,
    "spike_rank",
    range(1, len(spikes) + 1)
)


# ============================================================
# 12. SELECT AUDIT COLUMNS
# ============================================================

audit_columns = [
    "spike_rank",
    "date",
    "previous_date",
    "next_date",
    "previous_buying_rate",
    "buying_rate",
    "next_buying_rate",
    "fx_log_return",
    "abs_fx_log_return",
    "fx_return_squared",
    "actual_log_return",
    "simple_return_percent",
    "abs_simple_return_percent",
    "gap_days",
    "long_gap"
]

spike_audit = spikes[
    audit_columns
].copy()


# ============================================================
# 13. DISPLAY TOP SPIKES
# ============================================================

print("\n" + "=" * 90)
print("PROPOSED SPIKE OBSERVATIONS")
print("=" * 90)

print(
    f"\nTotal proposed spikes: "
    f"{len(spike_audit):,}"
)

print(
    f"Percentage of training observations: "
    f"{(len(spike_audit) / len(train)) * 100:.2f}%"
)

print(
    f"Percentage of non-zero training returns: "
    f"{(len(spike_audit) / len(nonzero_train)) * 100:.2f}%"
)


print("\nTop 30 largest proposed spikes:\n")

print(
    spike_audit.head(30).to_string(
        index=False,
        formatters={
            "fx_log_return":
                lambda x: f"{x:.6f}",

            "abs_fx_log_return":
                lambda x: f"{x:.6f}",

            "fx_return_squared":
                lambda x: f"{x:.6f}",

            "actual_log_return":
                lambda x: f"{x:.8f}",

            "simple_return_percent":
                lambda x: f"{x:.4f}%",

            "abs_simple_return_percent":
                lambda x: f"{x:.4f}%"
        }
    )
)


# ============================================================
# 14. YEARLY SPIKE COUNTS
# ============================================================

spikes["year"] = (
    spikes["date"].dt.year
)

yearly_counts = (
    spikes.groupby("year")
    .size()
    .reset_index(name="spike_count")
)

yearly_counts["percentage_of_all_training"] = (
    yearly_counts["spike_count"]
    / len(train)
    * 100
)

yearly_counts = yearly_counts.sort_values(
    "year"
)


print("\n" + "=" * 90)
print("SPIKE COUNTS BY YEAR")
print("=" * 90)

print(
    yearly_counts.to_string(
        index=False,
        formatters={
            "percentage_of_all_training":
                lambda x: f"{x:.2f}%"
        }
    )
)


# ============================================================
# 15. GAP CHECK
# ============================================================

long_gap_spikes = spike_audit[
    spike_audit["long_gap"] == 1
]

print("\n" + "=" * 90)
print("LONG-GAP CHECK")
print("=" * 90)

print(
    f"Spikes immediately associated with long-gap flag: "
    f"{len(long_gap_spikes):,}"
)

if len(long_gap_spikes) > 0:
    print("\nThese observations need special attention:\n")

    print(
        long_gap_spikes[
            [
                "date",
                "previous_date",
                "next_date",
                "previous_buying_rate",
                "buying_rate",
                "next_buying_rate",
                "fx_log_return",
                "gap_days",
                "long_gap"
            ]
        ].to_string(index=False)
    )


# ============================================================
# 16. SAVE OUTPUTS
# ============================================================

os.makedirs("results", exist_ok=True)

spike_audit.to_csv(
    OUTPUT_FILE,
    index=False
)

yearly_counts.to_csv(
    YEARLY_OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 90)
print("RESULTS SAVED")
print("=" * 90)

print(f"Detailed audit: {OUTPUT_FILE}")
print(f"Yearly counts:  {YEARLY_OUTPUT_FILE}")

print("=" * 90)