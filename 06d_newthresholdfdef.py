#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 12:17:36 2026

@author: setup
"""

# -*- coding: utf-8 -*-
"""
Step 06D: Compare candidate spike definitions.

Purpose:
    Compare percentile-based spike thresholds using:
        1. All training observations
        2. Non-zero training returns only

Volatility proxy:
    Squared daily log return (r_t^2)

Important:
    Thresholds are calculated from TRAINING DATA ONLY.
"""

import os
import pandas as pd
import numpy as np


# ============================================================
# 1. FILE PATH
# ============================================================

INPUT_FILE = "data_cleaned/daily_transformed_dataset_2006_2025.csv"
OUTPUT_FILE = "results/spike_definition_comparison.csv"

TRAIN_END = "2020-12-31"

PERCENTILES = [0.90, 0.95, 0.97]


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df["fx_log_return"] = pd.to_numeric(
    df["fx_log_return"],
    errors="coerce"
)

df = df.dropna(
    subset=["date", "fx_log_return"]
).copy()

df = df.sort_values("date").reset_index(drop=True)


# ============================================================
# 3. CREATE SQUARED RETURN
# ============================================================

df["fx_return_squared"] = (
    df["fx_log_return"] ** 2
)


# ============================================================
# 4. TRAINING DATA ONLY
# ============================================================

train = df[
    df["date"] <= pd.Timestamp(TRAIN_END)
].copy()

print("=" * 80)
print("STEP 06D: SPIKE DEFINITION COMPARISON")
print("=" * 80)

print(f"\nTraining observations: {len(train):,}")
print(
    f"Training period: "
    f"{train['date'].min().date()} "
    f"to "
    f"{train['date'].max().date()}"
)


# ============================================================
# 5. CREATE NON-ZERO SUBSET
# ============================================================

nonzero_train = train[
    train["fx_log_return"] != 0
].copy()

print(
    f"Non-zero return observations: "
    f"{len(nonzero_train):,}"
)

print(
    f"Zero return observations: "
    f"{(train['fx_log_return'] == 0).sum():,}"
)


# ============================================================
# 6. CALCULATE BOTH TYPES OF THRESHOLDS
# ============================================================

results = []


for population_name, population in [
    ("All observations", train),
    ("Non-zero returns only", nonzero_train)
]:

    print("\n" + "-" * 80)
    print(population_name)
    print("-" * 80)

    for percentile in PERCENTILES:

        # ----------------------------------------------------
        # Threshold
        # ----------------------------------------------------

        threshold = population[
            "fx_return_squared"
        ].quantile(percentile)


        # ----------------------------------------------------
        # Equivalent absolute log return
        # ----------------------------------------------------

        abs_log_return = np.sqrt(
            threshold
        )


        # ----------------------------------------------------
        # Convert to percentage movement
        #
        # The stored fx_log_return is multiplied by 100,
        # so divide by 100 before using exp().
        # ----------------------------------------------------

        raw_log_return = (
            abs_log_return / 100
        )

        percentage_movement = (
            np.exp(raw_log_return) - 1
        ) * 100


        # ----------------------------------------------------
        # Apply threshold to ALL training observations
        #
        # This lets us see how many actual training
        # observations would be classified as spikes.
        # ----------------------------------------------------

        spike_mask = (
            train["fx_return_squared"] > threshold
        )

        spike_count = spike_mask.sum()

        spike_percentage = (
            spike_count / len(train)
        ) * 100


        # ----------------------------------------------------
        # Number of non-zero observations classified
        # as spikes
        # ----------------------------------------------------

        nonzero_spike_count = (
            nonzero_train["fx_return_squared"] > threshold
        ).sum()

        if len(nonzero_train) > 0:
            nonzero_spike_percentage = (
                nonzero_spike_count
                / len(nonzero_train)
            ) * 100
        else:
            nonzero_spike_percentage = np.nan


        results.append({
            "threshold_population": population_name,
            "percentile": percentile * 100,
            "threshold_r_squared": threshold,
            "absolute_stored_log_return": abs_log_return,
            "equivalent_absolute_log_return": raw_log_return,
            "approx_price_movement_percent": percentage_movement,
            "training_spike_count": spike_count,
            "training_spike_percentage": spike_percentage,
            "nonzero_spike_count": nonzero_spike_count,
            "nonzero_spike_percentage": nonzero_spike_percentage
        })


# ============================================================
# 7. RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 80)
print("COMPARISON RESULTS")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        formatters={
            "percentile":
                lambda x: f"{x:.0f}%",

            "threshold_r_squared":
                lambda x: f"{x:.8f}",

            "absolute_stored_log_return":
                lambda x: f"{x:.6f}",

            "equivalent_absolute_log_return":
                lambda x: f"{x:.8f}",

            "approx_price_movement_percent":
                lambda x: f"{x:.4f}%",

            "training_spike_count":
                lambda x: f"{int(x):,}",

            "training_spike_percentage":
                lambda x: f"{x:.2f}%",

            "nonzero_spike_count":
                lambda x: f"{int(x):,}",

            "nonzero_spike_percentage":
                lambda x: f"{x:.2f}%"
        }
    )
)


# ============================================================
# 9. SAVE
# ============================================================

os.makedirs("results", exist_ok=True)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 80)
print("RESULT SAVED")
print("=" * 80)

print(f"File: {OUTPUT_FILE}")

print("=" * 80)