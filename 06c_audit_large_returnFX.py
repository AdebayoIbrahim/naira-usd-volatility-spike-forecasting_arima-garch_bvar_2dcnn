#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 11:30:54 2026

@author: setup
"""

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


# ============================================================
# 1. FILE
# ============================================================

INPUT_FILE = "data_cleaned/daily_transformed_dataset_2006_2025.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

df["fx_log_return"] = pd.to_numeric(
    df["fx_log_return"],
    errors="coerce"
)

df["buying_rate"] = pd.to_numeric(
    df["buying_rate"],
    errors="coerce"
)

df = df.sort_values("date").reset_index(drop=True)


# ============================================================
# 3. CREATE EXTRA CHECK COLUMNS
# ============================================================

df["abs_fx_log_return"] = df["fx_log_return"].abs()

df["fx_return_squared"] = (
    df["fx_log_return"] ** 2
)

# Simple percentage change implied by the log return
df["implied_simple_return_pct"] = (
    np.exp(df["fx_log_return"]) - 1
) * 100


# ============================================================
# 4. TRAINING PERIOD ONLY
# ============================================================

train = df[
    df["date"] <= pd.Timestamp("2020-12-31")
].copy()


# ============================================================
# 5. TOP EXTREME RETURNS
# ============================================================

top_returns = train.sort_values(
    "abs_fx_log_return",
    ascending=False
).head(20)


print("=" * 90)
print("EXTREME FX LOG-RETURN AUDIT")
print("=" * 90)

print("\nTop 20 absolute FX log returns:\n")

print(
    top_returns[
        [
            "date",
            "buying_rate",
            "fx_log_return",
            "abs_fx_log_return",
            "fx_return_squared",
            "gap_days",
            "long_gap"
        ]
    ].to_string(index=False)
)


# ============================================================
# 6. CHECK THE LARGEST OBSERVATION
# ============================================================

largest = train.loc[
    train["abs_fx_log_return"].idxmax()
]

print("\n" + "=" * 90)
print("LARGEST OBSERVATION")
print("=" * 90)

print(f"Date:               {largest['date'].date()}")
print(f"Buying rate:        {largest['buying_rate']}")
print(f"Log return:         {largest['fx_log_return']}")
print(f"Absolute log return:{largest['abs_fx_log_return']}")
print(f"Squared return:     {largest['fx_return_squared']}")
print(f"Gap days:           {largest['gap_days']}")
print(f"Long gap:           {largest['long_gap']}")


# ============================================================
# 7. SHOW PREVIOUS AND NEXT OBSERVATIONS
# ============================================================

largest_index = largest.name

start = max(0, largest_index - 3)
end = min(len(train), largest_index + 4)

print("\n" + "=" * 90)
print("OBSERVATIONS AROUND THE EXTREME VALUE")
print("=" * 90)

print(
    train.loc[
        start:end,
        [
            "date",
            "buying_rate",
            "fx_log_return",
            "gap_days",
            "long_gap"
        ]
    ].to_string(index=False)
)