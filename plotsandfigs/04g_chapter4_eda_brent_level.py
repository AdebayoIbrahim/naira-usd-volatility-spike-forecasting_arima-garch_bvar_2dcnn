#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 19:53:26 2026

@author: setup
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CHAPTER 4 - EDA
# [FIG 4.5 HERE]
# Daily Brent Crude Oil Price
# ============================================================


# ------------------------------------------------------------
# 1. Project paths
# ------------------------------------------------------------
PROJECT_ROOT = Path.cwd()

FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = (
    PROJECT_ROOT
    / "data_cleaned"
    / "analysis_ready_daily_dataset_2006_2025_corrected.csv"
)


# ------------------------------------------------------------
# 2. Check dataset
# ------------------------------------------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )


# ------------------------------------------------------------
# 3. Load data
# ------------------------------------------------------------
df = pd.read_csv(
    DATA_PATH,
    parse_dates=["date"]
)

df = (
    df.sort_values("date")
      .reset_index(drop=True)
)


# ------------------------------------------------------------
# 4. Identify Brent price column
# ------------------------------------------------------------
possible_brent_columns = [
    "brent_price",
    "brent",
    "Brent",
    "Brent_Price",
    "DCOILBRENTEU",
    "value"
]

brent_column = next(
    (
        col
        for col in possible_brent_columns
        if col in df.columns
    ),
    None
)

if brent_column is None:
    raise ValueError(
        "Brent price column was not found.\n\n"
        f"Available columns are:\n{list(df.columns)}\n\n"
        "Update 'possible_brent_columns' with the "
        "actual Brent column name in your dataset."
    )


# ------------------------------------------------------------
# 5. Keep valid observations
# ------------------------------------------------------------
plot_df = (
    df[["date", brent_column]]
    .dropna()
    .copy()
)


# ------------------------------------------------------------
# 6. Create Figure 4.5
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    plot_df["date"],
    plot_df[brent_column],
    linewidth=0.8
)

ax.set_title(
    "Daily Brent Crude Oil Price, 2006–2025"
)

ax.set_xlabel("Date")
ax.set_ylabel("US Dollars per Barrel")

fig.tight_layout()


# ------------------------------------------------------------
# 7. Save figure
# ------------------------------------------------------------
figure_path = (
    FIGURES_DIR
    / "figure_4_5_brent_crude_oil_price.png"
)

fig.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ------------------------------------------------------------
# 8. Print verification
# ------------------------------------------------------------
print("\n" + "=" * 65)
print("[FIG 4.5 HERE]")
print("=" * 65)

print("Daily Brent Crude Oil Price")

print(f"\nInput dataset:")
print(DATA_PATH)

print(f"Brent column used: {brent_column}")

print(f"Observations plotted: {len(plot_df):,}")

print(
    f"Date range: "
    f"{plot_df['date'].min().date()} "
    f"to "
    f"{plot_df['date'].max().date()}"
)

print(
    f"Minimum price: "
    f"{plot_df[brent_column].min():.4f}"
)

print(
    f"Maximum price: "
    f"{plot_df[brent_column].max():.4f}"
)

print("\nSaved to:")
print(figure_path)

print("\nCompleted successfully.")