#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 19:45:40 2026

@author: setup
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CHAPTER 4 - EDA
# [FIG 4.4 HERE]
# Daily Naira-US Dollar Exchange Rate
# ============================================================


# ------------------------------------------------------------
# 1. Project paths
# ------------------------------------------------------------
PROJECT_ROOT = Path.cwd()

FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Select the best available cleaned dataset
# ------------------------------------------------------------
preferred_path = (
    PROJECT_ROOT
    / "data_cleaned"
    / "analysis_ready_daily_dataset_2006_2025_corrected.csv"
)

fallback_path = (
    PROJECT_ROOT
    / "data_cleaned"
    / "forecasting_dataset_2006_2025.csv"
)

if preferred_path.exists():
    DATA_PATH = preferred_path
elif fallback_path.exists():
    DATA_PATH = fallback_path
else:
    raise FileNotFoundError(
        "Neither of the expected datasets was found:\n"
        f"{preferred_path}\n"
        f"{fallback_path}"
    )


# ------------------------------------------------------------
# 3. Load data
# ------------------------------------------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["date"])

df = (
    df.sort_values("date")
      .reset_index(drop=True)
)


# ------------------------------------------------------------
# 4. Check required column
# ------------------------------------------------------------
if "buying_rate" not in df.columns:
    raise ValueError(
        "The dataset does not contain the required "
        "'buying_rate' column."
    )


# ------------------------------------------------------------
# 5. Keep valid observations
# ------------------------------------------------------------
plot_df = (
    df[["date", "buying_rate"]]
    .dropna()
    .copy()
)


# ------------------------------------------------------------
# 6. Create Figure 4.4
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    plot_df["date"],
    plot_df["buying_rate"],
    linewidth=0.8
)

ax.set_title(
    "Daily Naira-US Dollar Exchange Rate, 2006–2025"
)

ax.set_xlabel("Date")
ax.set_ylabel("Naira per US Dollar")

fig.tight_layout()


# ------------------------------------------------------------
# 7. Save figure
# ------------------------------------------------------------
figure_path = (
    FIGURES_DIR
    / "figure_4_4_fx_exchange_rate_level.png"
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
print("[FIG 4.4 HERE]")
print("=" * 65)

print("Daily Naira-US Dollar Exchange Rate")

print(f"\nInput dataset:")
print(DATA_PATH)

print(f"\nObservations plotted: {len(plot_df):,}")
print(
    f"Date range: "
    f"{plot_df['date'].min().date()} "
    f"to "
    f"{plot_df['date'].max().date()}"
)

print(
    f"Minimum rate: "
    f"{plot_df['buying_rate'].min():.4f}"
)

print(
    f"Maximum rate: "
    f"{plot_df['buying_rate'].max():.4f}"
)

print("\nSaved to:")
print(figure_path)

print("\nCompleted successfully.")