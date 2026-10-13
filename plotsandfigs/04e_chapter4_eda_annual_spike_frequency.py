#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 11:07:42 2026

@author: setup
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CHAPTER 4 - EDA
# [FIG 4.3 HERE]
# Annual Volatility-Spike Frequency
# ============================================================


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------
PROJECT_ROOT = Path.cwd()

DATA_PATH = (
    PROJECT_ROOT
    / "data_cleaned"
    / "forecasting_dataset_2006_2025_spike_gap_target_safe.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Load final forecasting dataset
# ------------------------------------------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["target_date"]
)

df = (
    df.sort_values("target_date")
      .reset_index(drop=True)
)


# ------------------------------------------------------------
# 3. Keep valid target observations
# ------------------------------------------------------------
df = df.dropna(
    subset=["target_date", "target_spike"]
).copy()

df["target_spike"] = df["target_spike"].astype(int)

df["year"] = df["target_date"].dt.year


# ------------------------------------------------------------
# 4. Calculate annual spike frequency
# ------------------------------------------------------------
annual = (
    df.groupby("year")
      .agg(
          observations=("target_spike", "size"),
          spikes=("target_spike", "sum")
      )
      .reset_index()
)

annual["non_spikes"] = (
    annual["observations"] - annual["spikes"]
)

annual["spike_rate_percent"] = (
    annual["spikes"]
    / annual["observations"]
    * 100
)

annual["non_spike_rate_percent"] = (
    annual["non_spikes"]
    / annual["observations"]
    * 100
)


# ------------------------------------------------------------
# 5. Save annual summary
# ------------------------------------------------------------
csv_path = (
    RESULTS_DIR
    / "figure_4_3_annual_spike_frequency.csv"
)

annual.to_csv(
    csv_path,
    index=False
)


# ------------------------------------------------------------
# 6. Create Figure 4.3
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))

ax.bar(
    annual["year"].astype(str),
    annual["spike_rate_percent"]
)

# Mark beginning of out-of-sample period
test_start_position = list(annual["year"]).index(2021)

ax.axvline(
    test_start_position - 0.5,
    linestyle="--",
    linewidth=1.0,
    label="Start of out-of-sample period (2021)"
)

ax.set_title(
    "Annual Frequency of Naira-US Dollar Volatility Spikes, 2006–2025"
)

ax.set_xlabel("Year")
ax.set_ylabel("Volatility spikes (%)")

ax.tick_params(axis="x", rotation=60)

ax.legend()

fig.tight_layout()


# ------------------------------------------------------------
# 7. Save figure
# ------------------------------------------------------------
figure_path = (
    FIGURES_DIR
    / "figure_4_3_annual_spike_frequency.png"
)

fig.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ------------------------------------------------------------
# 8. Print results
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("[FIG 4.3 HERE]")
print("=" * 70)

print(
    "Annual Frequency of Naira-US Dollar "
    "Volatility Spikes, 2006–2025"
)

print("\nAnnual summary:")
print(
    annual[
        [
            "year",
            "observations",
            "spikes",
            "spike_rate_percent"
        ]
    ].to_string(index=False)
)

print("\nCSV saved to:")
print(csv_path)

print("\nFigure saved to:")
print(figure_path)

print("\nCompleted successfully.")

