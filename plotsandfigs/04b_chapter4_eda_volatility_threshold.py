#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CHAPTER FOUR - EXPLORATORY DATA ANALYSIS

Readable version of the daily FX volatility and spike-threshold plot.

WHAT THIS FIGURE SHOWS
----------------------
1. Upper panel: the full range of daily squared log returns, including
   very large observations.
2. Lower panel: a zoomed view from 0 to 0.50, so the spike threshold
   at 0.10927724 can be seen clearly.

Important: observations above 0.50 are clipped ONLY in the lower panel
for display. No observations are removed from the upper panel or from
any of the summary statistics.

Input:
    data_cleaned/forecasting_dataset_2006_2025_spike_gap_target_safe.csv

Output:
    figures/figure_4_2_fx_volatility_spike_threshold.png

In the thesis, number this as Figure 4.5 if it is the fifth figure
appearing in Chapter Four. The output filename retains the old project
filename so any later scripts that expect it will keep working.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter


# ============================================================
# 1. FILE LOCATIONS
# ============================================================

# Run this script from the root of your project in Spyder.
PROJECT_ROOT = Path.cwd()

DATA_PATH = (
    PROJECT_ROOT
    / "data_cleaned"
    / "forecasting_dataset_2006_2025_spike_gap_target_safe.csv"
)

FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_PATH = (
    FIGURES_DIR
    / "figure_4_2_fx_volatility_spike_threshold.png"
)


# ============================================================
# 2. SETTINGS
# ============================================================

# This value is fixed from the study's training-period definition.
# A day is labelled a spike when squared log return is GREATER than it.
SPIKE_THRESHOLD = 0.10927724

# The lower panel zooms in on this range so the threshold is visible.
# Values higher than this are capped in the lower panel only.
ZOOM_MAX = 0.50

# The common study period ends in October 2025.
STUDY_END = pd.Timestamp("2025-10-31")


# ============================================================
# 3. LOAD AND CHECK THE DATA
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}\n\n"
        "In Spyder, set the working directory to your project root, "
        "which should contain the data_cleaned folder."
    )

# Read first, then convert columns explicitly so errors are easy to diagnose.
df = pd.read_csv(DATA_PATH)

required_columns = ["date", "fx_return_squared"]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}\n"
        f"Columns found: {list(df.columns)}"
    )

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["fx_return_squared"] = pd.to_numeric(
    df["fx_return_squared"], errors="coerce"
)

# Keep valid records up to the end of the study period.
df = df.dropna(subset=["date", "fx_return_squared"])
df = df.loc[df["date"] <= STUDY_END].copy()
df = df.loc[np.isfinite(df["fx_return_squared"])].copy()
df = df.sort_values("date").reset_index(drop=True)

if df.empty:
    raise ValueError(
        "No valid observations remain after cleaning and applying the study end date."
    )

if (df["fx_return_squared"] < 0).any():
    raise ValueError(
        "Negative values were found in fx_return_squared. "
        "Squared returns should not be negative."
    )


# ============================================================
# 4. LABEL SPIKES
# ============================================================

# True means the squared daily log return exceeds the fixed threshold.
df["is_spike"] = df["fx_return_squared"] > SPIKE_THRESHOLD

# Full observed series, used in both panels.
dates = df["date"]
volatility = df["fx_return_squared"]

# For the zoomed panel only, cap very high values at the top of the plot.
# The full values remain unchanged in the original `volatility` series.
volatility_zoomed = volatility.clip(upper=ZOOM_MAX)

spike_mask = df["is_spike"]
spike_dates = dates[spike_mask]
spike_values_zoomed = volatility_zoomed[spike_mask]


# ============================================================
# 5. CREATE TWO EASY-TO-READ PANELS
# ============================================================

fig, (ax_full, ax_zoom) = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(13, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [1, 1.25]},
    constrained_layout=True,
)

fig.suptitle(
    "Daily Naira/USD Volatility and Spike Threshold (2006–2025)",
    fontsize=15,
    fontweight="bold",
)


# ------------------------------------------------------------
# Upper panel: FULL DATA RANGE
# ------------------------------------------------------------

ax_full.plot(
    dates,
    volatility,
    color="steelblue",
    linewidth=0.75,
    label=r"Squared return ($r_t^2$)",
)

ax_full.set_title(
    "A. Full series",
    loc="left",
    fontsize=11,
)
ax_full.set_ylabel(r"Squared log return ($r_t^2$)")
ax_full.set_ylim(bottom=0)
ax_full.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
ax_full.legend(loc="upper right", frameon=True)



# ------------------------------------------------------------
# Lower panel: ZOOM NEAR THE THRESHOLD
# ------------------------------------------------------------

# Shade the part of this zoomed panel that is above the spike threshold.
ax_zoom.axhspan(
    SPIKE_THRESHOLD,
    ZOOM_MAX,
    color="indianred",
    alpha=0.08,
    zorder=0,
)

ax_zoom.plot(
    dates,
    volatility_zoomed,
    color="steelblue",
    linewidth=0.75,
    label=rf"Volatility (values capped at {ZOOM_MAX:.2f})",
    zorder=2,
)

# Highlight observations that meet the study's spike definition.
ax_zoom.scatter(
    spike_dates,
    spike_values_zoomed,
    s=8,
    color="firebrick",
    alpha=0.65,
    label="Spike days",
    zorder=3,
)

# The key threshold line is easy to see in this zoomed panel.
ax_zoom.axhline(
    SPIKE_THRESHOLD,
    color="firebrick",
    linestyle="--",
    linewidth=1.5,
    label=f"Threshold = {SPIKE_THRESHOLD:.4f}",
    zorder=4,
)

ax_zoom.set_title(
    f"B. Threshold detail (values above {ZOOM_MAX:.2f} capped)",
    loc="left",
    fontsize=11,
)
ax_zoom.set_ylabel(r"Squared log return ($r_t^2$)")
ax_zoom.set_xlabel("Date")
ax_zoom.set_ylim(0, ZOOM_MAX)
ax_zoom.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
ax_zoom.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
ax_zoom.legend(loc="upper right", frameon=True, fontsize=9, ncol=1)



# ------------------------------------------------------------
# Date formatting
# ------------------------------------------------------------

ax_zoom.xaxis.set_major_locator(mdates.YearLocator(2))
ax_zoom.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

for label in ax_zoom.get_xticklabels():
    label.set_rotation(0)
    label.set_ha("center")


# ============================================================
# 6. SAVE FIGURE
# ============================================================

fig.savefig(
    FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 7. PRINT A SHORT VERIFICATION SUMMARY
# ============================================================

n_observations = int(volatility.notna().sum())
n_spikes = int(spike_mask.sum())
spike_rate = 100 * n_spikes / n_observations
n_capped_in_zoom = int((volatility > ZOOM_MAX).sum())

print("\n" + "=" * 72)
print("FIGURE 4.5: VOLATILITY AND SPIKE THRESHOLD")
print("=" * 72)
print(f"Observations:                {n_observations:,}")
print(f"Period shown:                {dates.min().date()} to {dates.max().date()}")
print(f"Spike threshold:             {SPIKE_THRESHOLD:.8f}")
print(f"Observed spike days:         {n_spikes:,}")
print(f"Observed spike rate:         {spike_rate:.2f}%")
print(f"Values above {ZOOM_MAX:.2f}, capped only in lower panel: {n_capped_in_zoom:,}")
print("\nSaved figure:")
print(FIGURE_PATH)
print("\nCompleted successfully.")
