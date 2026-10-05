import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD TRANSFORMED DATA
# ============================================================

input_file = "data_cleaned/daily_transformed_dataset_2006_2025.csv"

df = pd.read_csv(input_file, parse_dates=["date"])

df = df.sort_values("date").reset_index(drop=True)


# ============================================================
# 2. CREATE ABSOLUTE FX RETURN
# ============================================================

df["fx_abs_return"] = df["fx_log_return"].abs()


# ============================================================
# 3. CREATE VOLATILITY SEGMENTS
# ============================================================
# A long gap marks the beginning of a new independent segment.
#
# This prevents rolling volatility windows from crossing
# major gaps in the available data.

df["volatility_segment"] = (
    df["long_gap"]
    .fillna(0)
    .astype(int)
    .cumsum()
)


# ============================================================
# 4. GAP-AWARE ROLLING VOLATILITY
# ============================================================

windows = [5, 10, 20]

for window in windows:

    column_name = f"fx_vol_{window}obs"

    df[column_name] = (
        df.groupby("volatility_segment")["fx_log_return"]
        .transform(
            lambda x: x.rolling(
                window=window,
                min_periods=window
            ).std()
        )
    )


# ============================================================
# 5. VOLATILITY SUMMARY
# ============================================================

summary_rows = []

for window in windows:

    col = f"fx_vol_{window}obs"

    series = df[col].dropna()

    summary_rows.append({
        "window": window,
        "n": len(series),
        "mean": series.mean(),
        "std": series.std(),
        "min": series.min(),
        "median": series.median(),
        "max": series.max(),
        "p75": series.quantile(0.75),
        "p90": series.quantile(0.90),
        "p95": series.quantile(0.95),
        "p99": series.quantile(0.99)
    })


volatility_summary = pd.DataFrame(summary_rows)

print("\nRolling volatility summary:")
print(volatility_summary)


# ============================================================
# 6. VOLATILITY CORRELATIONS
# ============================================================

volatility_columns = [
    "fx_vol_5obs",
    "fx_vol_10obs",
    "fx_vol_20obs"
]

volatility_correlation = df[volatility_columns].corr()

print("\nVolatility correlations:")
print(volatility_correlation)


# ============================================================
# 7. TOP VOLATILITY PERIODS
# ============================================================

top_periods = (
    df[
        [
            "date",
            "fx_vol_5obs",
            "fx_vol_10obs",
            "fx_vol_20obs"
        ]
    ]
    .dropna(subset=["fx_vol_20obs"])
    .sort_values("fx_vol_20obs", ascending=False)
    .head(20)
)

print("\nTop 20-observation volatility periods:")
print(top_periods)


# ============================================================
# 8. SAVE RESULTS
# ============================================================

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

volatility_summary.to_csv(
    "results/volatility_window_summary.csv",
    index=False
)

volatility_correlation.to_csv(
    "results/volatility_window_correlations.csv"
)

top_periods.to_csv(
    "results/top_volatility_periods.csv",
    index=False
)


# ============================================================
# 9. SAVE UPDATED DATASET
# ============================================================

output_file = (
    "data_cleaned/daily_volatility_dataset_2006_2025.csv"
)

df.to_csv(output_file, index=False)


# ============================================================
# 10. PLOT VOLATILITY WINDOWS
# ============================================================

plt.figure(figsize=(14, 7))

plt.plot(
    df["date"],
    df["fx_vol_5obs"],
    label="5-observation volatility"
)

plt.plot(
    df["date"],
    df["fx_vol_10obs"],
    label="10-observation volatility"
)

plt.plot(
    df["date"],
    df["fx_vol_20obs"],
    label="20-observation volatility"
)

plt.xlabel("Date")
plt.ylabel("Rolling volatility")
plt.title("Nigerian Naira–USD Rolling Volatility")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "figures/fx_rolling_volatility_comparison.png",
    dpi=300
)

plt.show()


showdf =  pd.read_csv(output_file);

print("\nUpdated volatility dataset saved to:")
print(output_file)