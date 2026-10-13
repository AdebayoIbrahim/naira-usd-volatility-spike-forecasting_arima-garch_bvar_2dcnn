
# ============================================================
# STEP 10: FINAL BVAR OOS VISUALIZATIONS
# ============================================================
# Plot 1: Zoomed actual vs predicted volatility only
# Plot 2: Monthly actual vs predicted spike-rate trends
#
# Uses existing forecasts only. Does not refit the BVAR.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# 1. SETTINGS
# ============================================================

INPUT_FILE = Path(
    "results/bvar8_final_log1p_common_sample_forecasts.csv"
)

OUTPUT_DIR = Path("results/figures")

SPIKE_THRESHOLD = 0.10927724
PROBABILITY_CUTOFF = 0.50

ZOOM_MAX = 0.50
EXPECTED_OBSERVATIONS = 1083

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD EXISTING OUT-OF-SAMPLE FORECASTS
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Forecast file not found: {INPUT_FILE}\n"
        "Run Step 09S first."
    )

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "target_date",
    "actual_volatility",
    "forecast_volatility",
    "actual_spike",
    "predicted_spike",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

df["target_date"] = pd.to_datetime(
    df["target_date"],
    errors="coerce"
)

numeric_columns = [
    "actual_volatility",
    "forecast_volatility",
    "actual_spike",
    "predicted_spike",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df = (
    df.dropna(subset=required_columns)
      .sort_values("target_date")
      .reset_index(drop=True)
)

if len(df) != EXPECTED_OBSERVATIONS:
    raise ValueError(
        f"Expected {EXPECTED_OBSERVATIONS} complete "
        f"forecast observations, found {len(df)}. "
        "Check the saved forecast file before plotting."
    )

for col in ["actual_spike", "predicted_spike"]:
    if not df[col].isin([0, 1]).all():
        raise ValueError(
            f"{col} must contain only binary 0/1 labels."
        )

df["actual_spike"] = df["actual_spike"].astype(int)
df["predicted_spike"] = df["predicted_spike"].astype(int)

dates = df["target_date"]
actual_vol = df["actual_volatility"]
forecast_vol = df["forecast_volatility"]

actual = df["actual_spike"]
predicted = df["predicted_spike"]

print("=" * 70)
print("BVAR(8)-LOG1P OUT-OF-SAMPLE VISUALIZATIONS")
print("=" * 70)
print(f"Observations: {len(df):,}")
print(f"First target date: {dates.min().date()}")
print(f"Last target date:  {dates.max().date()}")


# ============================================================
# 3. CLASSIFICATION METRICS
# ============================================================

tp = int(((actual == 1) & (predicted == 1)).sum())
fp = int(((actual == 0) & (predicted == 1)).sum())
fn = int(((actual == 1) & (predicted == 0)).sum())
tn = int(((actual == 0) & (predicted == 0)).sum())

precision = tp / (tp + fp) if tp + fp else np.nan
recall = tp / (tp + fn) if tp + fn else np.nan

f1 = (
    2 * precision * recall / (precision + recall)
    if np.isfinite(precision + recall)
    and precision + recall > 0
    else np.nan
)

accuracy = (tp + tn) / len(df)

print("\nSPIKE CLASSIFICATION")
print("-" * 40)
print(f"Actual spikes:    {actual.sum():,}")
print(f"Predicted spikes: {predicted.sum():,}")
print(f"TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"Accuracy:  {accuracy:.4f}")


# ============================================================
# 4. PLOT ONE: ZOOMED VOLATILITY COMPARISON ONLY
# ============================================================
# Cap values at 0.50 FOR DISPLAY ONLY.
# Original forecasts and evaluation values are unchanged.
# ============================================================

actual_zoom = actual_vol.clip(lower=0, upper=ZOOM_MAX)
forecast_zoom = forecast_vol.clip(lower=0, upper=ZOOM_MAX)

actual_clipped = int((actual_vol > ZOOM_MAX).sum())
forecast_clipped = int((forecast_vol > ZOOM_MAX).sum())

fig, ax = plt.subplots(
    figsize=(14, 6),
    constrained_layout=True,
)

ax.plot(
    dates,
    actual_zoom,
    label="Actual volatility",
    linewidth=1.0,
)

ax.plot(
    dates,
    forecast_zoom,
    label="BVAR predicted volatility",
    linewidth=1.0,
)

ax.axhline(
    SPIKE_THRESHOLD,
    linestyle="--",
    linewidth=1.3,
    label=f"Spike threshold ({SPIKE_THRESHOLD:.4f})",
)

ax.set_ylim(0, ZOOM_MAX * 1.12)

ax.set_title(
    "BVAR(8)-log1p: Actual vs Predicted Volatility "
    "(Zoomed Out-of-Sample View)",
    fontsize=13,
    fontweight="bold",
)

ax.set_xlabel("Target date")
ax.set_ylabel("Squared FX log return ($r_t^2$)")

ax.xaxis.set_major_locator(
    mdates.AutoDateLocator(minticks=6, maxticks=12)
)
ax.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(
        ax.xaxis.get_major_locator()
    )
)

ax.grid(True, alpha=0.25)
ax.legend(loc="upper left")

ax.text(
    0.99,
    0.96,
    (
        f"Display cap: {ZOOM_MAX:.2f}\n"
        f"Actual observations capped: {actual_clipped}\n"
        f"Forecast observations capped: {forecast_clipped}\n"
        "Values above the cap are clipped for display only."
    ),
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=9,
    bbox=dict(boxstyle="round", alpha=0.12),
)

volatility_path = (
    OUTPUT_DIR / "bvar_actual_vs_predicted_volatility.png"
)

fig.savefig(
    volatility_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)

print(f"\nSaved: {volatility_path}")


# ============================================================
# 5. PLOT TWO: MONTHLY SPIKE-RATE TRENDS
# ============================================================
# Compare the percentage of forecast observations labelled
# as spikes in each month.
#
# Using rates rather than raw counts accounts for months with
# different numbers of available forecast observations.
# ============================================================

monthly = (
    df.set_index("target_date")
      .resample("MS")
      .agg(
          actual_spike_rate=("actual_spike", "mean"),
          predicted_spike_rate=("predicted_spike", "mean"),
          forecast_observations=("actual_spike", "size"),
      )
)

monthly["actual_spike_rate"] *= 100
monthly["predicted_spike_rate"] *= 100

# Empty months remain NaN, so gaps in the evaluation sample
# are not incorrectly presented as zero-spike months.

fig, ax = plt.subplots(
    figsize=(14, 6),
    constrained_layout=True,
)

ax.plot(
    monthly.index,
    monthly["actual_spike_rate"],
    marker="o",
    markersize=4,
    linewidth=1.8,
    label="Actual monthly spike rate",
)

ax.plot(
    monthly.index,
    monthly["predicted_spike_rate"],
    marker="s",
    markersize=4,
    linewidth=1.8,
    label="Predicted monthly spike rate",
)

ax.set_title(
    "BVAR(8)-log1p: Actual vs Predicted Spike-Rate Trends",
    fontsize=13,
    fontweight="bold",
)

ax.set_xlabel("Month")
ax.set_ylabel("Spike rate (% of forecast observations)")
ax.set_ylim(bottom=0)

ax.xaxis.set_major_locator(
    mdates.AutoDateLocator(minticks=6, maxticks=12)
)
ax.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(
        ax.xaxis.get_major_locator()
    )
)

ax.grid(True, alpha=0.25)
ax.legend(loc="upper left")

metrics_text = (
    f"Actual spike threshold: {SPIKE_THRESHOLD:.8f}\n"
    f"Prediction probability cutoff: {PROBABILITY_CUTOFF:.2f}\n\n"
    f"TP: {tp}  |  FP: {fp}\n"
    f"FN: {fn}  |  TN: {tn}\n"
    f"Precision: {precision:.1%}\n"
    f"Recall: {recall:.1%}\n"
    f"F1-score: {f1:.4f}"
)

ax.text(
    0.99,
    0.96,
    metrics_text,
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=9,
    bbox=dict(boxstyle="round", alpha=0.12),
)

spike_path = (
    OUTPUT_DIR / "bvar_actual_vs_predicted_spike_trends.png"
)

fig.savefig(
    spike_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)

print(f"Saved: {spike_path}")

print("\nBoth revised plots completed.")
