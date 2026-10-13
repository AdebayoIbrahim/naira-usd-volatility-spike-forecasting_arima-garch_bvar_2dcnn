from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# CHAPTER 4 - EDA
# [FIG 4.6 HERE]
# Daily Nigeria External Reserves, 2006–2025
# ============================================================


# ------------------------------------------------------------
# 1. Project paths
# ------------------------------------------------------------
PROJECT_ROOT = Path.cwd()

DATA_PATH = (
    PROJECT_ROOT
    / "data_cleaned"
    / "analysis_ready_daily_dataset_2006_2025_corrected.csv"
)

FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


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
# 4. Check required columns
# ------------------------------------------------------------
required_columns = [
    "date",
    "gross_reserves"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}\n\n"
        f"Available columns:\n{list(df.columns)}"
    )


# ------------------------------------------------------------
# 5. Prepare data
# ------------------------------------------------------------
plot_df = (
    df[["date", "gross_reserves"]]
    .dropna()
    .copy()
)

# Convert from USD to billions of USD
plot_df["reserves_billion_usd"] = (
    plot_df["gross_reserves"] / 1e9
)


# ------------------------------------------------------------
# 6. Create Figure 4.6
# ------------------------------------------------------------
fig, ax = plt.subplots(
    figsize=(13, 5.8),
    dpi=150
)

ax.plot(
    plot_df["date"],
    plot_df["reserves_billion_usd"],
    color="#176B87",
    linewidth=1.6
)


# ------------------------------------------------------------
# 7. Title and labels
# ------------------------------------------------------------
ax.set_title(
    "Daily Nigeria External Reserves, 2006–2025",
    fontsize=15,
    fontweight="bold",
    pad=12
)

ax.set_xlabel(
    "Date",
    fontsize=11
)

ax.set_ylabel(
    "Gross External Reserves (US$ Billion)",
    fontsize=11
)


# ------------------------------------------------------------
# 8. Date formatting
# ------------------------------------------------------------
ax.xaxis.set_major_locator(
    mdates.YearLocator(2)
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%Y")
)

plt.setp(
    ax.get_xticklabels(),
    rotation=45,
    ha="right"
)


# ------------------------------------------------------------
# 9. Grid and axes
# ------------------------------------------------------------
ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.35
)

ax.grid(
    axis="x",
    visible=False
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# ------------------------------------------------------------
# 10. Layout
# ------------------------------------------------------------
fig.tight_layout()


# ------------------------------------------------------------
# 11. Save Figure 4.6
# ------------------------------------------------------------
figure_path = (
    FIGURES_DIR
    / "figure_4_6_external_reserves_level.png"
)

fig.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ------------------------------------------------------------
# 12. Print verification
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("[FIG 4.6 HERE]")
print("=" * 70)

print("Daily Nigeria External Reserves")

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
    f"Minimum reserves: "
    f"{plot_df['reserves_billion_usd'].min():.4f} "
    f"billion USD"
)

print(
    f"Maximum reserves: "
    f"{plot_df['reserves_billion_usd'].max():.4f} "
    f"billion USD"
)

print("\nSaved to:")
print(figure_path)

print("\nCompleted successfully.")