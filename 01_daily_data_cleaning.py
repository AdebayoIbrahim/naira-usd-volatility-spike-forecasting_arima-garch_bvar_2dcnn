# -----------------------------
# 1. Imports
# -----------------------------
import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------
# 2. Project directories
# -----------------------------
BASE_DIR  = Path(".") #set as current path

#project_folders
DATASET_DIR = BASE_DIR / "dataset"
CLEANED_DIR = BASE_DIR / "data_cleaned"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

#raw files
EXCHANGE_FILE = DATASET_DIR / "usdexchangerate.xlsx"
BRENT_FILE = DATASET_DIR / "brentcrudeoil.xlsx"
RESERVES_FILE = DATASET_DIR / "externalreserves.xlsx"

# -----------------------------
# 4. Study window
# -----------------------------
# The external-reserves dataset starts later than the other two.
# Therefore the common usable period begins on 2006-04-03.
#
# The Brent file currently ends on 2025-10-31.
# Therefore this specific raw-data version currently ends there.
STUDY_START = pd.Timestamp("2006-04-03")
STUDY_END = pd.Timestamp("2025-10-31")


# ============================================================
# 5. Load the raw datasets
# ============================================================

exchange_raw = pd.read_excel(
    EXCHANGE_FILE,
    sheet_name="CBN USD Rates"
)

brent_raw = pd.read_excel(
    BRENT_FILE,
    sheet_name="Daily"
)

reserves_raw = pd.read_excel(
    RESERVES_FILE,
    sheet_name="Sheet1"
)


# ============================================================
# 6. Display raw dataset information
# ============================================================

print("\n" + "=" * 70)
print("RAW DATASET INFORMATION")
print("=" * 70)

print("\nCBN Exchange Rate:")
print("Shape:", exchange_raw.shape)
print("Columns:", exchange_raw.columns.tolist())

print("\nBrent Crude Oil:")
print("Shape:", brent_raw.shape)
print("Columns:", brent_raw.columns.tolist())

print("\nExternal Reserves:")
print("Shape:", reserves_raw.shape)
print("Columns:", reserves_raw.columns.tolist())


# ============================================================
# 7. Standardize column names
# ============================================================

exchange_raw.columns = (
    exchange_raw.columns
    .astype(str)
    .str.strip()
    .str.lower()
)

brent_raw.columns = (
    brent_raw.columns
    .astype(str)
    .str.strip()
    .str.lower()
)

reserves_raw.columns = (
    reserves_raw.columns
    .astype(str)
    .str.strip()
)

# ============================================================
# 8. Clean exchange-rate dataset
# ============================================================

exchange = exchange_raw.copy()

exchange["date"] = pd.to_datetime(
    exchange["date"],
    errors="coerce"
)

# Keep only US Dollar observations
exchange = exchange[
    exchange["currency"]
    .astype(str)
    .str.upper()
    .str.strip()
    .eq("US DOLLAR")
].copy()

# Convert rate columns to numeric
for col in [
    "buying_rate",
    "central_rate",
    "selling_rate"
]:
    exchange[col] = pd.to_numeric(
        exchange[col],
        errors="coerce"
    )

# Remove invalid dates
exchange = exchange.dropna(
    subset=["date"]
)

# Remove duplicate dates
exchange = (
    exchange
    .sort_values("date")
    .drop_duplicates(subset=["date"], keep="last")
    .reset_index(drop=True)
)


# ============================================================
# 9. Clean Brent crude oil dataset
# ============================================================

brent = brent_raw.copy()

brent["observation_date"] = pd.to_datetime(
    brent["observation_date"],
    errors="coerce"
)

brent["dcoilbrenteu"] = pd.to_numeric(
    brent["dcoilbrenteu"],
    errors="coerce"
)

brent = brent.rename(
    columns={
        "observation_date": "date",
        "dcoilbrenteu": "brent_price"
    }
)

brent = brent.dropna(
    subset=["date"]
)

brent = (
    brent
    .sort_values("date")
    .drop_duplicates(subset=["date"], keep="last")
    .reset_index(drop=True)
)

# ============================================================
# 10. Clean external-reserves dataset
# ============================================================

reserves = reserves_raw.copy()

reserves["Date"] = pd.to_datetime(
    reserves["Date"],
    dayfirst=True,
    errors="coerce"
)

for col in [
    "Gross (USD)",
    "Liquid (USD)",
    "Blocked (USD)",
    "Block Percent (%)"
]:
    reserves[col] = pd.to_numeric(
        reserves[col],
        errors="coerce"
    )

reserves = reserves.rename(
    columns={
        "Date": "date",
        "Gross (USD)": "gross_reserves",
        "Liquid (USD)": "liquid_reserves",
        "Blocked (USD)": "blocked_reserves",
        "Block Percent (%)": "blocked_percent"
    }
)

reserves = reserves.dropna(
    subset=["date"]
)

reserves = (
    reserves
    .sort_values("date")
    .drop_duplicates(subset=["date"], keep="last")
    .reset_index(drop=True)
)


# ============================================================
# Correct verified reserve data-entry anomalies
# ============================================================

# 2007-11-20:
# Gross reserves was recorded as 508.9996 billion,
# while the liquid + blocked components indicate
# the intended value is approximately 50.89996 billion.
reserves.loc[
    reserves["date"] == pd.Timestamp("2007-11-20"),
    "gross_reserves"
] = 50_899_960_000.0


# 2006-11-07:
# Liquid reserves was recorded as 399.4951 billion,
# while the surrounding values and reserve components
# indicate the intended value is approximately 39.94951 billion.
reserves.loc[
    reserves["date"] == pd.Timestamp("2006-11-07"),
    "liquid_reserves"
] = 39_949_510_000.0


print("\nCorrected reserve observations:")
print(
    reserves[
        reserves["date"].isin([
            pd.Timestamp("2007-11-20"),
            pd.Timestamp("2006-11-07")
        ])
    ][
        [
            "date",
            "gross_reserves",
            "liquid_reserves",
            "blocked_reserves",
            "blocked_percent"
        ]
    ].to_string(index=False)
)



# ============================================================
# 11. Report raw date ranges
# ============================================================

print("\n" + "=" * 70)
print("DATE RANGES AFTER BASIC CLEANING")
print("=" * 70)

print(
    "\nExchange rate:",
    exchange["date"].min(),
    "to",
    exchange["date"].max()
)

print(
    "Brent:",
    brent["date"].min(),
    "to",
    brent["date"].max()
)

print(
    "External reserves:",
    reserves["date"].min(),
    "to",
    reserves["date"].max()
)


# ============================================================
# 12. Restrict all datasets to common study window
# ============================================================

exchange = exchange[
    (exchange["date"] >= STUDY_START) &
    (exchange["date"] <= STUDY_END)
].copy()

brent = brent[
    (brent["date"] >= STUDY_START) &
    (brent["date"] <= STUDY_END)
].copy()

reserves = reserves[
    (reserves["date"] >= STUDY_START) &
    (reserves["date"] <= STUDY_END)
].copy()


# ============================================================
# 13. Keep the variables we currently need
# ============================================================

exchange = exchange[
    [
        "date",
        "buying_rate",
        "central_rate",
        "selling_rate"
    ]
].copy()

brent = brent[
    [
        "date",
        "brent_price"
    ]
].copy()

reserves = reserves[
    [
        "date",
        "gross_reserves",
        "liquid_reserves",
        "blocked_reserves",
        "blocked_percent"
    ]
].copy()


# ============================================================
# 14. Missing values BEFORE merging
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES BEFORE MERGING")
print("=" * 70)

print("\nExchange rate:")
print(exchange.isna().sum())

print("\nBrent:")
print(brent.isna().sum())

print("\nExternal reserves:")
print(reserves.isna().sum())



# ============================================================
# Inspect missing Brent observations
# ============================================================

missing_brent = brent[
    brent["brent_price"].isna()
].copy()

print("\n" + "=" * 70)
print("MISSING BRENT OBSERVATIONS")
print("=" * 70)

print("\nNumber of missing Brent values:")
print(len(missing_brent))

print("\nFirst 30 missing Brent dates:")
print(
    missing_brent.head(30).to_string(index=False)
)

print("\nLast 30 missing Brent dates:")
print(
    missing_brent.tail(30).to_string(index=False)
)

# Day-of-week distribution
missing_brent["day_of_week"] = (
    missing_brent["date"]
    .dt.day_name()
)

print("\nMissing Brent observations by day of week:")
print(
    missing_brent["day_of_week"]
    .value_counts()
)

# Monthly distribution
missing_brent["year_month"] = (
    missing_brent["date"]
    .dt.to_period("M")
    .astype(str)
)

print("\nMissing Brent observations by year:")
print(
    missing_brent["date"]
    .dt.year
    .value_counts()
    .sort_index()
)

# ============================================================
# 15. Merge the three datasets
# ============================================================

# We use an INNER JOIN initially.
# This keeps only dates for which all three datasets
# actually contain observations.
merged = (
    exchange
    .merge(
        brent,
        on="date",
        how="inner"
    )
    .merge(
        reserves,
        on="date",
        how="inner"
    )
)

merged = (
    merged
    .sort_values("date")
    .reset_index(drop=True)
)


# ============================================================
# 16. Basic diagnostics after merge
# ============================================================

print("\n" + "=" * 70)
print("MERGED DATASET")
print("=" * 70)

print("\nShape:")
print(merged.shape)

print("\nDate range:")
print(
    merged["date"].min(),
    "to",
    merged["date"].max()
)

print("\nMissing values:")
print(merged.isna().sum())

print("\nDuplicate dates:")
print(
    merged["date"].duplicated().sum()
)



# ============================================================
# Inspect remaining missing Brent values after merge
# ============================================================

missing_brent_merged = merged[
    merged["brent_price"].isna()
].copy()

print("\n" + "=" * 70)
print("REMAINING MISSING BRENT VALUES AFTER MERGE")
print("=" * 70)

print("\nNumber of missing Brent observations:")
print(len(missing_brent_merged))

print("\nMissing Brent dates:")
print(
    missing_brent_merged[
        ["date", "buying_rate", "gross_reserves"]
    ].to_string(index=False)
)

print("\nMissing Brent by day of week:")
print(
    missing_brent_merged["date"]
    .dt.day_name()
    .value_counts()
)

print("\nMissing Brent by year:")
print(
    missing_brent_merged["date"]
    .dt.year
    .value_counts()
    .sort_index()
)

# ============================================================
# Check date gaps in merged dataset
# ============================================================

date_diff = merged["date"].diff().dt.days

print("\n" + "=" * 70)
print("DATE GAP DIAGNOSTICS")
print("=" * 70)

print("\nGap distribution:")
print(
    date_diff.value_counts().sort_index()
)

print("\nLargest gap between consecutive observations:")
print(
    date_diff.max(),
    "days"
)

print("\nDates with gaps greater than 5 days:")
large_gaps = merged.loc[
    date_diff > 5,
    ["date"]
].copy()

print(large_gaps)

# ============================================================
# Investigate long gaps
# ============================================================

date_diff = merged["date"].diff().dt.days

long_gap_indices = date_diff[
    date_diff > 5
].index

print("\n" + "=" * 70)
print("LONG GAP INVESTIGATION")
print("=" * 70)

for idx in long_gap_indices:

    previous_date = merged.loc[idx - 1, "date"]
    current_date = merged.loc[idx, "date"]
    gap_days = date_diff.loc[idx]

    print("\n" + "-" * 60)
    print("Previous observation:", previous_date)
    print("Current observation :", current_date)
    print("Gap:", gap_days, "days")

    print("\nPrevious row:")
    print(
        merged.loc[idx - 1].to_string()
    )

    print("\nCurrent row:")
    print(
        merged.loc[idx].to_string()
    )
    
    
    
    # ============================================================
# 18. Create observation-gap indicator
# ============================================================

merged["gap_days"] = (
    merged["date"]
    .diff()
    .dt.days
)

print("\n" + "=" * 70)
print("OBSERVATION GAP SUMMARY")
print("=" * 70)

print("\nRows with gaps greater than 1 day:")
print((merged["gap_days"] > 1).sum())

print("\nRows with gaps greater than 3 days:")
print((merged["gap_days"] > 3).sum())

print("\nRows with gaps greater than 5 days:")
print((merged["gap_days"] > 5).sum())

print("\nLargest gap:")
print(merged["gap_days"].max(), "days")


# ============================================================
# 19. Save cleaned individual datasets
# ============================================================

exchange_output = (
    CLEANED_DIR /
    "clean_daily_exchange_rate_2006_2025.csv"
)

brent_output = (
    CLEANED_DIR /
    "clean_daily_brent_2006_2025.csv"
)

reserves_output = (
    CLEANED_DIR /
    "clean_daily_external_reserves_2006_2025.csv"
)

exchange.to_csv(
    exchange_output,
    index=False
)

brent.to_csv(
    brent_output,
    index=False
)

reserves.to_csv(
    reserves_output,
    index=False
)


# ============================================================
# 20. Save the merged dataset
# ============================================================

# This is the current cleaned merged source dataset.
# We are NOT yet calling it the final modelling dataset
# because the Brent missing-value treatment and time-series
# transformations have not been finalized.

merged_output = (
    CLEANED_DIR /
    "merged_daily_macro_dataset_2006_2025.csv"
)

merged.to_csv(
    merged_output,
    index=False
)

print("\n" + "=" * 70)
print("CLEANED FILES SAVED")
print("=" * 70)

print("\nClean exchange-rate dataset:")
print(exchange_output)

print("\nClean Brent dataset:")
print(brent_output)

print("\nClean external-reserves dataset:")
print(reserves_output)

print("\nMerged daily dataset:")
print(merged_output)


# ============================================================
# 21. Verify saved merged dataset
# ============================================================

check = pd.read_csv(merged_output)

check["date"] = pd.to_datetime(check["date"])

print("\n" + "=" * 70)
print("SAVED MERGED DATASET VERIFICATION")
print("=" * 70)

print("\nShape:")
print(check.shape)

print("\nDate range:")
print(check["date"].min(), "to", check["date"].max())

print("\nMissing values:")
print(check.isna().sum())

print("\nDuplicate dates:")
print(check["date"].duplicated().sum())

print("\nDaily data cleaning and initial merge completed successfully.")