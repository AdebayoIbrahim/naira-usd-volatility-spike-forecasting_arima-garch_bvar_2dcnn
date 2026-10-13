#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 05:51:12 2026

@author: setup
"""

from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# CHAPTER 4 - BVAR RESULTS
# [TABLE 4.7 HERE]
# BVAR Lag-Order Selection
# ============================================================


# ------------------------------------------------------------
# 1. Project paths
# ------------------------------------------------------------
PROJECT_ROOT = Path.cwd()

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "var_lag_selection_spike_corrected.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Check input file
# ------------------------------------------------------------
if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"VAR lag-selection file not found:\n{INPUT_PATH}"
    )


# ------------------------------------------------------------
# 3. Read file
# ------------------------------------------------------------
df = pd.read_csv(INPUT_PATH)

print("\nFile being used:")
print(INPUT_PATH)

print("\nAvailable columns:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 4. Helper for finding columns
# ------------------------------------------------------------
def find_column(dataframe, candidates):
    """
    Find a column by exact or partial case-insensitive match.
    """

    normalized = {
        column.lower().strip(): column
        for column in dataframe.columns
    }

    # Exact match
    for candidate in candidates:

        key = candidate.lower().strip()

        if key in normalized:
            return normalized[key]

    # Partial match
    for column in dataframe.columns:

        column_lower = column.lower()

        for candidate in candidates:

            if candidate.lower() in column_lower:
                return column

    return None


# ------------------------------------------------------------
# 5. Identify lag, AIC and BIC columns
# ------------------------------------------------------------
lag_col = find_column(
    df,
    [
        "lag",
        "lags",
        "p"
    ]
)

aic_col = find_column(
    df,
    [
        "aic",
        "AIC"
    ]
)

bic_col = find_column(
    df,
    [
        "bic",
        "BIC"
    ]
)


if lag_col is None or aic_col is None or bic_col is None:

    raise ValueError(
        "\nCould not identify the required lag-selection columns.\n\n"
        f"Lag column found: {lag_col}\n"
        f"AIC column found: {aic_col}\n"
        f"BIC column found: {bic_col}\n\n"
        f"Available columns:\n{df.columns.tolist()}"
    )


# ------------------------------------------------------------
# 6. Prepare the selection table
# ------------------------------------------------------------
table = df[
    [
        lag_col,
        aic_col,
        bic_col
    ]
].copy()

table.columns = [
    "Lag",
    "AIC",
    "BIC"
]


# ------------------------------------------------------------
# 7. Convert numeric values
# ------------------------------------------------------------
table["Lag"] = pd.to_numeric(
    table["Lag"],
    errors="coerce"
)

table["AIC"] = pd.to_numeric(
    table["AIC"],
    errors="coerce"
)

table["BIC"] = pd.to_numeric(
    table["BIC"],
    errors="coerce"
)


table = (
    table
    .dropna()
    .sort_values("Lag")
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# 8. Identify preferred lags
# ------------------------------------------------------------
best_aic_lag = int(
    table.loc[
        table["AIC"].idxmin(),
        "Lag"
    ]
)

best_bic_lag = int(
    table.loc[
        table["BIC"].idxmin(),
        "Lag"
    ]
)


# ------------------------------------------------------------
# 9. Mark selected lag
# ------------------------------------------------------------
table["Selected_by_BIC"] = (
    table["Lag"] == best_bic_lag
)


# ------------------------------------------------------------
# 10. Round values
# ------------------------------------------------------------
table["AIC"] = table["AIC"].round(6)
table["BIC"] = table["BIC"].round(6)


# ------------------------------------------------------------
# 11. Save CSV
# ------------------------------------------------------------
csv_path = (
    RESULTS_DIR
    / "table_4_7_bvar_lag_selection.csv"
)

table.to_csv(
    csv_path,
    index=False
)


# ------------------------------------------------------------
# 12. Save LaTeX
# ------------------------------------------------------------
latex_path = (
    RESULTS_DIR
    / "table_4_7_bvar_lag_selection.tex"
)

latex_table = table.to_latex(
    index=False,
    escape=False,
    caption="BVAR Lag-Order Selection",
    label="tab:4.7"
)

latex_path.write_text(
    latex_table,
    encoding="utf-8"
)


# ------------------------------------------------------------
# 13. Print results
# ------------------------------------------------------------
print("\n" + "=" * 75)
print("[TABLE 4.7 HERE]")
print("=" * 75)

print("BVAR Lag-Order Selection")

print("\n")
print(table.to_string(index=False))

print("\nBest lag by AIC:")
print(f"VAR({best_aic_lag})")

print("\nBest lag by BIC:")
print(f"VAR({best_bic_lag})")

print("\nCSV saved to:")
print(csv_path)

print("\nLaTeX saved to:")
print(latex_path)

print("\nCompleted successfully.")