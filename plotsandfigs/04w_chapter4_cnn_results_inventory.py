#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 16:36:34 2026

@author: setup
"""

from pathlib import Path
import pandas as pd


# ============================================================
# CHAPTER 4 - CNN RESULTS
# Inventory existing CNN result files
# ============================================================


PROJECT_ROOT = Path.cwd()
RESULTS_DIR = PROJECT_ROOT / "results"


if not RESULTS_DIR.exists():
    raise FileNotFoundError(
        f"Results folder not found:\n{RESULTS_DIR}"
    )


print("\n" + "=" * 80)
print("CNN RESULT FILES IN RESULTS FOLDER")
print("=" * 80)


# ------------------------------------------------------------
# 1. Find all CNN-related files
# ------------------------------------------------------------
all_files = sorted(
    RESULTS_DIR.iterdir(),
    key=lambda path: path.name.lower()
)

cnn_files = [
    path
    for path in all_files
    if "cnn" in path.name.lower()
]


if not cnn_files:
    print("\nNo CNN-related files were found.")
    raise SystemExit


# ------------------------------------------------------------
# 2. Print filenames and basic CSV structure
# ------------------------------------------------------------
for path in cnn_files:

    print("\n" + "-" * 80)
    print(f"FILE: {path.name}")

    if path.suffix.lower() != ".csv":
        print("Type: non-CSV")
        continue

    try:

        df = pd.read_csv(path)

        print(
            f"Rows: {len(df):,}"
        )

        print(
            f"Columns: {df.columns.tolist()}"
        )

        print("\nFirst 3 rows:")

        print(
            df.head(3).to_string(index=False)
        )

    except Exception as exc:

        print(
            f"Could not read CSV: {exc}"
        )


print("\n" + "=" * 80)
print(
    f"Total CNN-related files found: {len(cnn_files)}"
)
print("=" * 80)