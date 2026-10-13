#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 05:53:55 2026

@author: setup
"""

from pathlib import Path


# ============================================================
# CHAPTER 4 - BVAR RESULTS
# Find existing BVAR validation/result files
# ============================================================

PROJECT_ROOT = Path.cwd()
RESULTS_DIR = PROJECT_ROOT / "results"


if not RESULTS_DIR.exists():
    raise FileNotFoundError(
        f"Results folder not found:\n{RESULTS_DIR}"
    )


print("\n" + "=" * 75)
print("BVAR RESULT FILES IN RESULTS FOLDER")
print("=" * 75)

files = sorted(
    RESULTS_DIR.glob("*.csv"),
    key=lambda p: p.name.lower()
)

bvar_files = [
    path for path in files
    if "bvar" in path.name.lower()
    or "var" in path.name.lower()
]

if not bvar_files:
    print("\nNo BVAR/VAR CSV files found.")
else:
    for path in bvar_files:
        print(path.name)

print("\nTotal files found:", len(bvar_files))