#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 05:10:46 2026

@author: setup
"""

from pathlib import Path
import warnings

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox


# ============================================================
# CHAPTER 4 - ARIMA-GARCH RESULTS
# [TABLE 4.4 HERE]
# ARIMA Candidate Model Selection
# ============================================================


# ------------------------------------------------------------
# 1. Project paths
# ------------------------------------------------------------
PROJECT_ROOT = Path.cwd()

DATA_PATH = (
    PROJECT_ROOT
    / "data_cleaned"
    / "forecasting_dataset_2006_2025_spike_gap_target_safe.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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
# 4. Select training-period FX returns
# ------------------------------------------------------------
TRAIN_END = pd.Timestamp("2020-12-31")

train = df.loc[
    df["date"] <= TRAIN_END,
    ["date", "fx_log_return"]
].copy()

# Remove missing return values
train = train.dropna(
    subset=["fx_log_return"]
)

y = train["fx_log_return"].astype(float)

print("\nTraining period:")
print(f"Start: {train['date'].min().date()}")
print(f"End:   {train['date'].max().date()}")
print(f"Observations used: {len(y):,}")


# ------------------------------------------------------------
# 5. Candidate ARIMA models
# ------------------------------------------------------------
candidates = []

for p in range(4):
    for q in range(4):

        order = (p, 0, q)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            try:
                model = ARIMA(
                    y,
                    order=order,
                    trend="c"
                )

                fitted = model.fit()

                # Residual diagnostic
                residuals = fitted.resid.dropna()

                lb = acorr_ljungbox(
                    residuals,
                    lags=[10],
                    return_df=True
                )

                candidates.append(
                    {
                        "Model": f"ARIMA({p},0,{q})",
                        "p": p,
                        "d": 0,
                        "q": q,
                        "AIC": fitted.aic,
                        "BIC": fitted.bic,
                        "Ljung_Box_Lag10": lb["lb_stat"].iloc[0],
                        "Ljung_Box_pvalue_Lag10": lb["lb_pvalue"].iloc[0],
                        "Converged": True
                    }
                )

            except Exception as exc:

                candidates.append(
                    {
                        "Model": f"ARIMA({p},0,{q})",
                        "p": p,
                        "d": 0,
                        "q": q,
                        "AIC": float("nan"),
                        "BIC": float("nan"),
                        "Ljung_Box_Lag10": float("nan"),
                        "Ljung_Box_pvalue_Lag10": float("nan"),
                        "Converged": False
                    }
                )

                print(
                    f"\nFailed: ARIMA({p},0,{q})"
                    f"\nReason: {exc}"
                )


# ------------------------------------------------------------
# 6. Create results table
# ------------------------------------------------------------
results = pd.DataFrame(candidates)

results = results.sort_values(
    by="AIC",
    na_position="last"
).reset_index(drop=True)


# ------------------------------------------------------------
# 7. Identify preferred model
# ------------------------------------------------------------
valid_results = results[
    results["Converged"]
    & results["AIC"].notna()
].copy()

best_aic_model = valid_results.loc[
    valid_results["AIC"].idxmin(),
    "Model"
]

best_bic_model = valid_results.loc[
    valid_results["BIC"].idxmin(),
    "Model"
]

# Mark the model selected by AIC
results["Selected_by_AIC"] = (
    results["Model"] == best_aic_model
)


# ------------------------------------------------------------
# 8. Presentation rounding
# ------------------------------------------------------------
results_display = results.copy()

results_display["AIC"] = (
    results_display["AIC"].round(6)
)

results_display["BIC"] = (
    results_display["BIC"].round(6)
)

results_display["Ljung_Box_Lag10"] = (
    results_display["Ljung_Box_Lag10"].round(6)
)

results_display["Ljung_Box_pvalue_Lag10"] = (
    results_display["Ljung_Box_pvalue_Lag10"].round(6)
)


# ------------------------------------------------------------
# 9. Save CSV
# ------------------------------------------------------------
csv_path = (
    RESULTS_DIR
    / "table_4_4_arima_model_selection.csv"
)

results_display.to_csv(
    csv_path,
    index=False
)


# ------------------------------------------------------------
# 10. Save LaTeX
# ------------------------------------------------------------
latex_path = (
    RESULTS_DIR
    / "table_4_4_arima_model_selection.tex"
)

latex_table = results_display[
    [
        "Model",
        "AIC",
        "BIC",
        "Ljung_Box_Lag10",
        "Ljung_Box_pvalue_Lag10",
        "Selected_by_AIC"
    ]
].to_latex(
    index=False,
    escape=False,
    caption="ARIMA Candidate Model Selection",
    label="tab:4.4"
)

latex_path.write_text(
    latex_table,
    encoding="utf-8"
)


# ------------------------------------------------------------
# 11. Print results
# ------------------------------------------------------------
print("\n" + "=" * 75)
print("[TABLE 4.4 HERE]")
print("=" * 75)

print("ARIMA Candidate Model Selection")

print("\n")
print(
    results_display[
        [
            "Model",
            "AIC",
            "BIC",
            "Ljung_Box_Lag10",
            "Ljung_Box_pvalue_Lag10",
            "Selected_by_AIC"
        ]
    ].to_string(index=False)
)

print("\nBest model by AIC:")
print(best_aic_model)

print("\nBest model by BIC:")
print(best_bic_model)

print("\nCSV saved to:")
print(csv_path)

print("\nLaTeX saved to:")
print(latex_path)

print("\nCompleted successfully.")