#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 05:14:33 2026

@author: setup
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model


# ============================================================
# CHAPTER 4 - ARIMA-GARCH RESULTS
# [TABLE 4.5 HERE]
# GARCH Candidate Model Selection
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
# 2. Load data
# ------------------------------------------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["date"]
)

df = (
    df.sort_values("date")
      .reset_index(drop=True)
)


# ------------------------------------------------------------
# 3. Training-period FX returns
# ------------------------------------------------------------
TRAIN_END = pd.Timestamp("2020-12-31")

train = df.loc[
    df["date"] <= TRAIN_END,
    ["date", "fx_log_return"]
].dropna()

y = train["fx_log_return"].astype(float)


# ------------------------------------------------------------
# 4. Fit selected ARIMA mean equation
# ------------------------------------------------------------
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    arima_model = ARIMA(
        y,
        order=(0, 0, 1),
        trend="c"
    )

    arima_fit = arima_model.fit()


# ------------------------------------------------------------
# 5. Obtain ARIMA residuals
# ------------------------------------------------------------
residuals = (
    arima_fit.resid
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .astype(float)
)


# ------------------------------------------------------------
# 6. Candidate GARCH specifications
# ------------------------------------------------------------
garch_orders = [
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2)
]

distributions = [
    "normal",
    "t"
]


results = []


# ------------------------------------------------------------
# 7. Fit candidate GARCH models
# ------------------------------------------------------------
for p, q in garch_orders:

    for dist in distributions:

        label = f"GARCH({p},{q})-{dist}"

        try:

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                model = arch_model(
                    residuals,
                    mean="Zero",
                    vol="GARCH",
                    p=p,
                    q=q,
                    dist=dist,
                    rescale=True
                )

                fit = model.fit(
                    disp="off",
                    update_freq=0
                )

            # ------------------------------------------------
            # Standardized residuals
            # ------------------------------------------------
            std_resid = pd.Series(
                fit.std_resid
            ).replace(
                [np.inf, -np.inf],
                np.nan
            ).dropna()

            # ------------------------------------------------
            # Ljung-Box on standardized residuals
            # ------------------------------------------------
            lb_resid = acorr_ljungbox(
                std_resid,
                lags=[10],
                return_df=True
            )

            # ------------------------------------------------
            # Ljung-Box on squared standardized residuals
            # ------------------------------------------------
            lb_sq = acorr_ljungbox(
                std_resid ** 2,
                lags=[10],
                return_df=True
            )

            results.append(
                {
                    "Model": label,
                    "p": p,
                    "q": q,
                    "Distribution": dist,
                    "AIC": fit.aic,
                    "BIC": fit.bic,
                    "LogLikelihood": fit.loglikelihood,
                    "Converged": fit.convergence_flag == 0,
                    "Ljung_Box_pvalue_Residuals": (
                        lb_resid["lb_pvalue"].iloc[0]
                    ),
                    "Ljung_Box_pvalue_Squared_Residuals": (
                        lb_sq["lb_pvalue"].iloc[0]
                    )
                }
            )

        except Exception as exc:

            results.append(
                {
                    "Model": label,
                    "p": p,
                    "q": q,
                    "Distribution": dist,
                    "AIC": np.nan,
                    "BIC": np.nan,
                    "LogLikelihood": np.nan,
                    "Converged": False,
                    "Ljung_Box_pvalue_Residuals": np.nan,
                    "Ljung_Box_pvalue_Squared_Residuals": np.nan
                }
            )

            print(
                f"\nFailed: {label}"
                f"\nReason: {exc}"
            )


# ------------------------------------------------------------
# 8. Create results table
# ------------------------------------------------------------
results_df = pd.DataFrame(results)

valid = results_df[
    (results_df["Converged"])
    & results_df["AIC"].notna()
].copy()

if valid.empty:
    raise RuntimeError(
        "No GARCH model converged successfully."
    )


# ------------------------------------------------------------
# 9. Identify preferred model
# ------------------------------------------------------------
best_aic_idx = valid["AIC"].idxmin()
best_bic_idx = valid["BIC"].idxmin()

best_aic_model = valid.loc[
    best_aic_idx,
    "Model"
]

best_bic_model = valid.loc[
    best_bic_idx,
    "Model"
]


results_df["Selected_by_AIC"] = (
    results_df["Model"] == best_aic_model
)


# ------------------------------------------------------------
# 10. Presentation rounding
# ------------------------------------------------------------
results_display = results_df.copy()

for col in [
    "AIC",
    "BIC",
    "LogLikelihood",
    "Ljung_Box_pvalue_Residuals",
    "Ljung_Box_pvalue_Squared_Residuals"
]:
    results_display[col] = (
        results_display[col].round(6)
    )


# ------------------------------------------------------------
# 11. Save CSV
# ------------------------------------------------------------
csv_path = (
    RESULTS_DIR
    / "table_4_5_garch_model_selection.csv"
)

results_display.to_csv(
    csv_path,
    index=False
)


# ------------------------------------------------------------
# 12. Save LaTeX
# ------------------------------------------------------------
latex_path = (
    RESULTS_DIR
    / "table_4_5_garch_model_selection.tex"
)

latex_table = results_display[
    [
        "Model",
        "Distribution",
        "AIC",
        "BIC",
        "Converged",
        "Ljung_Box_pvalue_Residuals",
        "Ljung_Box_pvalue_Squared_Residuals",
        "Selected_by_AIC"
    ]
].to_latex(
    index=False,
    escape=False,
    caption="GARCH Candidate Model Selection",
    label="tab:4.5"
)

latex_path.write_text(
    latex_table,
    encoding="utf-8"
)


# ------------------------------------------------------------
# 13. Print results
# ------------------------------------------------------------
print("\n" + "=" * 85)
print("[TABLE 4.5 HERE]")
print("=" * 85)

print("GARCH Candidate Model Selection")

print("\n")
print(
    results_display[
        [
            "Model",
            "Distribution",
            "AIC",
            "BIC",
            "Converged",
            "Ljung_Box_pvalue_Residuals",
            "Ljung_Box_pvalue_Squared_Residuals",
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