# ============================================================
# STEP 08D: ARIMA-GARCH COMMON-SAMPLE ROC-AUC
# ============================================================

import pandas as pd
from sklearn.metrics import roc_auc_score


INPUT_FILE = (
    "results/arima_garch_common_sample_forecasts.csv"
)

METRICS_FILE = (
    "results/arima_garch_common_sample_metrics.csv"
)


df = pd.read_csv(INPUT_FILE)

if df.empty:
    raise ValueError("ARIMA-GARCH common-sample forecast file is empty.")

auc = roc_auc_score(
    df["actual_spike"],
    df["predicted_volatility"],
)

print("\nARIMA-GARCH COMMON-SAMPLE ROC-AUC")
print("--------------------------------")
print("Forecast observations:", len(df))
print("ROC-AUC:", auc)

metrics = pd.read_csv(METRICS_FILE)
metrics["AUC"] = auc
metrics.to_csv(METRICS_FILE, index=False)

print("\nUpdated metrics saved:")
print(METRICS_FILE)
