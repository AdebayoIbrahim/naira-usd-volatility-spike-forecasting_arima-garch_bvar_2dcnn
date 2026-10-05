#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 18:10:16 2026

@author: setup
"""

# ============================================================
# STEP 10C-V5
# LEAKAGE-FREE 2D-CNN VALIDATION + ARCHITECTURE COMPARISON
# ============================================================
#
# PURPOSE
# -------
# Re-evaluate the 2D-CNN before the final 2021-2025 test.
#
# Main corrections:
#
# 1. Feature scaler is fitted ONLY on 2006-2013.
# 2. 2014-2020 is used strictly as validation.
# 3. Final 2021-2025 test remains completely untouched.
# 4. Compare two CNN kernel designs:
#
#       CNN-A: kernel = (2, 3)
#       CNN-B: kernel = (1, 3)
#
# 5. Architecture selection is based primarily on validation PR-AUC.
#
# The spike threshold remains frozen:
#
#       0.10927724
#
# The model predicts:
#
#       target_volatility = next-day squared FX return
#
# and:
#
#       target_spike = 1[target_volatility > threshold]
# ============================================================


import os

# Suppress most TensorFlow startup messages.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. SETTINGS
# ============================================================

INPUT_FILE = (
    "data_cleaned/"
    "forecasting_dataset_2006_2025_spike_gap_target_safe.csv"
)

RESULTS_DIR = "results"
MODELS_DIR = "models"

WINDOW_SIZE = 20

SPIKE_THRESHOLD = 0.10927724

TRAIN_END = pd.Timestamp("2013-12-31")

VALIDATION_START = pd.Timestamp("2014-01-01")

VALIDATION_END = pd.Timestamp("2020-12-31")

# Important:
# Final test is everything after 2020-12-31.
#
# It is NOT loaded for model selection.

FEATURES = [
    "fx_log_return",
    "fx_return_squared",
    "brent_log_return",
    "reserves_log_change",
]

TARGET_VOLATILITY = "target_volatility"

TARGET_SPIKE = "target_spike"

FEATURE_KERNELS = {
    "CNN_A_kernel_2x3": (2, 3),
    "CNN_B_kernel_1x3": (1, 3),
}

RANDOM_SEED = 2026

BATCH_SIZE = 32

EPOCHS = 100

LEARNING_RATE = 0.001

DROPOUT_RATE = 0.30

VOLATILITY_LOSS_WEIGHT = 0.30

SPIKE_LOSS_WEIGHT = 0.70

# Keep the same spike-aware training idea used in the previous
# successful multitask version.
NORMAL_WEIGHT = 1.0

SPIKE_WEIGHT = 5.0


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

random.seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)

tf.keras.utils.set_random_seed(
    RANDOM_SEED
)


# ============================================================
# 3. OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True,
)

os.makedirs(
    MODELS_DIR,
    exist_ok=True,
)


# ============================================================
# 4. LOAD DATA
# ============================================================

print("=" * 90)
print("STEP 10C-V5: LEAKAGE-FREE 2D-CNN VALIDATION")
print("=" * 90)

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=[
        "date",
        "target_date",
    ],
)

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)


# ============================================================
# 5. INPUT VALIDATION
# ============================================================

required_columns = [
    "date",
    "target_date",
    "segment_id",
    *FEATURES,
    TARGET_VOLATILITY,
    TARGET_SPIKE,
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns: "
        f"{missing_columns}"
    )


print("\n" + "-" * 90)
print("INPUT VALIDATION")
print("-" * 90)

print(
    f"Rows loaded: {len(df):,}"
)

print(
    f"Date range: "
    f"{df['date'].min().date()} "
    f"to "
    f"{df['date'].max().date()}"
)

print(
    f"Features: {FEATURES}"
)

print(
    f"Window size: {WINDOW_SIZE}"
)

print(
    f"Frozen spike threshold: "
    f"{SPIKE_THRESHOLD:.8f}"
)


# ============================================================
# 6. BASIC DATA CHECKS
# ============================================================

if df["date"].duplicated().any():

    raise ValueError(
        "Duplicate dates detected."
    )


if df[FEATURES].isna().any().any():

    raise ValueError(
        "Missing values found in predictor variables."
    )


# target rows at segment boundaries are expected to have
# missing targets because there is no valid next observation.

invalid_targets = df[
    df[TARGET_VOLATILITY].notna()
    & (
        df[TARGET_VOLATILITY]
        < 0
    )
]

if not invalid_targets.empty:

    raise ValueError(
        "Negative target volatility detected."
    )


# ============================================================
# 7. BUILD GAP-SAFE WINDOWS
# ============================================================
#
# For every valid origin:
#
#         [t-19, ..., t-1, t]
#                       ↓
#                    target
#                       ↓
#                       t+1
#
# The 20 observations are taken ONLY within the same segment.
#
# This prevents a long market-data gap from becoming a fake
# consecutive sequence.
# ============================================================

def build_windows(
    data,
    features,
    window_size,
):
    X_rows = []

    volatility_targets = []

    spike_targets = []

    origin_dates = []

    target_dates = []

    segment_ids = []

    skipped_missing_target = 0

    skipped_short_segment = 0

    for segment_id, segment in data.groupby(
        "segment_id",
        sort=True,
    ):

        segment = (
            segment
            .sort_values("date")
            .reset_index(drop=True)
        )

        if len(segment) <= window_size:

            skipped_short_segment += 1

            continue

        values = segment[
            features
        ].to_numpy(
            dtype=float
        )

        for position in range(
            window_size - 1,
            len(segment) - 1,
        ):

            origin_row = segment.iloc[position]

            next_row = segment.iloc[
                position + 1
            ]

            # The safe dataset should already remove targets that
            # cross a segment boundary.
            target_vol = (
                origin_row[TARGET_VOLATILITY]
            )

            target_spike = (
                origin_row[TARGET_SPIKE]
            )

            target_date = (
                origin_row["target_date"]
            )

            if (
                pd.isna(target_vol)
                or pd.isna(target_spike)
                or pd.isna(target_date)
            ):

                skipped_missing_target += 1

                continue

            # Safety check:
            # target_date must correspond to the next observation.
            if (
                target_date
                != next_row["date"]
            ):

                raise ValueError(
                    "Target-date mismatch at "
                    f"{origin_row['date']}. "
                    f"Expected {next_row['date']}, "
                    f"found {target_date}."
                )

            # 20 consecutive observations inside this segment.
            window = values[
                position - window_size + 1:
                position + 1
            ]

            if window.shape != (
                window_size,
                len(features),
            ):

                raise ValueError(
                    "Unexpected window shape."
                )

            # CNN layout:
            #
            # variables × time
            #
            # Current window:
            #
            # 20 rows × 4 variables
            #
            # Transpose so that final input becomes:
            #
            #       4 variables × 20 time points × 1 channel

            window = window.T

            X_rows.append(window)

            volatility_targets.append(
                float(target_vol)
            )

            spike_targets.append(
                int(target_spike)
            )

            origin_dates.append(
                origin_row["date"]
            )

            target_dates.append(
                target_date
            )

            segment_ids.append(
                segment_id
            )

    X = np.asarray(
        X_rows,
        dtype=np.float32,
    )

    y_vol = np.asarray(
        volatility_targets,
        dtype=np.float32,
    )

    y_spike = np.asarray(
        spike_targets,
        dtype=np.float32,
    )

    metadata = pd.DataFrame(
        {
            "origin_date": origin_dates,
            "target_date": target_dates,
            "segment_id": segment_ids,
        }
    )

    return (
        X,
        y_vol,
        y_spike,
        metadata,
        skipped_missing_target,
        skipped_short_segment,
    )


(
    X_raw,
    y_vol_raw,
    y_spike,
    metadata,
    skipped_missing_target,
    skipped_short_segment,
) = build_windows(
    df,
    FEATURES,
    WINDOW_SIZE,
)


# ============================================================
# 8. WINDOW CHECKS
# ============================================================

print("\n" + "-" * 90)
print("WINDOW CONSTRUCTION")
print("-" * 90)

print(
    f"Total valid windows: "
    f"{len(X_raw):,}"
)

print(
    f"Skipped missing-target origins: "
    f"{skipped_missing_target:,}"
)

print(
    f"Skipped short segments: "
    f"{skipped_short_segment:,}"
)

print(
    f"Raw window shape: "
    f"{X_raw.shape}"
)

if X_raw.ndim != 3:

    raise ValueError(
        "Expected raw windows with shape "
        "(n, variables, time)."
    )


# Add channel dimension:
#
# (n, 4, 20)
#       ↓
# (n, 4, 20, 1)

# ============================================================
# 9. DEFINE DEVELOPMENT / VALIDATION WINDOWS
# ============================================================

train_mask = (
    metadata["target_date"]
    <= TRAIN_END
)

validation_mask = (
    (metadata["target_date"] >= VALIDATION_START)
    &
    (metadata["target_date"] <= VALIDATION_END)
)

X_train_raw = X_raw[
    train_mask.to_numpy()
]

X_validation_raw = X_raw[
    validation_mask.to_numpy()
]

y_train_vol_raw = y_vol_raw[
    train_mask.to_numpy()
]

y_validation_vol_raw = y_vol_raw[
    validation_mask.to_numpy()
]

y_train_spike = y_spike[
    train_mask.to_numpy()
]

y_validation_spike = y_spike[
    validation_mask.to_numpy()
]

train_metadata = metadata.loc[
    train_mask
].reset_index(drop=True)

validation_metadata = metadata.loc[
    validation_mask
].reset_index(drop=True)


print("\n" + "-" * 90)
print("DEVELOPMENT / VALIDATION SPLIT")
print("-" * 90)

print(
    f"Training windows: "
    f"{len(X_train_raw):,}"
)

print(
    f"Validation windows: "
    f"{len(X_validation_raw):,}"
)

print(
    f"Training target period: "
    f"{train_metadata['target_date'].min().date()} "
    f"to "
    f"{train_metadata['target_date'].max().date()}"
)

print(
    f"Validation target period: "
    f"{validation_metadata['target_date'].min().date()} "
    f"to "
    f"{validation_metadata['target_date'].max().date()}"
)

print(
    f"Training spikes: "
    f"{int(y_train_spike.sum()):,}"
)

print(
    f"Validation spikes: "
    f"{int(y_validation_spike.sum()):,}"
)

print(
    f"Training spike rate: "
    f"{100 * y_train_spike.mean():.4f}%"
)

print(
    f"Validation spike rate: "
    f"{100 * y_validation_spike.mean():.4f}%"
)


if len(X_train_raw) == 0:

    raise ValueError(
        "No training windows."
    )


if len(X_validation_raw) == 0:

    raise ValueError(
        "No validation windows."
    )


if y_validation_spike.sum() == 0:

    raise ValueError(
        "Validation contains no spike events."
    )


# ============================================================
# 10. FIT SCALER ONLY ON TRAINING PERIOD
# ============================================================
#
# This is the key leakage correction.
#
# We do NOT fit the scaler on:
#
#       2006-2020
#
# Instead:
#
#       scaler.fit(2006-2013)
#
# and then apply it to validation.
#
# ============================================================

print("\n" + "-" * 90)
print("LEAKAGE-FREE FEATURE SCALING")
print("-" * 90)

train_feature_rows = df[
    df["date"] <= TRAIN_END
][FEATURES].to_numpy(
    dtype=np.float64
)

if np.isnan(
    train_feature_rows
).any():

    raise ValueError(
        "NaN values found in training features."
    )


scaler = StandardScaler()

scaler.fit(
    train_feature_rows
)


print(
    "Scaler fitted using ONLY "
    f"observations through {TRAIN_END.date()}."
)

print(
    "\nTraining scaler means:"
)

for name, value in zip(
    FEATURES,
    scaler.mean_,
):

    print(
        f"  {name:<25} "
        f"{value: .8f}"
    )


print(
    "\nTraining scaler standard deviations:"
)

for name, value in zip(
    FEATURES,
    scaler.scale_,
):

    print(
        f"  {name:<25} "
        f"{value: .8f}"
    )


# ============================================================
# 11. APPLY THE TRAINING SCALER TO WINDOWS
# ============================================================

def scale_windows(
    X,
    scaler,
):
    """
    X shape:
        (n, variables, time)

    StandardScaler expects:
        (observations, variables)

    So reshape the windows, transform, and restore.
    """

    n, variables, time = X.shape

    reshaped = (
        X
        .transpose(0, 2, 1)
        .reshape(
            -1,
            variables,
        )
    )

    scaled = scaler.transform(
        reshaped
    )

    scaled = (
        scaled
        .reshape(
            n,
            time,
            variables,
        )
        .transpose(0, 2, 1)
    )

    return scaled.astype(
        np.float32
    )


X_train_scaled = scale_windows(
    X_train_raw,
    scaler,
)

X_validation_scaled = scale_windows(
    X_validation_raw,
    scaler,
)


# Add CNN channel.

X_train = X_train_scaled[
    ..., np.newaxis
]

X_validation = X_validation_scaled[
    ..., np.newaxis
]


print(
    "\nTraining CNN shape:",
    X_train.shape
)

print(
    "Validation CNN shape:",
    X_validation.shape
)


# ============================================================
# 12. LOG1P VOLATILITY TARGET
# ============================================================

y_train_vol_log = np.log1p(
    np.maximum(
        y_train_vol_raw,
        0,
    )
)

y_validation_vol_log = np.log1p(
    np.maximum(
        y_validation_vol_raw,
        0,
    )
)


# ============================================================
# 13. CLASS WEIGHTS
# ============================================================
#
# Keep the same moderate spike weighting used in the previous
# multitask experiment rather than immediately introducing
# another hyperparameter.
#
# Normal = 1
# Spike  = 5
# ============================================================

train_sample_weights = np.where(
    y_train_spike == 1,
    SPIKE_WEIGHT,
    NORMAL_WEIGHT,
).astype(
    np.float32
)

print("\n" + "-" * 90)
print("SPIKE-AWARE TRAINING WEIGHTS")
print("-" * 90)

print(
    f"Normal weight: "
    f"{NORMAL_WEIGHT:.2f}"
)

print(
    f"Spike weight: "
    f"{SPIKE_WEIGHT:.2f}"
)


# ============================================================
# 14. MODEL BUILDER
# ============================================================

def build_multitask_cnn(
    input_shape,
    temporal_feature_kernel,
):

    tf.keras.backend.clear_session()

    inputs = tf.keras.Input(
        shape=input_shape,
        name="cnn_input",
    )

    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=temporal_feature_kernel,
        padding="same",
        activation="relu",
        name="conv2d_block1",
    )(inputs)

    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        name="maxpool_block1",
    )(x)

    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=temporal_feature_kernel,
        padding="same",
        activation="relu",
        name="conv2d_block2",
    )(x)

    x = tf.keras.layers.Flatten(
        name="flatten_shared",
    )(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        name="shared_dense",
    )(x)

    x = tf.keras.layers.Dropout(
        DROPOUT_RATE,
        name="shared_dropout",
    )(x)

    # --------------------------
    # Volatility regression head
    # --------------------------

    volatility_hidden = tf.keras.layers.Dense(
        32,
        activation="relu",
        name="volatility_hidden",
    )(x)

    volatility = tf.keras.layers.Dense(
        1,
        activation="softplus",
        name="volatility",
    )(volatility_hidden)

    # --------------------------
    # Spike classification head
    # --------------------------

    spike_hidden = tf.keras.layers.Dense(
        32,
        activation="relu",
        name="spike_hidden",
    )(x)

    spike_probability = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="spike_probability",
    )(spike_hidden)

    model = tf.keras.Model(
        inputs=inputs,
        outputs={
            "volatility": volatility,
            "spike_probability": spike_probability,
        },
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss={
            "volatility": tf.keras.losses.Huber(),
            "spike_probability": (
                tf.keras.losses.BinaryCrossentropy()
            ),
        },
        loss_weights={
            "volatility": VOLATILITY_LOSS_WEIGHT,
            "spike_probability": SPIKE_LOSS_WEIGHT,
        },
        metrics={
            "volatility": [
                tf.keras.metrics.MeanAbsoluteError(
                    name="mae"
                ),
            ],
            "spike_probability": [
                tf.keras.metrics.AUC(
                    name="auc",
                    curve="ROC",
                ),
                tf.keras.metrics.AUC(
                    name="pr_auc",
                    curve="PR",
                ),
                tf.keras.metrics.Precision(
                    name="precision"
                ),
                tf.keras.metrics.Recall(
                    name="recall"
                ),
            ],
        },
    )

    return model


# ============================================================
# 15. THRESHOLD SELECTION
# ============================================================
#
# IMPORTANT:
#
# 0.10927724 = actual-volatility threshold.
#
# The CNN classification threshold is different.
#
# Here we choose the probability threshold using ONLY the
# validation set.
#
# We maximize F1.
# ============================================================

def select_probability_threshold(
    y_true,
    probabilities,
):

    precision_values, recall_values, thresholds = (
        precision_recall_curve(
            y_true,
            probabilities,
        )
    )

    if len(thresholds) == 0:

        return 0.50

    f1_values = (
        2
        * precision_values[:-1]
        * recall_values[:-1]
        / np.maximum(
            precision_values[:-1]
            + recall_values[:-1],
            1e-12,
        )
    )

    best_index = int(
        np.nanargmax(
            f1_values
        )
    )

    return float(
        thresholds[best_index]
    )


# ============================================================
# 16. MODEL EVALUATION
# ============================================================

def evaluate_validation_predictions(
    model_name,
    model,
    X_data,
    y_vol_raw_data,
    y_vol_log_data,
    y_spike_data,
    metadata_data,
):

    predictions = model.predict(
        X_data,
        verbose=0,
    )

    predicted_vol_log = np.asarray(
        predictions["volatility"]
    ).reshape(-1)

    predicted_probability = np.asarray(
        predictions["spike_probability"]
    ).reshape(-1)

    # Inverse log1p transformation.
    predicted_volatility = (
        np.expm1(
            np.maximum(
                predicted_vol_log,
                0,
            )
        )
    )

    predicted_volatility = np.maximum(
        predicted_volatility,
        0,
    )

    # --------------------------------------------------------
    # Threshold selection
    # --------------------------------------------------------

    probability_threshold = (
        select_probability_threshold(
            y_spike_data,
            predicted_probability,
        )
    )

    predicted_spike = (
        predicted_probability
        >= probability_threshold
    ).astype(int)

    # --------------------------------------------------------
    # Continuous metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_vol_raw_data,
        predicted_volatility,
    )

    mse = mean_squared_error(
        y_vol_raw_data,
        predicted_volatility,
    )

    rmse = np.sqrt(
        mse
    )

    # --------------------------------------------------------
    # Classification metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_spike_data,
        predicted_spike,
    )

    precision = precision_score(
        y_spike_data,
        predicted_spike,
        zero_division=0,
    )

    recall = recall_score(
        y_spike_data,
        predicted_spike,
        zero_division=0,
    )

    f1 = f1_score(
        y_spike_data,
        predicted_spike,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_spike_data,
        predicted_probability,
    )

    pr_auc = average_precision_score(
        y_spike_data,
        predicted_probability,
    )

    brier = brier_score_loss(
        y_spike_data,
        predicted_probability,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_spike_data,
        predicted_spike,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn
        / max(
            tn + fp,
            1,
        )
    )

    # --------------------------------------------------------
    # Forecast association diagnostics
    # --------------------------------------------------------

    actual_spike_mean = (
        np.mean(
            y_vol_raw_data[
                y_spike_data == 1
            ]
        )
        if np.any(
            y_spike_data == 1
        )
        else np.nan
    )

    predicted_spike_mean = (
        np.mean(
            predicted_volatility[
                y_spike_data == 1
            ]
        )
        if np.any(
            y_spike_data == 1
        )
        else np.nan
    )

    print("\n" + "=" * 90)
    print(
        f"{model_name}: VALIDATION RESULTS"
    )
    print("=" * 90)

    print(
        f"\nValidation observations: "
        f"{len(y_spike_data):,}"
    )

    print(
        f"Actual spikes: "
        f"{int(y_spike_data.sum()):,}"
    )

    print(
        f"Predicted spikes: "
        f"{int(predicted_spike.sum()):,}"
    )

    print(
        f"Selected probability threshold: "
        f"{probability_threshold:.6f}"
    )

    print("\nCONTINUOUS VOLATILITY")

    print(
        f"MAE:  {mae:.6f}"
    )

    print(
        f"MSE:  {mse:.6f}"
    )

    print(
        f"RMSE: {rmse:.6f}"
    )

    print("\nSPIKE CLASSIFICATION")

    print(
        f"Accuracy:    {accuracy:.6f}"
    )

    print(
        f"Precision:   {precision:.6f}"
    )

    print(
        f"Recall:      {recall:.6f}"
    )

    print(
        f"F1:          {f1:.6f}"
    )

    print(
        f"Specificity: {specificity:.6f}"
    )

    print(
        f"ROC-AUC:     {roc_auc:.6f}"
    )

    print(
        f"PR-AUC:      {pr_auc:.6f}"
    )

    print(
        f"Brier score: {brier:.6f}"
    )

    print("\nCONFUSION MATRIX")

    print(
        f"TN={tn:,}  "
        f"FP={fp:,}  "
        f"FN={fn:,}  "
        f"TP={tp:,}"
    )

    print("\nSPIKE-VOLATILITY CHECK")

    print(
        f"Mean actual volatility on spikes: "
        f"{actual_spike_mean:.6f}"
    )

    print(
        f"Mean predicted volatility on spikes: "
        f"{predicted_spike_mean:.6f}"
    )

    # --------------------------------------------------------
    # Save forecast table
    # --------------------------------------------------------

    prediction_output = (
        metadata_data.copy()
    )

    prediction_output[
        "actual_volatility"
    ] = y_vol_raw_data

    prediction_output[
        "predicted_volatility"
    ] = predicted_volatility

    prediction_output[
        "actual_spike"
    ] = y_spike_data.astype(int)

    prediction_output[
        "predicted_probability"
    ] = predicted_probability

    prediction_output[
        "predicted_spike"
    ] = predicted_spike

    prediction_output[
        "probability_threshold"
    ] = probability_threshold

    prediction_file = (
        os.path.join(
            RESULTS_DIR,
            f"{model_name}_validation_predictions.csv",
        )
    )

    prediction_output.to_csv(
        prediction_file,
        index=False,
    )

    # --------------------------------------------------------
    # Return metrics
    # --------------------------------------------------------

    metrics = {
        "model": model_name,
        "validation_observations": len(
            y_spike_data
        ),
        "actual_spikes": int(
            y_spike_data.sum()
        ),
        "predicted_spikes": int(
            predicted_spike.sum()
        ),
        "probability_threshold": (
            probability_threshold
        ),
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Specificity": specificity,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Brier": brier,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "mean_predicted_volatility_on_spikes": (
            predicted_spike_mean
        ),
        "mean_actual_volatility_on_spikes": (
            actual_spike_mean
        ),
        "prediction_file": prediction_file,
    }

    return metrics


# ============================================================
# 17. TRAIN + VALIDATE BOTH CNN SPECIFICATIONS
# ============================================================

all_results = []

trained_models = {}

for model_name, kernel in (
    FEATURE_KERNELS.items()
):

    print("\n" + "#" * 90)
    print(
        f"TRAINING {model_name}"
    )
    print(
        f"Kernel: {kernel}"
    )
    print("#" * 90)

    # Same random seed before each model makes the comparison
    # more reproducible.
    tf.keras.utils.set_random_seed(
        RANDOM_SEED
    )

    model = build_multitask_cnn(
        input_shape=(
            len(FEATURES),
            WINDOW_SIZE,
            1,
        ),
        temporal_feature_kernel=kernel,
    )

    model.summary()

    model_file = os.path.join(
        MODELS_DIR,
        f"{model_name}.keras",
    )

    checkpoint = (
        tf.keras.callbacks.ModelCheckpoint(
            model_file,
            monitor=(
                "val_spike_probability_auc"
            ),
            mode="max",
            save_best_only=True,
            verbose=1,
        )
    )

    early_stopping = (
        tf.keras.callbacks.EarlyStopping(
            monitor=(
                "val_spike_probability_auc"
            ),
            mode="max",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        )
    )

    reduce_lr = (
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=(
                "val_spike_probability_auc"
            ),
            mode="max",
            factor=0.5,
            patience=7,
            min_lr=1e-5,
            verbose=1,
        )
    )

    history = model.fit(
        X_train,
        {
            "volatility": y_train_vol_log,
            "spike_probability": (
                y_train_spike
            ),
        },
        sample_weight={
            "volatility": np.ones(
                len(y_train_vol_log),
                dtype=np.float32,
            ),
            "spike_probability": (
                train_sample_weights
            ),
        },
        validation_data=(
            X_validation,
            {
                "volatility": (
                    y_validation_vol_log
                ),
                "spike_probability": (
                    y_validation_spike
                ),
            },
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=[
            checkpoint,
            early_stopping,
            reduce_lr,
        ],
        verbose=1,
    )

    # Restore best checkpoint.
    model = tf.keras.models.load_model(
        model_file
    )

    trained_models[
        model_name
    ] = model

    # Save training history.
    history_df = pd.DataFrame(
        history.history
    )

    history_file = os.path.join(
        RESULTS_DIR,
        f"{model_name}_training_history.csv",
    )

    history_df.to_csv(
        history_file,
        index=False,
    )

    metrics = (
        evaluate_validation_predictions(
            model_name=model_name,
            model=model,
            X_data=X_validation,
            y_vol_raw_data=(
                y_validation_vol_raw
            ),
            y_vol_log_data=(
                y_validation_vol_log
            ),
            y_spike_data=(
                y_validation_spike
            ),
            metadata_data=(
                validation_metadata
            ),
        )
    )

    metrics[
        "kernel"
    ] = str(kernel)

    metrics[
        "model_file"
    ] = model_file

    all_results.append(
        metrics
    )


# ============================================================
# 18. MODEL COMPARISON
# ============================================================

comparison_df = pd.DataFrame(
    all_results
)

comparison_file = os.path.join(
    RESULTS_DIR,
    "cnn_v5_architecture_comparison.csv",
)

comparison_df.to_csv(
    comparison_file,
    index=False,
)


# ============================================================
# 19. PRE-SPECIFIED ARCHITECTURE SELECTION RULE
# ============================================================
#
# Primary criterion:
#       highest PR-AUC
#
# Tie-break:
#       highest ROC-AUC
#
# Final tie-break:
#       highest F1
#
# This selection is performed ONLY on 2014-2020.
# ============================================================

selection_df = (
    comparison_df
    .sort_values(
        by=[
            "PR_AUC",
            "ROC_AUC",
            "F1",
        ],
        ascending=False,
    )
    .reset_index(drop=True)
)

selected_model_name = (
    selection_df.loc[
        0,
        "model",
    ]
)

selected_row = (
    selection_df.iloc[0]
)

selected_threshold = float(
    selected_row[
        "probability_threshold"
    ]
)

selected_kernel = (
    selected_row[
        "kernel"
    ]
)


print("\n" + "=" * 90)
print("CNN ARCHITECTURE SELECTION")
print("=" * 90)

print(
    "\nSelection rule:"
)

print(
    "1. Highest validation PR-AUC"
)

print(
    "2. ROC-AUC as tie-break"
)

print(
    "3. F1 as final tie-break"
)

print(
    f"\nSelected architecture: "
    f"{selected_model_name}"
)

print(
    f"Selected kernel: "
    f"{selected_kernel}"
)

print(
    f"Frozen probability threshold: "
    f"{selected_threshold:.6f}"
)


# ============================================================
# 20. SAVE SELECTION CONFIGURATION
# ============================================================

selection_config = {
    "selected_model": selected_model_name,
    "selected_kernel": selected_kernel,
    "selected_probability_threshold": (
        selected_threshold
    ),
    "spike_threshold": SPIKE_THRESHOLD,
    "window_size": WINDOW_SIZE,
    "features": FEATURES,
    "training_period": (
        f"{metadata.loc[train_mask, 'target_date'].min().date()}"
        f" to "
        f"{metadata.loc[train_mask, 'target_date'].max().date()}"
    ),
    "validation_period": (
        f"{metadata.loc[validation_mask, 'target_date'].min().date()}"
        f" to "
        f"{metadata.loc[validation_mask, 'target_date'].max().date()}"
    ),
    "scaler_fit_period": (
        f"through {TRAIN_END.date()}"
    ),
    "selection_rule": [
        "PR_AUC",
        "ROC_AUC",
        "F1",
    ],
}


selection_config_file = os.path.join(
    RESULTS_DIR,
    "cnn_v5_selected_configuration.json",
)

with open(
    selection_config_file,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        selection_config,
        file,
        indent=4,
    )


# ============================================================
# 21. FINAL SAFETY CHECK
# ============================================================
#
# The script must prove that no validation or test dates entered
# scaler fitting.
# ============================================================

assert (
    train_feature_rows.shape[0]
    ==
    len(
        df[
            df["date"]
            <= TRAIN_END
        ]
    )
)

assert (
    validation_metadata["target_date"]
    > TRAIN_END
).all()

assert (
    validation_metadata["target_date"]
    <= VALIDATION_END
).all()


print("\n" + "=" * 90)
print("STEP 10C-V5 FINISHED")
print("=" * 90)

print(
    "\nValidation comparison saved to:"
)

print(
    comparison_file
)

print(
    "\nSelected configuration saved to:"
)

print(
    selection_config_file
)

print(
    "\nIMPORTANT:"
)

print(
    "2021-2025 final test set was NOT used."
)

print(
    "\nNext stage:"
)

print(
    "Refit the selected CNN on ALL 2006-2020 training data,"
)

print(
    "freeze the selected architecture/settings,"
)

print(
    "then evaluate exactly once on the untouched 2021-2025 test set."
)