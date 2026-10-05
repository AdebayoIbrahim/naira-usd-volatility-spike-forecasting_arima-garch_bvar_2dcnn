#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 11:34:12 2026

@author: setup
"""

# ============================================================
# STEP 10E
# FINAL CNN-A REFIT + UNTOUCHED 2021-2025 TEST EVALUATION
# ============================================================
#
# PURPOSE
# -------
# Refit the CNN-A architecture selected during V5 validation
# using ALL pre-test observations (2006-2020), then evaluate
# exactly once on the untouched 2021-2025 test period.
#
# SELECTED FROM V5
# ----------------
# Architecture:
#     CNN-A
#
# Kernel:
#     (2, 3)
#
# Probability threshold:
#     0.328393
#
# Best validation epoch:
#     9
#
# TARGETS
# -------
# Raw volatility:
#     r_(t+1)^2
#
# CNN volatility target:
#     log1p(r_(t+1)^2)
#
# Actual spike definition:
#     r_(t+1)^2 > 0.10927724
#
# CNN spike head:
#     sigmoid output in [0, 1]
#
# IMPORTANT
# ---------
# Raw volatility is retained separately from log1p volatility.
# log1p is used for MODEL TRAINING only.
#
# Test data:
#     2021-2025
#
# The test set is NOT used for:
#     - scaling
#     - architecture selection
#     - epoch selection
#     - probability threshold selection
#     - training
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import json
import random
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)


warnings.filterwarnings("ignore")


# ============================================================
# 2. SETTINGS
# ============================================================

INPUT_FILE = (
    "data_cleaned/"
    "forecasting_dataset_2006_2025_spike_gap_target_safe.csv"
)

RESULTS_DIR = "results"
MODELS_DIR = "models"


# ------------------------------------------------------------
# Date configuration
# ------------------------------------------------------------

TRAIN_END = pd.Timestamp(
    "2020-12-31"
)

TEST_START = pd.Timestamp(
    "2021-01-01"
)


# ------------------------------------------------------------
# Window
# ------------------------------------------------------------

WINDOW_SIZE = 20


# ------------------------------------------------------------
# Features
# ------------------------------------------------------------

FEATURES = [
    "fx_log_return",
    "fx_return_squared",
    "brent_log_return",
    "reserves_log_change",
]


# ------------------------------------------------------------
# Targets
# ------------------------------------------------------------

TARGET_VOLATILITY = (
    "target_volatility"
)

TARGET_SPIKE = (
    "target_spike"
)


# ------------------------------------------------------------
# Frozen spike definition
# ------------------------------------------------------------

SPIKE_THRESHOLD = 0.10927724


# Equivalent threshold on log1p scale.
#
# This is mathematically equivalent to the raw threshold because
# log1p is monotonic.

LOG_SPIKE_THRESHOLD = np.log1p(
    SPIKE_THRESHOLD
)


# ------------------------------------------------------------
# Frozen CNN classification cutoff
#
# Selected entirely from 2014-2020 validation in V5.
# ------------------------------------------------------------

PROBABILITY_CUTOFF = 0.328393


# ------------------------------------------------------------
# Selected architecture
# ------------------------------------------------------------

KERNEL_SIZE = (
    2,
    3
)


# ------------------------------------------------------------
# Training settings inherited from V5
# ------------------------------------------------------------

RANDOM_SEED = 2026

BATCH_SIZE = 32

LEARNING_RATE = 0.001

DROPOUT_RATE = 0.30

VOLATILITY_LOSS_WEIGHT = 0.30

SPIKE_LOSS_WEIGHT = 0.70

NORMAL_WEIGHT = 1.0

SPIKE_WEIGHT = 5.0


# ------------------------------------------------------------
# FINAL EPOCH COUNT
#
# V5 CNN-A reached its best validation epoch at epoch 9.
#
# Because the final model has no validation set anymore, we
# freeze this training duration rather than looking at the
# 2021-2025 test period.
# ------------------------------------------------------------

FINAL_EPOCHS = 20


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(
    RANDOM_SEED
)

random.seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)

tf.keras.utils.set_random_seed(
    RANDOM_SEED
)


# ============================================================
# 4. CREATE DIRECTORIES
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)


# ============================================================
# 5. HEADER
# ============================================================

print("\n" + "=" * 90)

print(
    "STEP 10E — FINAL CNN-A REFIT"
)

print("=" * 90)


# ============================================================
# 6. LOAD DATA
# ============================================================

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


print("\n" + "-" * 90)

print(
    "INPUT DATA"
)

print("-" * 90)

print(
    f"Rows: {len(df):,}"
)

print(
    f"Date range: "
    f"{df['date'].min().date()} "
    f"to "
    f"{df['date'].max().date()}"
)


# ============================================================
# 7. INPUT VALIDATION
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


# Duplicate dates are not allowed.
if df["date"].duplicated().any():
    raise ValueError(
        "Duplicate dates detected."
    )


# Predictor variables must be complete.
if df[FEATURES].isna().any().any():
    raise ValueError(
        "Missing values found in predictor variables."
    )


# Target volatility must not be negative.
invalid_volatility = df.loc[
    df[TARGET_VOLATILITY].notna()
    &
    (
        df[TARGET_VOLATILITY]
        < 0
    )
]

if not invalid_volatility.empty:
    raise ValueError(
        "Negative target volatility detected."
    )


print(
    "\nInput validation passed."
)


# ============================================================
# 8. BUILD GAP-SAFE WINDOWS
# ============================================================
#
# Each sample contains:
#
#       t-19, t-18, ..., t-1, t
#
# and predicts:
#
#       t+1
#
# The crucial point is that windows are built WITHIN
# segment_id.
#
# Therefore a long gap such as:
#
#       2009-01-26 → 2009-07-02
#
# cannot be treated as if those two observations were
# consecutive trading observations.
# ============================================================


def build_gap_safe_windows(
    data,
    features,
    window_size,
):

    X_rows = []

    raw_vol_targets = []

    spike_targets = []

    origin_dates = []

    target_dates = []

    segment_ids = []

    skipped_missing_targets = 0

    skipped_short_segments = 0


    # --------------------------------------------------------
    # Process each continuous segment independently
    # --------------------------------------------------------

    for segment_id, segment in data.groupby(
        "segment_id",
        sort=True,
    ):

        segment = (
            segment
            .sort_values("date")
            .reset_index(drop=True)
        )


        # Need at least:
        #
        #   20 observations for input
        #   + 1 observation for target
        #

        if len(segment) <= window_size:

            skipped_short_segments += 1

            continue


        values = segment[
            features
        ].to_numpy(
            dtype=np.float64
        )


        # ----------------------------------------------------
        # Generate windows
        # ----------------------------------------------------

        for position in range(
            window_size - 1,
            len(segment) - 1,
        ):

            origin_row = (
                segment.iloc[position]
            )

            next_row = (
                segment.iloc[position + 1]
            )


            # Target information is attached to the origin row.
            #
            # target_volatility:
            #     next-day squared return
            #
            # target_spike:
            #     binary spike label
            #
            target_vol = (
                origin_row[
                    TARGET_VOLATILITY
                ]
            )

            target_spike = (
                origin_row[
                    TARGET_SPIKE
                ]
            )

            target_date = (
                origin_row[
                    "target_date"
                ]
            )


            # ------------------------------------------------
            # Skip invalid target rows
            # ------------------------------------------------

            if (
                pd.isna(target_vol)
                or
                pd.isna(target_spike)
                or
                pd.isna(target_date)
            ):

                skipped_missing_targets += 1

                continue


            # ------------------------------------------------
            # Verify the target really is t+1
            # ------------------------------------------------

            if (
                target_date
                !=
                next_row["date"]
            ):

                raise ValueError(
                    "Target-date mismatch.\n"
                    f"Origin: {origin_row['date']}\n"
                    f"Expected: {next_row['date']}\n"
                    f"Found: {target_date}"
                )


            # ------------------------------------------------
            # Extract 20 consecutive observations
            # ------------------------------------------------

            window = values[
                position - window_size + 1:
                position + 1
            ]


            expected_shape = (
                window_size,
                len(features)
            )


            if window.shape != expected_shape:

                raise ValueError(
                    "Unexpected window shape: "
                    f"{window.shape}, "
                    f"expected {expected_shape}"
                )


            # Current layout:
            #
            #       20 time points × 4 variables
            #
            # CNN requires:
            #
            #       4 variables × 20 time points
            #

            window = window.T


            # ------------------------------------------------
            # Store
            # ------------------------------------------------

            X_rows.append(
                window
            )

            raw_vol_targets.append(
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


    # ========================================================
    # Convert to NumPy arrays
    # ========================================================

    X = np.asarray(
        X_rows,
        dtype=np.float32
    )

    y_vol_raw = np.asarray(
        raw_vol_targets,
        dtype=np.float32
    )

    y_spike = np.asarray(
        spike_targets,
        dtype=np.float32
    )


    metadata = pd.DataFrame(
        {
            "origin_date":
                origin_dates,

            "target_date":
                target_dates,

            "segment_id":
                segment_ids,
        }
    )


    return (
        X,
        y_vol_raw,
        y_spike,
        metadata,
        skipped_missing_targets,
        skipped_short_segments,
    )


(
    X_raw,
    y_vol_raw,
    y_spike,
    metadata,
    skipped_missing_targets,
    skipped_short_segments,
) = build_gap_safe_windows(
    df,
    FEATURES,
    WINDOW_SIZE,
)


print("\n" + "-" * 90)

print(
    "GAP-SAFE WINDOW CONSTRUCTION"
)

print("-" * 90)

print(
    f"Valid windows: "
    f"{len(X_raw):,}"
)

print(
    f"Skipped missing-target origins: "
    f"{skipped_missing_targets:,}"
)

print(
    f"Skipped short segments: "
    f"{skipped_short_segments:,}"
)

print(
    f"Window shape: "
    f"{X_raw.shape}"
)


# ============================================================
# 9. VERIFY RAW SPIKE LABELS
# ============================================================
#
# The dataset already contains target_spike.
#
# We independently reconstruct it from raw volatility to make
# sure the frozen spike definition is still correct.
# ============================================================

derived_spike = (
    y_vol_raw
    >
    SPIKE_THRESHOLD
).astype(
    np.float32
)


if not np.array_equal(
    derived_spike,
    y_spike
):

    mismatch_count = int(
        np.sum(
            derived_spike
            !=
            y_spike
        )
    )

    raise ValueError(
        "Spike-label inconsistency detected. "
        f"Mismatches: {mismatch_count}"
    )


print(
    "\nSpike-label consistency check passed."
)


# ============================================================
# 10. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================
#
# NOTE:
#
# Split using TARGET DATE, not origin date.
#
# This ensures:
#
# training targets <= 2020-12-31
#
# test targets >= 2021-01-01
# ============================================================

train_mask = (
    metadata[
        "target_date"
    ]
    <= TRAIN_END
)

test_mask = (
    metadata[
        "target_date"
    ]
    >= TEST_START
)


train_mask_np = (
    train_mask
    .to_numpy()
)

test_mask_np = (
    test_mask
    .to_numpy()
)


X_train_raw = X_raw[
    train_mask_np
]

X_test_raw = X_raw[
    test_mask_np
]


# ------------------------------------------------------------
# Keep RAW volatility target separately
# ------------------------------------------------------------

y_train_vol_raw = y_vol_raw[
    train_mask_np
]

y_test_vol_raw = y_vol_raw[
    test_mask_np
]


# ------------------------------------------------------------
# Binary spike labels
# ------------------------------------------------------------

y_train_spike = y_spike[
    train_mask_np
]

y_test_spike = y_spike[
    test_mask_np
]


train_metadata = metadata.loc[
    train_mask
].reset_index(
    drop=True
)

test_metadata = metadata.loc[
    test_mask
].reset_index(
    drop=True
)


print("\n" + "-" * 90)

print(
    "FINAL TRAIN / TEST SPLIT"
)

print("-" * 90)

print(
    f"Training windows: "
    f"{len(X_train_raw):,}"
)

print(
    f"Test windows: "
    f"{len(X_test_raw):,}"
)


print(
    "\nTraining target period: "
    f"{train_metadata['target_date'].min().date()} "
    f"→ "
    f"{train_metadata['target_date'].max().date()}"
)

print(
    "Test target period: "
    f"{test_metadata['target_date'].min().date()} "
    f"→ "
    f"{test_metadata['target_date'].max().date()}"
)


print(
    "\nTraining spikes: "
    f"{int(y_train_spike.sum()):,} "
    f"({100 * y_train_spike.mean():.2f}%)"
)

print(
    "Test spikes: "
    f"{int(y_test_spike.sum()):,} "
    f"({100 * y_test_spike.mean():.2f}%)"
)


# ============================================================
# 11. TRAIN / TEST OVERLAP CHECK
# ============================================================

if (
    set(
        train_metadata[
            "target_date"
        ]
    )
    &
    set(
        test_metadata[
            "target_date"
        ]
    )
):

    raise ValueError(
        "Training/test target-date overlap detected."
    )


if not (
    test_metadata[
        "target_date"
    ]
    > TRAIN_END
).all():

    raise ValueError(
        "A test target occurs on or before the training cutoff."
    )


print(
    "\nChronological separation check passed."
)


# ============================================================
# 12. FIT FEATURE SCALER
# ============================================================
#
# CRITICAL:
#
# The scaler is fitted using ONLY observations dated
# 2006-2020.
#
# No 2021-2025 feature observation contributes to:
#
#       mean
#       standard deviation
#
# This avoids test leakage.
# ============================================================

print("\n" + "-" * 90)

print(
    "FINAL FEATURE SCALING"
)

print("-" * 90)


train_rows_for_scaler = df.loc[
    df["date"] <= TRAIN_END,
    FEATURES,
].to_numpy(
    dtype=np.float64
)


if np.isnan(
    train_rows_for_scaler
).any():

    raise ValueError(
        "NaN values found in scaler-training observations."
    )


scaler = StandardScaler()

scaler.fit(
    train_rows_for_scaler
)


print(
    "Scaler fitted through: "
    f"{TRAIN_END.date()}"
)


print(
    "\nTraining scaler means:"
)


for feature, value in zip(
    FEATURES,
    scaler.mean_,
):

    print(
        f"{feature:<25} "
        f"{value: .8f}"
    )


print(
    "\nTraining scaler standard deviations:"
)


for feature, value in zip(
    FEATURES,
    scaler.scale_,
):

    print(
        f"{feature:<25} "
        f"{value: .8f}"
    )


# ============================================================
# 13. SCALE WINDOWS
# ============================================================

def scale_windows(
    X,
    scaler,
):

    """
    Input:
        X = (n_samples, variables, time)

    StandardScaler expects:
        (observations, variables)

    Therefore:
        1. transpose
        2. reshape
        3. scale
        4. restore original layout
    """

    n_samples, n_variables, n_time = (
        X.shape
    )


    reshaped = (
        X
        .transpose(0, 2, 1)
        .reshape(
            -1,
            n_variables,
        )
    )


    scaled = scaler.transform(
        reshaped
    )


    scaled = (
        scaled
        .reshape(
            n_samples,
            n_time,
            n_variables,
        )
        .transpose(
            0,
            2,
            1,
        )
    )


    return scaled.astype(
        np.float32
    )


X_train_scaled = scale_windows(
    X_train_raw,
    scaler,
)

X_test_scaled = scale_windows(
    X_test_raw,
    scaler,
)


# ============================================================
# 14. ADD CHANNEL DIMENSION
# ============================================================

X_train = (
    X_train_scaled[
        ..., np.newaxis
    ]
)

X_test = (
    X_test_scaled[
        ..., np.newaxis
    ]
)


print("\nCNN input shapes:")

print(
    f"Training: {X_train.shape}"
)

print(
    f"Testing : {X_test.shape}"
)


# ============================================================
# 15. LOG1P VOLATILITY TARGET
# ============================================================
#
# IMPORTANT:
#
# We KEEP y_train_vol_raw as r².
#
# Only the target given to the CNN volatility head is transformed:
#
#       log(1 + r²)
#
# This means we can later compare predictions against the actual
# raw volatility correctly.
# ============================================================

y_train_vol_log = np.log1p(
    np.maximum(
        y_train_vol_raw,
        0,
    )
)


# ============================================================
# 16. SPIKE-AWARE SAMPLE WEIGHTS
# ============================================================

train_sample_weights = np.where(
    y_train_spike == 1,
    SPIKE_WEIGHT,
    NORMAL_WEIGHT,
).astype(
    np.float32
)


print("\n" + "-" * 90)

print(
    "SPIKE-AWARE TRAINING"
)

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
# 17. BUILD CNN-A
# ============================================================
#
# EXACT SELECTED ARCHITECTURE:
#
# Input:
#       4 × 20 × 1
#
# Conv2D:
#       32 filters, 2×3
#
# MaxPool:
#       2×2
#
# Conv2D:
#       64 filters, 2×3
#
# Flatten
#
# Dense 64
#
# Dropout 0.30
#
# Then two heads:
#
# Volatility:
#       Dense 32 → Softplus
#
# Spike:
#       Dense 32 → Sigmoid
#
# V5 used padding="same" on BOTH convolution layers.
# ============================================================

tf.keras.backend.clear_session()

inputs = tf.keras.Input(
    shape=(
        len(FEATURES),
        WINDOW_SIZE,
        1,
    ),
    name="cnn_input",
)


# ------------------------------------------------------------
# Convolution block 1
# ------------------------------------------------------------

x = tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=KERNEL_SIZE,
    padding="same",
    activation="relu",
    name="conv2d_block1",
)(inputs)


# ------------------------------------------------------------
# Pooling
# ------------------------------------------------------------

x = tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2),
    name="maxpool_block1",
)(x)


# ------------------------------------------------------------
# Convolution block 2
# ------------------------------------------------------------

x = tf.keras.layers.Conv2D(
    filters=64,
    kernel_size=KERNEL_SIZE,
    padding="same",
    activation="relu",
    name="conv2d_block2",
)(x)


# ------------------------------------------------------------
# Shared representation
# ------------------------------------------------------------

x = tf.keras.layers.Flatten(
    name="flatten_shared"
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


# ============================================================
# 18. VOLATILITY HEAD
# ============================================================

volatility_hidden = (
    tf.keras.layers.Dense(
        32,
        activation="relu",
        name="volatility_hidden",
    )(x)
)


volatility_output = (
    tf.keras.layers.Dense(
        1,
        activation="softplus",
        name="volatility",
    )(volatility_hidden)
)


# ============================================================
# 19. SPIKE CLASSIFICATION HEAD
# ============================================================

spike_hidden = (
    tf.keras.layers.Dense(
        32,
        activation="relu",
        name="spike_hidden",
    )(x)
)


spike_probability = (
    tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="spike_probability",
    )(spike_hidden)
)


# ============================================================
# 20. FINAL MODEL
# ============================================================

model = tf.keras.Model(
    inputs=inputs,
    outputs={
        "volatility":
            volatility_output,

        "spike_probability":
            spike_probability,
    },
    name="CNN_A_Final",
)


# ============================================================
# 21. COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss={

        "volatility":
            tf.keras.losses.Huber(),

        "spike_probability":
            tf.keras.losses.BinaryCrossentropy(),
    },

    loss_weights={

        "volatility":
            VOLATILITY_LOSS_WEIGHT,

        "spike_probability":
            SPIKE_LOSS_WEIGHT,
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


# ============================================================
# 22. MODEL SUMMARY
# ============================================================

print("\n" + "=" * 90)

print(
    "FINAL CNN-A ARCHITECTURE"
)

print("=" * 90)

model.summary()


# ============================================================
# 23. FINAL REFIT
# ============================================================
#
# We do NOT create another validation set here.
#
# Why?
#
# CNN-A was already selected in V5 using 2014-2020.
#
# The probability threshold was already selected:
#
#       0.328393
#
# The best epoch from V5 was:
#
#       epoch 9
#
# Therefore those settings are frozen and the model now uses
# ALL 2006-2020 training windows.
#
# The 2021-2025 period is untouched.
# ============================================================

print("\n" + "=" * 90)

print(
    "TRAINING FINAL CNN-A"
)

print("=" * 90)

print(
    f"\nEpochs: {FINAL_EPOCHS}"
)

print(
    "Training period: "
    "2006-2020"
)


history = model.fit(

    X_train,

    {
        "volatility":
            y_train_vol_log,

        "spike_probability":
            y_train_spike,
    },

    sample_weight={

        # Regression head:
        # every observation receives normal weight.
        "volatility":
            np.ones(
                len(y_train_vol_log),
                dtype=np.float32,
            ),

        # Classification head:
        # spikes receive higher weight.
        "spike_probability":
            train_sample_weights,
    },

    epochs=FINAL_EPOCHS,

    batch_size=BATCH_SIZE,

    shuffle=False,

    verbose=1,
)


# ============================================================
# 24. SAVE MODEL + SCALER
# ============================================================

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "cnn_A_20epoch_experiment.keras",
)

SCALER_PATH = os.path.join(
    MODELS_DIR,
    "cnn_A_20epoch_experiment_scaler.pkl",
)

PREDICTIONS_PATH = os.path.join(
    RESULTS_DIR,
    "cnn_A_20epoch_experiment_predictions.csv",
)

METRICS_PATH = os.path.join(
    RESULTS_DIR,
    "cnn_A_20epoch_experiment_metrics.csv",
)


model.save(
    MODEL_PATH
)


import joblib

joblib.dump(
    scaler,
    SCALER_PATH
)


print("\nSaved model:")
print(
    MODEL_PATH
)

print(
    "\nSaved scaler:"
)

print(
    SCALER_PATH
)


# ============================================================
# 25. FINAL TEST PREDICTIONS
# ============================================================
#
# THIS IS THE FIRST AND ONLY TIME THE TEST TARGETS are used
# for performance evaluation.
# ============================================================

print("\n" + "=" * 90)

print(
    "UNTUCHED 2021-2025 TEST EVALUATION"
)

print("=" * 90)


predictions = model.predict(
    X_test,
    verbose=0,
)


# Dictionary output:
#
# predictions["volatility"]
# predictions["spike_probability"]

pred_vol_log = np.asarray(
    predictions[
        "volatility"
    ]
).reshape(-1)


pred_spike_probability = np.asarray(
    predictions[
        "spike_probability"
    ]
).reshape(-1)


# ============================================================
# 26. BACK-TRANSFORM VOLATILITY
# ============================================================
#
# CNN predicted:
#
#       z = log(1 + r²)
#
# Recover:
#
#       r² = exp(z) - 1
# ============================================================

pred_volatility = np.expm1(
    pred_vol_log
)


# Numerical safety:
pred_volatility = np.maximum(
    pred_volatility,
    0,
)


# ============================================================
# 27. BINARY SPIKE PREDICTION
# ============================================================
#
# Important distinction:
#
# 0.10927724
#       actual volatility threshold
#
# 0.328393
#       CNN probability cutoff
# ============================================================

pred_spike = (
    pred_spike_probability
    >= PROBABILITY_CUTOFF
).astype(
    int
)


# ============================================================
# 28. CONTINUOUS VOLATILITY METRICS
# ============================================================
#
# Compare:
#
#       ACTUAL raw r²
#
# against:
#
#       PREDICTED raw r²
#
# NOT the log1p values.
# ============================================================

mae = mean_absolute_error(
    y_test_vol_raw,
    pred_volatility,
)

mse = mean_squared_error(
    y_test_vol_raw,
    pred_volatility,
)

rmse = np.sqrt(
    mse
)


# ============================================================
# 29. CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_test_spike,
    pred_spike,
)

precision = precision_score(
    y_test_spike,
    pred_spike,
    zero_division=0,
)

recall = recall_score(
    y_test_spike,
    pred_spike,
    zero_division=0,
)

f1 = f1_score(
    y_test_spike,
    pred_spike,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_test_spike,
    pred_spike_probability,
)

pr_auc = average_precision_score(
    y_test_spike,
    pred_spike_probability,
)

brier = brier_score_loss(
    y_test_spike,
    pred_spike_probability,
)


# ============================================================
# 30. CONFUSION MATRIX
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    y_test_spike,
    pred_spike,
    labels=[0, 1],
).ravel()


specificity = (
    tn
    /
    (tn + fp)
    if (tn + fp) > 0
    else 0
)


balanced_accuracy = (
    recall
    +
    specificity
) / 2


# ============================================================
# 31. BASIC TEST COUNTS
# ============================================================

actual_spike_count = int(
    y_test_spike.sum()
)

predicted_spike_count = int(
    pred_spike.sum()
)


actual_spike_rate = (
    y_test_spike.mean()
)

predicted_spike_rate = (
    pred_spike.mean()
)


# ============================================================
# 32. DIAGNOSTICS
# ============================================================

if actual_spike_count > 0:

    mean_probability_actual_spikes = (
        pred_spike_probability[
            y_test_spike == 1
        ].mean()
    )

    mean_predicted_volatility_spikes = (
        pred_volatility[
            y_test_spike == 1
        ].mean()
    )

else:

    mean_probability_actual_spikes = np.nan

    mean_predicted_volatility_spikes = np.nan


nonspike_mask = (
    y_test_spike == 0
)


if nonspike_mask.sum() > 0:

    mean_probability_nonspikes = (
        pred_spike_probability[
            nonspike_mask
        ].mean()
    )

    mean_predicted_volatility_nonspikes = (
        pred_volatility[
            nonspike_mask
        ].mean()
    )

else:

    mean_probability_nonspikes = np.nan

    mean_predicted_volatility_nonspikes = np.nan


# ============================================================
# 33. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 90)

print(
    "CNN-A FINAL TEST RESULTS"
)

print("=" * 90)


print("\n--- CONTINUOUS VOLATILITY ---")

print(
    f"MAE  : {mae:.6f}"
)

print(
    f"MSE  : {mse:.6f}"
)

print(
    f"RMSE : {rmse:.6f}"
)


print("\n--- SPIKE CLASSIFICATION ---")

print(
    f"Accuracy          : {accuracy:.6f}"
)

print(
    f"Balanced Accuracy : "
    f"{balanced_accuracy:.6f}"
)

print(
    f"Precision         : "
    f"{precision:.6f}"
)

print(
    f"Recall            : "
    f"{recall:.6f}"
)

print(
    f"Specificity       : "
    f"{specificity:.6f}"
)

print(
    f"F1 Score          : "
    f"{f1:.6f}"
)

print(
    f"ROC-AUC           : "
    f"{roc_auc:.6f}"
)

print(
    f"PR-AUC            : "
    f"{pr_auc:.6f}"
)

print(
    f"Brier Score       : "
    f"{brier:.6f}"
)


print("\n--- CONFUSION MATRIX ---")

print(
    f"TN: {tn}"
)

print(
    f"FP: {fp}"
)

print(
    f"FN: {fn}"
)

print(
    f"TP: {tp}"
)


print("\n--- SPIKE COUNTS ---")

print(
    f"Actual spikes: "
    f"{actual_spike_count:,} "
    f"({100 * actual_spike_rate:.2f}%)"
)

print(
    f"Predicted spikes: "
    f"{predicted_spike_count:,} "
    f"({100 * predicted_spike_rate:.2f}%)"
)


print("\n--- FROZEN SETTINGS ---")

print(
    f"Raw spike threshold: "
    f"{SPIKE_THRESHOLD:.8f}"
)

print(
    f"Equivalent log1p threshold: "
    f"{LOG_SPIKE_THRESHOLD:.8f}"
)

print(
    f"CNN probability cutoff: "
    f"{PROBABILITY_CUTOFF:.6f}"
)

print(
    f"Training epochs: "
    f"{FINAL_EPOCHS}"
)


print("\n--- PROBABILITY DIAGNOSTICS ---")

print(
    "Mean probability on actual spikes: "
    f"{mean_probability_actual_spikes:.6f}"
)

print(
    "Mean probability on non-spikes: "
    f"{mean_probability_nonspikes:.6f}"
)


print("\n--- VOLATILITY DIAGNOSTICS ---")

print(
    "Mean predicted volatility on actual spikes: "
    f"{mean_predicted_volatility_spikes:.6f}"
)

print(
    "Mean predicted volatility on non-spikes: "
    f"{mean_predicted_volatility_nonspikes:.6f}"
)


# ============================================================
# 34. TEST PROBABILITY DISTRIBUTION
# ============================================================

print(
    "\n--- TEST PROBABILITY PERCENTILES ---"
)

for percentile in [
    50,
    75,
    90,
    95,
    99,
    100,
]:

    value = np.percentile(
        pred_spike_probability,
        percentile,
    )

    print(
        f"{percentile:>3}th percentile: "
        f"{value:.6f}"
    )


# ============================================================
# 35. SAVE TEST PREDICTIONS
# ============================================================

test_predictions = pd.DataFrame(

    {

        "origin_date":
            test_metadata[
                "origin_date"
            ],

        "target_date":
            test_metadata[
                "target_date"
            ],

        "segment_id":
            test_metadata[
                "segment_id"
            ],

        # ---------------------------------------------
        # ACTUAL
        # ---------------------------------------------

        "actual_volatility":
            y_test_vol_raw,

        "actual_spike":
            y_test_spike.astype(int),

        # ---------------------------------------------
        # CNN VOLATILITY HEAD
        # ---------------------------------------------

        "predicted_log1p_volatility":
            pred_vol_log,

        "predicted_volatility":
            pred_volatility,

        # ---------------------------------------------
        # CNN SPIKE HEAD
        # ---------------------------------------------

        "predicted_spike_probability":
            pred_spike_probability,

        "predicted_spike":
            pred_spike,

        # ---------------------------------------------
        # FROZEN SETTINGS
        # ---------------------------------------------

        "raw_spike_threshold":
            SPIKE_THRESHOLD,

        "log1p_spike_threshold":
            LOG_SPIKE_THRESHOLD,

        "probability_cutoff":
            PROBABILITY_CUTOFF,
    }
)


PREDICTIONS_PATH = os.path.join(
    RESULTS_DIR,
    "cnn_A_final_test_predictions_20ep.csv"
)


test_predictions.to_csv(
    PREDICTIONS_PATH,
    index=False,
)


# ============================================================
# 36. SAVE METRICS
# ============================================================

metrics_row = {

    "model":
        "CNN-A-final",

    "kernel":
        "2x3",

    "window_size":
        WINDOW_SIZE,

    "train_start":
        str(
            train_metadata[
                "target_date"
            ].min().date()
        ),

    "train_end":
        str(
            train_metadata[
                "target_date"
            ].max().date()
        ),

    "test_start":
        str(
            test_metadata[
                "target_date"
            ].min().date()
        ),

    "test_end":
        str(
            test_metadata[
                "target_date"
            ].max().date()
        ),

    "final_epochs":
        FINAL_EPOCHS,

    "spike_threshold_raw":
        SPIKE_THRESHOLD,

    "spike_threshold_log1p":
        LOG_SPIKE_THRESHOLD,

    "probability_cutoff":
        PROBABILITY_CUTOFF,

    # Continuous
    "MAE":
        mae,

    "MSE":
        mse,

    "RMSE":
        rmse,

    # Classification
    "Accuracy":
        accuracy,

    "Balanced_Accuracy":
        balanced_accuracy,

    "Precision":
        precision,

    "Recall":
        recall,

    "Specificity":
        specificity,

    "F1":
        f1,

    "ROC_AUC":
        roc_auc,

    "PR_AUC":
        pr_auc,

    "Brier_Score":
        brier,

    # Confusion matrix
    "TN":
        tn,

    "FP":
        fp,

    "FN":
        fn,

    "TP":
        tp,

    # Counts
    "actual_spikes":
        actual_spike_count,

    "predicted_spikes":
        predicted_spike_count,

    "actual_spike_rate":
        actual_spike_rate,

    "predicted_spike_rate":
        predicted_spike_rate,
}


metrics_df = pd.DataFrame(
    [metrics_row]
)


METRICS_PATH = os.path.join(
    RESULTS_DIR,
    "cnn_A_final_test_metrics_20ep.csv"
)


metrics_df.to_csv(
    METRICS_PATH,
    index=False,
)


# ============================================================
# 37. SAVE SCALER PARAMETERS
# ============================================================

scaler_df = pd.DataFrame(

    {

        "feature":
            FEATURES,

        "training_mean":
            scaler.mean_,

        "training_std":
            scaler.scale_,
    }
)


SCALER_INFO_PATH = os.path.join(
    RESULTS_DIR,
    "cnn_A_final_scaler_parameters.csv"
)


scaler_df.to_csv(
    SCALER_INFO_PATH,
    index=False,
)


# ============================================================
# 38. SAVE RUN CONFIGURATION
# ============================================================

config = {

    "model":
        "CNN-A",

    "kernel":
        "(2, 3)",

    "window_size":
        WINDOW_SIZE,

    "features":
        FEATURES,

    "train_end":
        str(
            TRAIN_END.date()
        ),

    "test_start":
        str(
            TEST_START.date()
        ),

    "raw_spike_threshold":
        SPIKE_THRESHOLD,

    "log1p_spike_threshold":
        float(
            LOG_SPIKE_THRESHOLD
        ),

    "probability_cutoff":
        PROBABILITY_CUTOFF,

    "volatility_target":
        "log1p(next-day squared FX return)",

    "spike_target":
        "raw next-day squared FX return > 0.10927724",

    "spike_weight":
        SPIKE_WEIGHT,

    "normal_weight":
        NORMAL_WEIGHT,

    "volatility_loss_weight":
        VOLATILITY_LOSS_WEIGHT,

    "spike_loss_weight":
        SPIKE_LOSS_WEIGHT,

    "learning_rate":
        LEARNING_RATE,

    "batch_size":
        BATCH_SIZE,

    "dropout":
        DROPOUT_RATE,

    "final_epochs":
        FINAL_EPOCHS,

    "seed":
        RANDOM_SEED,

    "architecture_selected_from":
        "2014-2020 validation",

    "probability_cutoff_selected_from":
        "2014-2020 validation",

    "test_used_for_selection":
        False,
}


CONFIG_PATH = os.path.join(
    RESULTS_DIR,
    "cnn_A_final_configuration.json"
)


with open(
    CONFIG_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        config,
        file,
        indent=4,
    )


# ============================================================
# 39. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 90)

print(
    "FILES SAVED"
)

print("=" * 90)

print(
    f"\nModel:\n{MODEL_PATH}"
)

print(
    f"\nScaler:\n{SCALER_PATH}"
)

print(
    f"\nPredictions:\n{PREDICTIONS_PATH}"
)

print(
    f"\nMetrics:\n{METRICS_PATH}"
)

print(
    f"\nScaler parameters:\n{SCALER_INFO_PATH}"
)

print(
    f"\nConfiguration:\n{CONFIG_PATH}"
)


print("\n" + "=" * 90)

print(
    "STEP 10E COMPLETE"
)

print("=" * 90)