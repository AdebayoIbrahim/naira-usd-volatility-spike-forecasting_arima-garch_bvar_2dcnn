#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 15:54:06 2026

@author: setup
"""

# ============================================================
# STEP 10C-V2
# Multivariate 2D-CNN for Volatility Spike Forecasting
#
# Revised validation design:
#   Training:   2006 -> 2013
#   Validation: 2014 -> 2020
#   Test:       2021 -> 2025
#
# The test set remains completely untouched.
#
# Target:
#   log1p(next-period squared FX return)
#
# Training objective:
#   Huber loss + spike-aware sample weighting
#
# ============================================================

import os
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_DIR = "data_cleaned"
RESULTS_DIR = "results"
MODEL_DIR = "models"

TRAIN_X_FILE = os.path.join(
    DATA_DIR,
    "cnn_X_train_scaled.npy"
)

TRAIN_Y_FILE = os.path.join(
    DATA_DIR,
    "cnn_y_train_volatility.npy"
)

TRAIN_META_FILE = os.path.join(
    DATA_DIR,
    "cnn_train_metadata.csv"
)

TEST_X_FILE = os.path.join(
    DATA_DIR,
    "cnn_X_test_scaled.npy"
)

TEST_Y_FILE = os.path.join(
    DATA_DIR,
    "cnn_y_test_volatility.npy"
)

TEST_META_FILE = os.path.join(
    DATA_DIR,
    "cnn_test_metadata.csv"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "cnn_2d_volatility_model_v2.keras"
)

HISTORY_FILE = os.path.join(
    RESULTS_DIR,
    "cnn_2d_training_history_v2.csv"
)

VALIDATION_FILE = os.path.join(
    RESULTS_DIR,
    "cnn_validation_predictions_v2.csv"
)

CONFIG_FILE = os.path.join(
    RESULTS_DIR,
    "cnn_2d_model_config_v2.json"
)


# ============================================================
# 2. PROJECT SETTINGS
# ============================================================

RANDOM_SEED = 42

VALIDATION_START = pd.Timestamp(
    "2014-01-01"
)

VALIDATION_END = pd.Timestamp(
    "2020-12-31"
)

SPIKE_THRESHOLD = 0.10927724

SPIKE_WEIGHT = 5.0

EPOCHS = 100

BATCH_SIZE = 32

LEARNING_RATE = 0.001


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

np.random.seed(
    RANDOM_SEED
)

tf.random.set_seed(
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
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# 5. HEADER
# ============================================================

print("=" * 70)
print("STEP 10C-V2: REVISED MULTIVARIATE 2D-CNN")
print("=" * 70)


# ============================================================
# 6. LOAD TRAINING DATA
# ============================================================

print("\nLoading training predictors...")

X_train_full = np.load(
    TRAIN_X_FILE
)

print(
    f"Training input: {X_train_full.shape}"
)


print("\nLoading training volatility targets...")

y_vol_full = np.load(
    TRAIN_Y_FILE
)

print(
    f"Training target: {y_vol_full.shape}"
)


print("\nLoading training metadata...")

train_meta = pd.read_csv(
    TRAIN_META_FILE
)

train_meta["target_date"] = pd.to_datetime(
    train_meta["target_date"]
)


# ============================================================
# 7. LOAD TEST DATA
# ============================================================
#
# We load it for shape verification only.
#
# It will NOT be used for:
#   - fitting
#   - validation
#   - model selection
#
# ============================================================

print("\nLoading untouched test data...")

X_test = np.load(
    TEST_X_FILE
)

y_test_vol = np.load(
    TEST_Y_FILE
)

test_meta = pd.read_csv(
    TEST_META_FILE
)

test_meta["target_date"] = pd.to_datetime(
    test_meta["target_date"]
)


print(
    f"Test input:  {X_test.shape}"
)

print(
    f"Test target: {y_test_vol.shape}"
)


# ============================================================
# 8. BASIC VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("DATA VALIDATION")
print("-" * 70)


if len(X_train_full) != len(y_vol_full):
    raise ValueError(
        "Training X and y lengths do not match."
    )


if len(X_train_full) != len(train_meta):
    raise ValueError(
        "Training metadata does not match training data."
    )


if len(X_test) != len(y_test_vol):
    raise ValueError(
        "Test X and y lengths do not match."
    )


if len(X_test) != len(test_meta):
    raise ValueError(
        "Test metadata does not match test data."
    )


if X_train_full.shape[1:] != (4, 20, 1):
    raise ValueError(
        "Expected training shape (..., 4, 20, 1)."
    )


if X_test.shape[1:] != (4, 20, 1):
    raise ValueError(
        "Expected test shape (..., 4, 20, 1)."
    )


if np.isnan(X_train_full).any():
    raise ValueError(
        "NaN found in training predictors."
    )


if np.isinf(X_train_full).any():
    raise ValueError(
        "Inf found in training predictors."
    )


if np.isnan(X_test).any():
    raise ValueError(
        "NaN found in test predictors."
    )


if np.isinf(X_test).any():
    raise ValueError(
        "Inf found in test predictors."
    )


if np.isnan(y_vol_full).any():
    raise ValueError(
        "NaN found in training targets."
    )


if np.isnan(y_test_vol).any():
    raise ValueError(
        "NaN found in test targets."
    )


if np.any(y_vol_full < 0):
    raise ValueError(
        "Negative volatility found in training target."
    )


if np.any(y_test_vol < 0):
    raise ValueError(
        "Negative volatility found in test target."
    )


print(
    "PASS: All predictor and target checks passed."
)


# ============================================================
# 9. DATE-BASED VALIDATION SPLIT
# ============================================================
#
# Full model-development period:
#
#   2006 -> 2020
#
# We now split it as:
#
#   TRAIN      2006 -> 2013
#   VALIDATION 2014 -> 2020
#
# The final test remains:
#
#   2021 -> 2025
#
# ============================================================

train_dates_full = train_meta[
    "target_date"
]

train_mask = (
    train_dates_full
    < VALIDATION_START
)

validation_mask = (
    (train_dates_full >= VALIDATION_START)
    &
    (train_dates_full <= VALIDATION_END)
)


# ============================================================
# 10. CHECK THE SPLIT
# ============================================================

X_train = X_train_full[
    train_mask.to_numpy()
]

X_validation = X_train_full[
    validation_mask.to_numpy()
]

y_train_vol = y_vol_full[
    train_mask.to_numpy()
]

y_validation_vol = y_vol_full[
    validation_mask.to_numpy()
]

train_dates = train_dates_full[
    train_mask
].reset_index(drop=True)

validation_dates = train_dates_full[
    validation_mask
].reset_index(drop=True)


print("\n" + "-" * 70)
print("CHRONOLOGICAL TRAIN / VALIDATION SPLIT")
print("-" * 70)

print(
    f"Training observations:   {len(X_train)}"
)

print(
    f"Validation observations: {len(X_validation)}"
)

print(
    f"Training period:   "
    f"{train_dates.min()} → {train_dates.max()}"
)

print(
    f"Validation period: "
    f"{validation_dates.min()} → {validation_dates.max()}"
)


if (
    train_dates.max()
    >= validation_dates.min()
):

    raise ValueError(
        "Training and validation periods overlap."
    )


print(
    "PASS: Training and validation are chronologically separated."
)


# ============================================================
# 11. CHECK VALIDATION SPIKE COUNT
# ============================================================

y_train_spike = (
    y_train_vol > SPIKE_THRESHOLD
).astype(np.int32)

y_validation_spike = (
    y_validation_vol > SPIKE_THRESHOLD
).astype(np.int32)


train_spike_count = int(
    y_train_spike.sum()
)

validation_spike_count = int(
    y_validation_spike.sum()
)


train_spike_rate = (
    train_spike_count
    / len(y_train_spike)
)

validation_spike_rate = (
    validation_spike_count
    / len(y_validation_spike)
)


print("\n" + "-" * 70)
print("SPIKE DISTRIBUTION")
print("-" * 70)

print(
    f"Training spikes: "
    f"{train_spike_count}"
)

print(
    f"Training spike rate: "
    f"{train_spike_rate:.4%}"
)

print(
    f"Validation spikes: "
    f"{validation_spike_count}"
)

print(
    f"Validation spike rate: "
    f"{validation_spike_rate:.4%}"
)

print(
    f"\nSpike threshold: "
    f"{SPIKE_THRESHOLD}"
)


if validation_spike_count < 10:

    print(
        "\nWARNING: Validation still contains fewer "
        "than 10 spikes."
    )

else:

    print(
        "\nPASS: Validation contains enough spike "
        "events for a more meaningful check."
    )


# ============================================================
# 12. LOG1P TARGET TRANSFORMATION
# ============================================================
#
# Original volatility:
#
#       v = r²
#
# Training target:
#
#       z = log(1 + v)
#
# This reduces the effect of extreme volatility values
# during neural-network optimization.
#
# ============================================================

print("\n" + "-" * 70)
print("LOG1P TARGET TRANSFORMATION")
print("-" * 70)


y_train = np.log1p(
    y_train_vol
)

y_validation = np.log1p(
    y_validation_vol
)


print(
    f"Original training mean: "
    f"{np.mean(y_train_vol):.6f}"
)

print(
    f"Original training maximum: "
    f"{np.max(y_train_vol):.6f}"
)

print(
    f"Log1p training mean: "
    f"{np.mean(y_train):.6f}"
)

print(
    f"Log1p training maximum: "
    f"{np.max(y_train):.6f}"
)


# ============================================================
# 13. CREATE SPIKE-AWARE SAMPLE WEIGHTS
# ============================================================
#
# Normal observation:
#
#       weight = 1
#
# Spike observation:
#
#       weight = 5
#
# This means an actual spike contributes five times as much
# to the training loss as a normal observation.
#
# The purpose is to reduce the tendency of the model to learn:
#
#       "Just predict nearly zero because spikes are rare."
#
# ============================================================

print("\n" + "-" * 70)
print("SPIKE-AWARE TRAINING WEIGHTS")
print("-" * 70)


train_sample_weights = np.ones(
    len(y_train_vol),
    dtype=np.float32
)

train_sample_weights[
    y_train_spike == 1
] = SPIKE_WEIGHT


validation_sample_weights = np.ones(
    len(y_validation_vol),
    dtype=np.float32
)

validation_sample_weights[
    y_validation_spike == 1
] = SPIKE_WEIGHT


normal_weight_total = np.sum(
    train_sample_weights[
        y_train_spike == 0
    ]
)

spike_weight_total = np.sum(
    train_sample_weights[
        y_train_spike == 1
    ]
)

total_weight = np.sum(
    train_sample_weights
)


print(
    f"Normal observation weight: "
    f"1.0"
)

print(
    f"Spike observation weight: "
    f"{SPIKE_WEIGHT}"
)

print(
    f"Total normal-observation weight: "
    f"{normal_weight_total:.1f}"
)

print(
    f"Total spike-observation weight: "
    f"{spike_weight_total:.1f}"
)

print(
    f"Share of total training weight assigned "
    f"to spikes: "
    f"{spike_weight_total / total_weight:.4%}"
)


# ============================================================
# 14. BUILD 2D-CNN
# ============================================================

print("\n" + "=" * 70)
print("BUILDING 2D-CNN")
print("=" * 70)


model = Sequential([

    Input(
        shape=(4, 20, 1)
    ),

    # --------------------------------------------------------
    # Convolution block 1
    # --------------------------------------------------------

    Conv2D(
        filters=32,
        kernel_size=(2, 3),
        activation="relu",
        padding="same"
    ),

    MaxPooling2D(
        pool_size=(2, 2)
    ),

    # --------------------------------------------------------
    # Convolution block 2
    # --------------------------------------------------------

    Conv2D(
        filters=64,
        kernel_size=(2, 3),
        activation="relu",
        padding="same"
    ),

    # --------------------------------------------------------
    # Dense prediction block
    # --------------------------------------------------------

    Flatten(),

    Dense(
        units=64,
        activation="relu"
    ),

    Dropout(
        rate=0.30
    ),

    # --------------------------------------------------------
    # Continuous non-negative volatility forecast
    # --------------------------------------------------------

    Dense(
        units=1,
        activation="softplus"
    )
])


# ============================================================
# 15. COMPILE MODEL
# ============================================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)


model.compile(
    optimizer=optimizer,

    loss=tf.keras.losses.Huber(),

    metrics=[
        tf.keras.metrics.MeanAbsoluteError(
            name="mae"
        )
    ]
)


# ============================================================
# 16. DISPLAY ARCHITECTURE
# ============================================================

model.summary()


# ============================================================
# 17. CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=15,
    min_delta=1e-5,
    restore_best_weights=True,
    verbose=1
)


reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=6,
    min_lr=1e-6,
    verbose=1
)


checkpoint = ModelCheckpoint(
    MODEL_FILE,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)


# ============================================================
# 18. TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("STARTING CNN TRAINING")
print("=" * 70)

history = model.fit(

    X_train,
    y_train,

    sample_weight=train_sample_weights,

    validation_data=(
        X_validation,
        y_validation,
        validation_sample_weights
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    # Preserve chronological order
    shuffle=False,

    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ],

    verbose=1
)


# ============================================================
# 19. LOAD BEST MODEL
# ============================================================

print("\nLoading best CNN model...")

best_model = tf.keras.models.load_model(
    MODEL_FILE
)


# ============================================================
# 20. VALIDATION FORECAST
# ============================================================

print("\nGenerating validation forecasts...")

validation_pred_log = (
    best_model.predict(
        X_validation,
        verbose=0
    )
    .reshape(-1)
)


# ============================================================
# 21. RETURN TO ORIGINAL VOLATILITY SCALE
# ============================================================

validation_pred_vol = np.expm1(
    validation_pred_log
)

# Numerical safety
validation_pred_vol = np.maximum(
    validation_pred_vol,
    0.0
)


# ============================================================
# 22. CREATE SPIKE PREDICTIONS
# ============================================================

validation_pred_spike = (
    validation_pred_vol
    > SPIKE_THRESHOLD
).astype(np.int32)


# ============================================================
# 23. VALIDATION SUMMARY
# ============================================================

predicted_spike_count = int(
    validation_pred_spike.sum()
)

actual_spike_count = int(
    y_validation_spike.sum()
)


print("\n" + "-" * 70)
print("VALIDATION FORECAST SUMMARY")
print("-" * 70)

print(
    f"Validation observations: "
    f"{len(y_validation_vol)}"
)

print(
    f"Actual spikes: "
    f"{actual_spike_count}"
)

print(
    f"Actual spike rate: "
    f"{actual_spike_count / len(y_validation_vol):.4%}"
)

print(
    f"Predicted spikes: "
    f"{predicted_spike_count}"
)

print(
    f"Predicted spike rate: "
    f"{predicted_spike_count / len(y_validation_vol):.4%}"
)

print(
    f"Spike threshold: "
    f"{SPIKE_THRESHOLD}"
)


# ============================================================
# 24. VALIDATION PREVIEW
# ============================================================

print("\nFirst 20 validation forecasts:")

validation_preview = pd.DataFrame({

    "date":
        validation_dates.iloc[:20],

    "actual_volatility":
        y_validation_vol[:20],

    "predicted_volatility":
        validation_pred_vol[:20],

    "actual_spike":
        y_validation_spike[:20],

    "predicted_spike":
        validation_pred_spike[:20]

})


print(
    validation_preview.to_string(
        index=False
    )
)


# ============================================================
# 25. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_df.index.name = "epoch"

history_df.reset_index(
    inplace=True
)

history_df["epoch"] = (
    history_df["epoch"] + 1
)


history_df.to_csv(
    HISTORY_FILE,
    index=False
)


# ============================================================
# 26. SAVE VALIDATION PREDICTIONS
# ============================================================

validation_results = pd.DataFrame({

    "target_date":
        validation_dates,

    "actual_volatility":
        y_validation_vol,

    "predicted_volatility":
        validation_pred_vol,

    "actual_spike":
        y_validation_spike,

    "predicted_spike":
        validation_pred_spike

})


validation_results.to_csv(
    VALIDATION_FILE,
    index=False
)


# ============================================================
# 27. SAVE MODEL CONFIGURATION
# ============================================================

config = {

    "model_type":
        "Multivariate 2D-CNN",

    "input_shape":
        [4, 20, 1],

    "features": [
        "fx_log_return",
        "fx_return_squared",
        "brent_log_return",
        "reserves_log_change"
    ],

    "window_size":
        20,

    "target":
        "next_period_fx_return_squared",

    "target_training_transform":
        "log1p",

    "inverse_transform":
        "expm1",

    "spike_threshold":
        SPIKE_THRESHOLD,

    "validation_start":
        str(VALIDATION_START.date()),

    "validation_end":
        str(VALIDATION_END.date()),

    "spike_weight":
        SPIKE_WEIGHT,

    "epochs_max":
        EPOCHS,

    "batch_size":
        BATCH_SIZE,

    "learning_rate":
        LEARNING_RATE,

    "random_seed":
        RANDOM_SEED,

    "loss":
        "Huber",

    "shuffle":
        False,

    "architecture": [

        "Conv2D(32, kernel=(2,3), activation=relu)",

        "MaxPooling2D(pool=(2,2))",

        "Conv2D(64, kernel=(2,3), activation=relu)",

        "Flatten",

        "Dense(64, activation=relu)",

        "Dropout(0.30)",

        "Dense(1, activation=softplus)"

    ]

}


with open(
    CONFIG_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        config,
        file,
        indent=4
    )


# ============================================================
# 28. FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("STEP 10C-V2 FINISHED")
print("=" * 70)

print("\nBest model:")
print(
    MODEL_FILE
)

print("\nTraining history:")
print(
    HISTORY_FILE
)

print("\nValidation forecasts:")
print(
    VALIDATION_FILE
)

print("\nModel configuration:")
print(
    CONFIG_FILE
)

print("\nFinal test set has NOT been used.")

print(
    "\nNext stage: inspect the validation performance, "
    "then proceed to Step 10D test evaluation."
)