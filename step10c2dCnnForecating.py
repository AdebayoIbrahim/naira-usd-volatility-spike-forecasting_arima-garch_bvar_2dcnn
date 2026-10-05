#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 13:26:59 2026

@author: setup
"""

# ============================================================
# STEP 10C
# Multivariate 2D-CNN for Next-Period Volatility Forecasting
#
# Input:
#   4 variables × 20 observations
#
# Target:
#   log1p(next-period squared FX return)
#
# Final output:
#   forecasted next-period volatility on original scale
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

TEST_X_FILE = os.path.join(
    DATA_DIR,
    "cnn_X_test_scaled.npy"
)

TRAIN_Y_FILE = os.path.join(
    DATA_DIR,
    "cnn_y_train_volatility.npy"
)

TEST_Y_FILE = os.path.join(
    DATA_DIR,
    "cnn_y_test_volatility.npy"
)

TRAIN_META_FILE = os.path.join(
    DATA_DIR,
    "cnn_train_metadata.csv"
)

TEST_META_FILE = os.path.join(
    DATA_DIR,
    "cnn_test_metadata.csv"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "cnn_2d_volatility_model.keras"
)

HISTORY_FILE = os.path.join(
    RESULTS_DIR,
    "cnn_2d_training_history.csv"
)

CONFIG_FILE = os.path.join(
    RESULTS_DIR,
    "cnn_2d_model_config.json"
)

VALIDATION_PREDICTIONS_FILE = os.path.join(
    RESULTS_DIR,
    "cnn_validation_predictions.csv"
)


# ============================================================
# 2. MODEL SETTINGS
# ============================================================

RANDOM_SEED = 42

VALIDATION_FRACTION = 0.20

EPOCHS = 100

BATCH_SIZE = 32

LEARNING_RATE = 0.001

SPIKE_THRESHOLD = 0.10927724


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_SEED)

tf.random.set_seed(RANDOM_SEED)


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
print("STEP 10C: MULTIVARIATE 2D-CNN")
print("=" * 70)


# ============================================================
# 6. LOAD DATA
# ============================================================

print("\nLoading scaled CNN predictors...")

X_train_full = np.load(
    TRAIN_X_FILE
)

X_test = np.load(
    TEST_X_FILE
)

print(
    f"Full training input: {X_train_full.shape}"
)

print(
    f"Test input:          {X_test.shape}"
)


print("\nLoading volatility targets...")

y_train_vol_full = np.load(
    TRAIN_Y_FILE
)

y_test_vol = np.load(
    TEST_Y_FILE
)

print(
    f"Full training target: {y_train_vol_full.shape}"
)

print(
    f"Test target:          {y_test_vol.shape}"
)


# ============================================================
# 7. LOAD METADATA
# ============================================================

train_meta = pd.read_csv(
    TRAIN_META_FILE
)

test_meta = pd.read_csv(
    TEST_META_FILE
)

train_meta["target_date"] = pd.to_datetime(
    train_meta["target_date"]
)

test_meta["target_date"] = pd.to_datetime(
    test_meta["target_date"]
)


# ============================================================
# 8. BASIC CONSISTENCY CHECKS
# ============================================================

if len(X_train_full) != len(y_train_vol_full):
    raise ValueError(
        "Training predictors and targets have different lengths."
    )

if len(X_test) != len(y_test_vol):
    raise ValueError(
        "Test predictors and targets have different lengths."
    )

if len(X_train_full) != len(train_meta):
    raise ValueError(
        "Training metadata length does not match training data."
    )

if len(X_test) != len(test_meta):
    raise ValueError(
        "Test metadata length does not match test data."
    )


if X_train_full.ndim != 4:
    raise ValueError(
        "Expected CNN input with 4 dimensions."
    )


if X_train_full.shape[1:] != (4, 20, 1):
    raise ValueError(
        "Expected each CNN input to have shape (4, 20, 1)."
    )


# ============================================================
# 9. VERIFY NO INVALID INPUT VALUES
# ============================================================

print("\nChecking input data...")

if np.isnan(X_train_full).any():
    raise ValueError(
        "NaN values detected in training predictors."
    )

if np.isnan(X_test).any():
    raise ValueError(
        "NaN values detected in test predictors."
    )

if np.isinf(X_train_full).any():
    raise ValueError(
        "Infinite values detected in training predictors."
    )

if np.isinf(X_test).any():
    raise ValueError(
        "Infinite values detected in test predictors."
    )

print("PASS: Predictor data contains no NaN/Inf values.")


# ============================================================
# 10. TARGET VALIDATION
# ============================================================

if np.isnan(y_train_vol_full).any():
    raise ValueError(
        "NaN values detected in training targets."
    )

if np.isnan(y_test_vol).any():
    raise ValueError(
        "NaN values detected in test targets."
    )

if np.any(y_train_vol_full < 0):
    raise ValueError(
        "Negative training volatility detected."
    )

if np.any(y_test_vol < 0):
    raise ValueError(
        "Negative test volatility detected."
    )

print(
    "PASS: Volatility targets are non-negative and valid."
)


# ============================================================
# 11. CHRONOLOGICAL VALIDATION SPLIT
# ============================================================
#
# We DO NOT randomly shuffle observations.
#
# The last 20% of the training period becomes validation.
#
# Example:
#
# Training period
# ├──────────────────────┬───────────────┤
# │ model-training data  │ validation    │
# └──────────────────────┴───────────────┘
#
# Test period remains untouched.
#
# ============================================================

n_train_full = len(X_train_full)

n_validation = int(
    np.floor(
        n_train_full * VALIDATION_FRACTION
    )
)

if n_validation <= 0:
    raise ValueError(
        "Validation set would be empty."
    )

split_index = (
    n_train_full - n_validation
)


X_train = X_train_full[
    :split_index
]

X_validation = X_train_full[
    split_index:
]

y_train_vol = y_train_vol_full[
    :split_index
]

y_validation_vol = y_train_vol_full[
    split_index:
]

train_dates = train_meta[
    "target_date"
].iloc[:split_index].reset_index(drop=True)

validation_dates = train_meta[
    "target_date"
].iloc[split_index:].reset_index(drop=True)


# ============================================================
# 12. DISPLAY SPLIT
# ============================================================

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
    f"Training dates:   "
    f"{train_dates.min()} → {train_dates.max()}"
)

print(
    f"Validation dates: "
    f"{validation_dates.min()} → {validation_dates.max()}"
)


if train_dates.max() >= validation_dates.min():
    raise ValueError(
        "Training and validation periods overlap."
    )

print(
    "PASS: Training and validation are chronologically separated."
)


# ============================================================
# 13. LOG1P TARGET TRANSFORMATION
# ============================================================
#
# Original target:
#
#       v = r²
#
# Transformed target:
#
#       z = log(1 + v)
#
# This reduces the effect of extreme right-skewness during
# neural-network training.
#
# ============================================================

print("\nTransforming volatility target with log1p...")

y_train = np.log1p(
    y_train_vol
)

y_validation = np.log1p(
    y_validation_vol
)


print(
    f"Original training target mean: "
    f"{np.mean(y_train_vol):.6f}"
)

print(
    f"Original training target max:  "
    f"{np.max(y_train_vol):.6f}"
)

print(
    f"Log1p training target mean:     "
    f"{np.mean(y_train):.6f}"
)

print(
    f"Log1p training target max:      "
    f"{np.max(y_train):.6f}"
)


# ============================================================
# 14. BUILD CNN ARCHITECTURE
# ============================================================

print("\n" + "-" * 70)
print("BUILDING 2D-CNN")
print("-" * 70)


model = Sequential([
    
    Input(
        shape=(4, 20, 1)
    ),

    # First convolution
    Conv2D(
        filters=32,
        kernel_size=(2, 3),
        activation="relu",
        padding="same"
    ),

    # Reduce spatial dimensions
    MaxPooling2D(
        pool_size=(2, 2)
    ),

    # Second convolution
    Conv2D(
        filters=64,
        kernel_size=(2, 3),
        activation="relu",
        padding="same"
    ),

    # Convert feature maps into a vector
    Flatten(),

    # Fully connected layer
    Dense(
        units=64,
        activation="relu"
    ),

    # Regularization
    Dropout(
        rate=0.30
    ),

    # One continuous volatility forecast
    Dense(
        units=1,
        activation="softplus"
    )
])


# ============================================================
# 15. COMPILE MODEL
# ============================================================
#
# Huber loss is used because volatility contains very large
# observations. It is less sensitive to extreme outliers
# than ordinary squared-error training while still learning
# a regression target.
#
# The evaluation stage will still report MAE and RMSE.
#
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
# 16. DISPLAY MODEL
# ============================================================

model.summary()


# ============================================================
# 17. CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=12,
    restore_best_weights=True,
    verbose=1
)


reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
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
# 18. TRAIN CNN
# ============================================================

print("\n" + "=" * 70)
print("STARTING CNN TRAINING")
print("=" * 70)

history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_validation,
        y_validation
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    shuffle=False,

    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ],

    verbose=1
)


# ============================================================
# 19. RESTORE BEST MODEL
# ============================================================

print("\nLoading best saved model...")

best_model = tf.keras.models.load_model(
    MODEL_FILE
)


# ============================================================
# 20. VALIDATION FORECAST
# ============================================================

print("\nGenerating validation forecasts...")

validation_pred_log = best_model.predict(
    X_validation,
    verbose=0
).reshape(-1)


# Convert log1p forecast back to original volatility scale
validation_pred_vol = np.expm1(
    validation_pred_log
)

# Numerical safety
validation_pred_vol = np.maximum(
    validation_pred_vol,
    0.0
)


# ============================================================
# 21. VALIDATION SPIKE FLAGS
# ============================================================

validation_actual_spike = (
    y_validation_vol
    > SPIKE_THRESHOLD
).astype(int)

validation_predicted_spike = (
    validation_pred_vol
    > SPIKE_THRESHOLD
).astype(int)


# ============================================================
# 22. VALIDATION SUMMARY
# ============================================================

validation_spikes_actual = int(
    validation_actual_spike.sum()
)

validation_spikes_predicted = int(
    validation_predicted_spike.sum()
)

validation_spike_rate = (
    validation_spikes_actual
    / len(validation_actual_spike)
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
    f"{validation_spikes_actual}"
)

print(
    f"Actual spike rate: "
    f"{validation_spike_rate:.4%}"
)

print(
    f"Predicted spikes: "
    f"{validation_spikes_predicted}"
)

print(
    f"Spike threshold: "
    f"{SPIKE_THRESHOLD}"
)


print("\nFirst 10 validation forecasts:")

validation_preview = pd.DataFrame({
    "date": validation_dates.iloc[:10],
    "actual_volatility": y_validation_vol[:10],
    "predicted_volatility": validation_pred_vol[:10],
    "actual_spike": validation_actual_spike[:10],
    "predicted_spike": validation_predicted_spike[:10]
})

print(
    validation_preview.to_string(
        index=False
    )
)


# ============================================================
# 23. SAVE TRAINING HISTORY
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
# 24. SAVE VALIDATION FORECASTS
# ============================================================

validation_results = pd.DataFrame({
    "target_date": validation_dates,
    "actual_volatility": y_validation_vol,
    "predicted_volatility": validation_pred_vol,
    "actual_spike": validation_actual_spike,
    "predicted_spike": validation_predicted_spike
})


validation_results.to_csv(
    VALIDATION_PREDICTIONS_FILE,
    index=False
)


# ============================================================
# 25. SAVE MODEL CONFIGURATION
# ============================================================

config = {
    "model_type": "Multivariate 2D-CNN",
    "input_shape": [4, 20, 1],
    "features": [
        "fx_log_return",
        "fx_return_squared",
        "brent_log_return",
        "reserves_log_change"
    ],
    "window_size": 20,
    "target": "next_period_fx_return_squared",
    "target_training_transform": "log1p",
    "inverse_transform": "expm1",
    "spike_threshold": SPIKE_THRESHOLD,
    "validation_fraction": VALIDATION_FRACTION,
    "epochs_requested": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "random_seed": RANDOM_SEED,
    "loss": "Huber",
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
# 26. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("STEP 10C FINISHED SUCCESSFULLY")
print("=" * 70)

print("\nSaved files:")

print(
    f"1. {MODEL_FILE}"
)

print(
    f"2. {HISTORY_FILE}"
)

print(
    f"3. {VALIDATION_PREDICTIONS_FILE}"
)

print(
    f"4. {CONFIG_FILE}"
)

print("\nCNN training is complete.")

print(
    "\nNext stage: Step 10D — evaluate the CNN on the "
    "untouched 2021–2025 test set."
)