# ============================================================
# STEP 10C-V3
# MULTITASK MULTIVARIATE 2D-CNN
#
# PRIMARY OBJECTIVE:
#   Forecast whether the next observation is a volatility spike
#
# SECONDARY OBJECTIVE:
#   Forecast the continuous next-period FX volatility
#
# DATA SPLIT:
#   TRAIN      : 2006-05-09 -> 2013-12-31
#   VALIDATION : 2014-01-02 -> 2020-12-31
#   TEST       : 2021-01-04 -> 2025-10-29
#
# TEST SET REMAINS COMPLETELY UNTOUCHED
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import Input, Model

from tensorflow.keras.layers import (
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
# 2. CONFIGURATION
# ============================================================

DATA_DIR = "data_cleaned"
RESULTS_DIR = "results"
MODEL_DIR = "models"


# ------------------------------------------------------------
# Input files
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Output files
# ------------------------------------------------------------

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "cnn_2d_multitask_model.keras"
)

HISTORY_FILE = os.path.join(
    RESULTS_DIR,
    "cnn_2d_multitask_training_history.csv"
)

VALIDATION_FILE = os.path.join(
    RESULTS_DIR,
    "cnn_2d_multitask_validation_predictions.csv"
)

CONFIG_FILE = os.path.join(
    RESULTS_DIR,
    "cnn_2d_multitask_config.json"
)


# ============================================================
# 3. PROJECT SETTINGS
# ============================================================

RANDOM_SEED = 42

VALIDATION_START = pd.Timestamp(
    "2014-01-01"
)

VALIDATION_END = pd.Timestamp(
    "2020-12-31"
)


# Frozen project spike threshold
SPIKE_THRESHOLD = 0.10927724


# Maximum number of epochs
EPOCHS = 100

BATCH_SIZE = 32

LEARNING_RATE = 0.001


# ------------------------------------------------------------
# Multi-task loss weights
#
# 30% continuous volatility forecasting
# 70% spike classification
# ------------------------------------------------------------

VOLATILITY_LOSS_WEIGHT = 0.30

SPIKE_LOSS_WEIGHT = 0.70


# ============================================================
# 4. REPRODUCIBILITY
# ============================================================

np.random.seed(
    RANDOM_SEED
)

tf.random.set_seed(
    RANDOM_SEED
)


# ============================================================
# 5. CREATE DIRECTORIES
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
# 6. HEADER
# ============================================================

print("=" * 70)
print("STEP 10C-V3: MULTITASK 2D-CNN")
print("=" * 70)


# ============================================================
# 7. LOAD TRAINING DATA
# ============================================================

print("\nLoading training CNN inputs...")

X_train_full = np.load(
    TRAIN_X_FILE
)

print(
    f"Training input shape: "
    f"{X_train_full.shape}"
)


print("\nLoading training volatility targets...")

y_vol_full = np.load(
    TRAIN_Y_FILE
)

print(
    f"Training target shape: "
    f"{y_vol_full.shape}"
)


print("\nLoading training metadata...")

train_meta = pd.read_csv(
    TRAIN_META_FILE
)

train_meta["target_date"] = pd.to_datetime(
    train_meta["target_date"]
)

train_meta = train_meta.reset_index(
    drop=True
)


# ============================================================
# 8. LOAD TEST DATA
# ============================================================
#
# We load the test set only to verify dimensions.
#
# It will NOT participate in:
#
#   - model fitting
#   - validation
#   - early stopping
#   - model selection
#   - threshold selection
#
# ============================================================

print("\nLoading test data for verification only...")

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

test_meta = test_meta.reset_index(
    drop=True
)


print(
    f"Test input shape: "
    f"{X_test.shape}"
)

print(
    f"Test target shape: "
    f"{y_test_vol.shape}"
)

print(
    "\nIMPORTANT: Test data remains untouched."
)


# ============================================================
# 9. INPUT CONSISTENCY CHECKS
# ============================================================

print("\n" + "-" * 70)
print("INPUT VALIDATION")
print("-" * 70)


if X_train_full.ndim != 4:

    raise ValueError(
        "Training CNN input must have 4 dimensions."
    )


if X_test.ndim != 4:

    raise ValueError(
        "Test CNN input must have 4 dimensions."
    )


expected_input_shape = (
    4,
    20,
    1
)


if X_train_full.shape[1:] != expected_input_shape:

    raise ValueError(
        "Training input must have shape "
        "(samples, 4, 20, 1)."
    )


if X_test.shape[1:] != expected_input_shape:

    raise ValueError(
        "Test input must have shape "
        "(samples, 4, 20, 1)."
    )


if len(X_train_full) != len(y_vol_full):

    raise ValueError(
        "Training X and y lengths do not match."
    )


if len(X_test) != len(y_test_vol):

    raise ValueError(
        "Test X and y lengths do not match."
    )


if len(X_train_full) != len(train_meta):

    raise ValueError(
        "Training metadata length does not match training data."
    )


if len(X_test) != len(test_meta):

    raise ValueError(
        "Test metadata length does not match test data."
    )


print(
    "PASS: Input dimensions and lengths are consistent."
)


# ============================================================
# 10. INVALID VALUE CHECKS
# ============================================================

print("\nChecking for NaN and Inf values...")


if np.isnan(X_train_full).any():

    raise ValueError(
        "NaN detected in training inputs."
    )


if np.isinf(X_train_full).any():

    raise ValueError(
        "Inf detected in training inputs."
    )


if np.isnan(X_test).any():

    raise ValueError(
        "NaN detected in test inputs."
    )


if np.isinf(X_test).any():

    raise ValueError(
        "Inf detected in test inputs."
    )


if np.isnan(y_vol_full).any():

    raise ValueError(
        "NaN detected in training targets."
    )


if np.isnan(y_test_vol).any():

    raise ValueError(
        "NaN detected in test targets."
    )


if np.any(y_vol_full < 0):

    raise ValueError(
        "Negative training volatility detected."
    )


if np.any(y_test_vol < 0):

    raise ValueError(
        "Negative test volatility detected."
    )


print(
    "PASS: No invalid values detected."
)


# ============================================================
# 11. CHRONOLOGICAL TRAIN / VALIDATION SPLIT
# ============================================================
#
# Full model-development data:
#
#   2006 -> 2020
#
# Split into:
#
#   Training:
#       2006 -> 2013
#
#   Validation:
#       2014 -> 2020
#
# Final test:
#       2021 -> 2025
#
# ============================================================

train_dates_full = (
    train_meta["target_date"]
)


train_mask = (
    train_dates_full
    < VALIDATION_START
)


validation_mask = (
    (train_dates_full >= VALIDATION_START)
    &
    (train_dates_full <= VALIDATION_END)
)


train_indices = np.where(
    train_mask.to_numpy()
)[0]


validation_indices = np.where(
    validation_mask.to_numpy()
)[0]


# ------------------------------------------------------------
# Select training data
# ------------------------------------------------------------

X_train = X_train_full[
    train_indices
]

y_train_vol = y_vol_full[
    train_indices
]

train_dates = train_dates_full.iloc[
    train_indices
].reset_index(drop=True)


# ------------------------------------------------------------
# Select validation data
# ------------------------------------------------------------

X_validation = X_train_full[
    validation_indices
]

y_validation_vol = y_vol_full[
    validation_indices
]

validation_dates = train_dates_full.iloc[
    validation_indices
].reset_index(drop=True)


# ============================================================
# 12. DISPLAY CHRONOLOGICAL SPLIT
# ============================================================

print("\n" + "-" * 70)
print("CHRONOLOGICAL TRAIN / VALIDATION SPLIT")
print("-" * 70)


print(
    f"Training observations:   "
    f"{len(X_train)}"
)


print(
    f"Validation observations: "
    f"{len(X_validation)}"
)


print(
    f"Training period: "
    f"{train_dates.min()} "
    f"→ "
    f"{train_dates.max()}"
)


print(
    f"Validation period: "
    f"{validation_dates.min()} "
    f"→ "
    f"{validation_dates.max()}"
)


if train_dates.max() >= validation_dates.min():

    raise ValueError(
        "Training and validation periods overlap."
    )


print(
    "PASS: Training and validation are chronologically separated."
)


# ============================================================
# 13. CREATE SPIKE LABELS
# ============================================================
#
# Spike definition is frozen:
#
#       r² > 0.10927724
#
# ============================================================

y_train_spike = (
    y_train_vol > SPIKE_THRESHOLD
).astype(np.int32)


y_validation_spike = (
    y_validation_vol > SPIKE_THRESHOLD
).astype(np.int32)


# ============================================================
# 14. SPIKE DISTRIBUTION
# ============================================================

train_spikes = int(
    y_train_spike.sum()
)

train_nonspikes = int(
    len(y_train_spike) - train_spikes
)


validation_spikes = int(
    y_validation_spike.sum()
)

validation_nonspikes = int(
    len(y_validation_spike)
    - validation_spikes
)


print("\n" + "-" * 70)
print("SPIKE DISTRIBUTION")
print("-" * 70)


print(
    f"Training non-spikes: "
    f"{train_nonspikes}"
)


print(
    f"Training spikes:     "
    f"{train_spikes}"
)


print(
    f"Training spike rate: "
    f"{train_spikes / len(y_train_spike):.4%}"
)


print(
    f"\nValidation non-spikes: "
    f"{validation_nonspikes}"
)


print(
    f"Validation spikes:     "
    f"{validation_spikes}"
)


print(
    f"Validation spike rate: "
    f"{validation_spikes / len(y_validation_spike):.4%}"
)


print(
    f"\nFrozen spike threshold: "
    f"{SPIKE_THRESHOLD}"
)


if train_spikes == 0:

    raise ValueError(
        "No spike observations exist in the training set."
    )


if validation_spikes == 0:

    raise ValueError(
        "No spike observations exist in the validation set."
    )


# ============================================================
# 15. LOG1P VOLATILITY TARGET
# ============================================================
#
# Continuous target:
#
#       v = r²
#
# Transformation:
#
#       z = log(1 + v)
#
# ============================================================

print("\n" + "-" * 70)
print("VOLATILITY TARGET TRANSFORMATION")
print("-" * 70)


y_train_logvol = np.log1p(
    y_train_vol
)


y_validation_logvol = np.log1p(
    y_validation_vol
)


print(
    f"Training original mean: "
    f"{np.mean(y_train_vol):.8f}"
)


print(
    f"Training original maximum: "
    f"{np.max(y_train_vol):.8f}"
)


print(
    f"Training log1p mean: "
    f"{np.mean(y_train_logvol):.8f}"
)


print(
    f"Training log1p maximum: "
    f"{np.max(y_train_logvol):.8f}"
)


# ============================================================
# 16. CALCULATE BALANCED CLASS WEIGHTS
# ============================================================
#
# Instead of manually selecting a spike weight,
# compute balanced weights directly from training frequency.
#
# ============================================================

classes = np.array([
    0,
    1
])


balanced_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train_spike
)


nonspike_weight = float(
    balanced_weights[0]
)


spike_weight = float(
    balanced_weights[1]
)


print("\n" + "-" * 70)
print("BALANCED CLASS WEIGHTS")
print("-" * 70)


print(
    f"Non-spike weight: "
    f"{nonspike_weight:.6f}"
)


print(
    f"Spike weight:     "
    f"{spike_weight:.6f}"
)


# ============================================================
# 17. CREATE SAMPLE WEIGHTS
# ============================================================
#
# We have two model outputs:
#
#   output 1 = volatility
#   output 2 = spike probability
#
# Therefore the sample-weight list follows the exact same
# order:
#
#   [volatility weights, spike weights]
#
# ============================================================

# Every observation has equal importance for regression
train_volatility_weights = np.ones(
    len(y_train_vol),
    dtype=np.float32
)


# Classification head receives balanced class weights
train_spike_weights = np.where(
    y_train_spike == 1,
    spike_weight,
    nonspike_weight
).astype(np.float32)


# For validation we do not artificially reweight observations.
# This keeps validation metrics easier to interpret.
validation_volatility_weights = np.ones(
    len(y_validation_vol),
    dtype=np.float32
)


validation_spike_weights = np.ones(
    len(y_validation_spike),
    dtype=np.float32
)


print(
    "\nPASS: Sample weights created."
)


# ============================================================
# 18. BUILD SHARED CNN
# ============================================================

print("\n" + "=" * 70)
print("BUILDING MULTITASK 2D-CNN")
print("=" * 70)


# ------------------------------------------------------------
# Input
# ------------------------------------------------------------

inputs = Input(
    shape=(4, 20, 1),
    name="cnn_input"
)


# ------------------------------------------------------------
# Convolution block 1
# ------------------------------------------------------------

x = Conv2D(
    filters=32,
    kernel_size=(2, 3),
    activation="relu",
    padding="same",
    name="conv2d_block1"
)(inputs)


x = MaxPooling2D(
    pool_size=(2, 2),
    name="maxpool_block1"
)(x)


# ------------------------------------------------------------
# Convolution block 2
# ------------------------------------------------------------

x = Conv2D(
    filters=64,
    kernel_size=(2, 3),
    activation="relu",
    padding="same",
    name="conv2d_block2"
)(x)


# ------------------------------------------------------------
# Shared feature representation
# ------------------------------------------------------------

x = Flatten(
    name="flatten_shared"
)(x)


x = Dense(
    units=64,
    activation="relu",
    name="shared_dense"
)(x)


x = Dropout(
    rate=0.30,
    name="shared_dropout"
)(x)


# ============================================================
# 19. VOLATILITY REGRESSION HEAD
# ============================================================

volatility_branch = Dense(
    units=32,
    activation="relu",
    name="volatility_hidden"
)(x)


volatility_output = Dense(
    units=1,
    activation="softplus",
    name="volatility"
)(volatility_branch)


# ============================================================
# 20. SPIKE CLASSIFICATION HEAD
# ============================================================

spike_branch = Dense(
    units=32,
    activation="relu",
    name="spike_hidden"
)(x)


spike_output = Dense(
    units=1,
    activation="sigmoid",
    name="spike_probability"
)(spike_branch)


# ============================================================
# 21. CREATE MULTI-OUTPUT MODEL
# ============================================================
#
# IMPORTANT:
# The order is:
#
#   [volatility, spike_probability]
#
# Targets and sample weights below use exactly this order.
#
# ============================================================

model = Model(
    inputs=inputs,

    outputs=[
        volatility_output,
        spike_output
    ],

    name="multitask_2d_cnn"
)


# ============================================================
# 22. COMPILE MODEL
# ============================================================
#
# Output 1:
#     Huber regression loss
#
# Output 2:
#     Binary cross-entropy classification loss
#
# Overall:
#
#     0.30 × volatility loss
#     +
#     0.70 × spike loss
#
# ============================================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)


model.compile(

    optimizer=optimizer,

    loss=[
        tf.keras.losses.Huber(),
        tf.keras.losses.BinaryCrossentropy()
    ],

    loss_weights=[
        VOLATILITY_LOSS_WEIGHT,
        SPIKE_LOSS_WEIGHT
    ],

    metrics=[
        [
            tf.keras.metrics.MeanAbsoluteError(
                name="mae"
            )
        ],

        [
            tf.keras.metrics.AUC(
                name="auc",
                curve="ROC"
            ),

            tf.keras.metrics.AUC(
                name="pr_auc",
                curve="PR"
            ),

            tf.keras.metrics.Precision(
                name="precision"
            ),

            tf.keras.metrics.Recall(
                name="recall"
            )
        ]
    ]
)


# ============================================================
# 23. DISPLAY MODEL
# ============================================================

model.summary()


# ============================================================
# 24. CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(

    MODEL_FILE,

    monitor="val_spike_probability_auc",

    mode="max",

    save_best_only=True,

    verbose=1
)


early_stopping = EarlyStopping(

    monitor="val_spike_probability_auc",

    mode="max",

    patience=15,

    min_delta=1e-4,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_spike_probability_auc",

    mode="max",

    factor=0.5,

    patience=6,

    min_lr=1e-6,

    verbose=1
)


# ============================================================
# 25. PREPARE TARGETS
# ============================================================
#
# EXACT MODEL OUTPUT ORDER:
#
#     1. volatility
#     2. spike_probability
#
# ============================================================

y_train_targets = [

    y_train_logvol,

    y_train_spike

]


y_validation_targets = [

    y_validation_logvol,

    y_validation_spike

]


# ============================================================
# 26. PREPARE SAMPLE WEIGHTS
# ============================================================

train_sample_weights = [

    train_volatility_weights,

    train_spike_weights

]


validation_sample_weights = [

    validation_volatility_weights,

    validation_spike_weights

]


# ============================================================
# 27. TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("STARTING MULTITASK CNN TRAINING")
print("=" * 70)


history = model.fit(

    X_train,

    y_train_targets,

    sample_weight=train_sample_weights,

    validation_data=(

        X_validation,

        y_validation_targets,

        validation_sample_weights

    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    # Preserve chronological order
    shuffle=False,

    callbacks=[

        checkpoint,

        early_stopping,

        reduce_lr

    ],

    verbose=1
)


# ============================================================
# 28. LOAD BEST MODEL
# ============================================================

print("\nLoading best saved CNN model...")


best_model = tf.keras.models.load_model(
    MODEL_FILE
)


# ============================================================
# 29. VALIDATION PREDICTIONS
# ============================================================

print("\nGenerating validation predictions...")


validation_predictions = best_model.predict(

    X_validation,

    verbose=0
)


# Model output order:
#
#   validation_predictions[0]
#       = volatility
#
#   validation_predictions[1]
#       = spike probability


validation_pred_logvol = (
    validation_predictions[0]
    .reshape(-1)
)


validation_pred_spike_probability = (
    validation_predictions[1]
    .reshape(-1)
)


# ============================================================
# 30. CONVERT VOLATILITY BACK TO ORIGINAL SCALE
# ============================================================
#
# Model predicts:
#
#       log(1 + volatility)
#
# Therefore:
#
#       volatility = exp(prediction) - 1
#
# ============================================================

validation_pred_volatility = np.expm1(
    validation_pred_logvol
)


# Numerical safety
validation_pred_volatility = np.maximum(
    validation_pred_volatility,
    0.0
)


# ============================================================
# 31. CREATE BINARY SPIKE PREDICTION
# ============================================================
#
# The classification head gives:
#
#       P(spike)
#
# We use 0.50 as the initial classification cutoff.
#
# IMPORTANT:
# This 0.50 probability cutoff is different from the
# volatility threshold 0.10927724.
#
# ============================================================

CLASSIFICATION_CUTOFF = 0.50


validation_pred_spike = (
    validation_pred_spike_probability
    >= CLASSIFICATION_CUTOFF
).astype(np.int32)


# ============================================================
# 32. VALIDATION SUMMARY
# ============================================================

actual_spike_count = int(
    y_validation_spike.sum()
)


predicted_spike_count = int(
    validation_pred_spike.sum()
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
    f"\nClassification cutoff: "
    f"{CLASSIFICATION_CUTOFF}"
)


print(
    f"Volatility spike threshold: "
    f"{SPIKE_THRESHOLD}"
)


# ============================================================
# 33. CONTINUOUS FORECAST SUMMARY
# ============================================================

print("\n" + "-" * 70)
print("CONTINUOUS VOLATILITY FORECAST")
print("-" * 70)


print(
    f"Mean actual volatility: "
    f"{np.mean(y_validation_vol):.8f}"
)


print(
    f"Mean predicted volatility: "
    f"{np.mean(validation_pred_volatility):.8f}"
)


print(
    f"Maximum actual volatility: "
    f"{np.max(y_validation_vol):.8f}"
)


print(
    f"Maximum predicted volatility: "
    f"{np.max(validation_pred_volatility):.8f}"
)


# ============================================================
# 34. PROBABILITY DISTRIBUTION
# ============================================================

print("\n" + "-" * 70)
print("SPIKE PROBABILITY DISTRIBUTION")
print("-" * 70)


for q in [
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    1.00
]:

    print(
        f"{q:.0%} percentile: "
        f"{np.quantile(validation_pred_spike_probability, q):.6f}"
    )


# ============================================================
# 35. FIRST 20 VALIDATION FORECASTS
# ============================================================

print("\n" + "-" * 70)
print("FIRST 20 VALIDATION FORECASTS")
print("-" * 70)


validation_preview = pd.DataFrame({

    "target_date":
        validation_dates.iloc[:20],

    "actual_volatility":
        y_validation_vol[:20],

    "predicted_volatility":
        validation_pred_volatility[:20],

    "actual_spike":
        y_validation_spike[:20],

    "predicted_spike_probability":
        validation_pred_spike_probability[:20],

    "predicted_spike":
        validation_pred_spike[:20]

})


print(
    validation_preview.to_string(
        index=False
    )
)


# ============================================================
# 36. SAVE VALIDATION PREDICTIONS
# ============================================================

validation_results = pd.DataFrame({

    "target_date":
        validation_dates,

    "actual_volatility":
        y_validation_vol,

    "predicted_volatility":
        validation_pred_volatility,

    "actual_spike":
        y_validation_spike,

    "predicted_spike_probability":
        validation_pred_spike_probability,

    "predicted_spike":
        validation_pred_spike

})


validation_results.to_csv(
    VALIDATION_FILE,
    index=False
)


print(
    f"\nSaved validation forecasts: "
    f"{VALIDATION_FILE}"
)


# ============================================================
# 37. SAVE TRAINING HISTORY
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


print(
    f"Saved training history: "
    f"{HISTORY_FILE}"
)


# ============================================================
# 38. SAVE MODEL CONFIGURATION
# ============================================================

config = {

    "model_type":
        "Multitask Multivariate 2D-CNN",

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

    "continuous_target":
        "next_period_fx_return_squared",

    "continuous_target_transform":
        "log1p",

    "continuous_inverse_transform":
        "expm1",

    "spike_definition":
        "next_period_fx_return_squared > frozen threshold",

    "spike_threshold":
        SPIKE_THRESHOLD,

    "classification_cutoff":
        CLASSIFICATION_CUTOFF,

    "training_period":
        "2006-05-09 to 2013-12-31",

    "validation_period":
        "2014-01-02 to 2020-12-31",

    "final_test_period":
        "2021-01-04 to 2025-10-29",

    "volatility_loss":
        "Huber",

    "spike_loss":
        "BinaryCrossentropy",

    "volatility_loss_weight":
        VOLATILITY_LOSS_WEIGHT,

    "spike_loss_weight":
        SPIKE_LOSS_WEIGHT,

    "nonspike_class_weight":
        nonspike_weight,

    "spike_class_weight":
        spike_weight,

    "epochs_max":
        EPOCHS,

    "batch_size":
        BATCH_SIZE,

    "learning_rate":
        LEARNING_RATE,

    "random_seed":
        RANDOM_SEED,

    "shuffle":
        False,

    "architecture": [

        "Conv2D(32, kernel=(2,3), activation=relu)",

        "MaxPooling2D(pool=(2,2))",

        "Conv2D(64, kernel=(2,3), activation=relu)",

        "Flatten",

        "Dense(64, activation=relu)",

        "Dropout(0.30)",

        "Volatility Dense(32, activation=relu)",

        "Volatility output Dense(1, activation=softplus)",

        "Spike Dense(32, activation=relu)",

        "Spike output Dense(1, activation=sigmoid)"

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


print(
    f"Saved model configuration: "
    f"{CONFIG_FILE}"
)


# ============================================================
# 39. FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("STEP 10C-V3 FINISHED SUCCESSFULLY")
print("=" * 70)


print(
    "\nBest model:"
)

print(
    MODEL_FILE
)


print(
    "\nTraining history:"
)

print(
    HISTORY_FILE
)


print(
    "\nValidation predictions:"
)

print(
    VALIDATION_FILE
)


print(
    "\nConfiguration:"
)

print(
    CONFIG_FILE
)


print(
    "\nFINAL TEST SET REMAINS UNTOUCHED."
)


print(
    "\nNext stage: evaluate the multitask CNN "
    "validation performance before Step 10D."
)