# ============================================================
# STEP 10A
# Build 20-Observation Multivariate 2D-CNN Windows
# Gap-Safe + Target-Safe Version
# ============================================================

import os
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_FILE = (
    "data_cleaned/"
    "forecasting_dataset_2006_2025_spike_gap_target_safe.csv"
)

OUTPUT_DIR = "data_cleaned"

WINDOW_SIZE = 20

# Date used to separate training and testing
TRAIN_END_DATE = pd.Timestamp("2020-12-31")

# Variables used as CNN input features
FEATURE_COLUMNS = [
    "fx_log_return",
    "fx_return_squared",
    "brent_log_return",
    "reserves_log_change"
]

# Target columns
TARGET_COLUMN = "target_volatility"
SPIKE_COLUMN = "target_spike"

DATE_COLUMN = "date"
SEGMENT_COLUMN = "segment_id"

THRESHOLD = 0.10927724


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 10A: BUILDING CNN WINDOWS")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df[DATE_COLUMN].min()} to {df[DATE_COLUMN].max()}")


# ============================================================
# 3. BASIC VALIDATION
# ============================================================

required_columns = (
    [DATE_COLUMN, SEGMENT_COLUMN]
    + FEATURE_COLUMNS
    + [TARGET_COLUMN, SPIKE_COLUMN]
)

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("\nAll required columns are present.")


# ============================================================
# 4. SORT DATA
# ============================================================

df = df.sort_values(
    [SEGMENT_COLUMN, DATE_COLUMN]
).reset_index(drop=True)


# ============================================================
# 5. STORAGE FOR WINDOWS
# ============================================================

X_windows = []

y_volatility = []
y_spike = []

target_dates = []
origin_dates = []

skipped_short_segments = 0
skipped_missing_targets = 0
skipped_missing_features = 0

segment_window_counts = {}


# ============================================================
# 6. BUILD WINDOWS SEGMENT-BY-SEGMENT
# ============================================================
#
# Each CNN input has:
#
#     4 variables × 20 observations
#
# The window is constructed entirely inside one
# continuous segment so that it never crosses a
# long data gap.
#
# For a given origin i:
#
#     Historical window:
#     rows [i-20, ..., i-1]
#
#     Target:
#     target_volatility stored on row i
#
#     target_spike:
#     spike indicator stored on row i
#
# ============================================================

print("\nBuilding windows...")

for segment_id, segment in df.groupby(
    SEGMENT_COLUMN,
    sort=True
):

    segment = segment.sort_values(
        DATE_COLUMN
    ).reset_index(drop=True)

    segment_length = len(segment)

    # --------------------------------------------------------
    # A segment must contain at least WINDOW_SIZE + 1 rows
    # --------------------------------------------------------

    if segment_length <= WINDOW_SIZE:
        skipped_short_segments += 1

        print(
            f"Skipping segment {segment_id}: "
            f"only {segment_length} observations."
        )

        continue

    segment_window_counts[segment_id] = 0

    # --------------------------------------------------------
    # Start at WINDOW_SIZE because we need 20 historical rows
    # --------------------------------------------------------

    for i in range(WINDOW_SIZE, segment_length):

        # ----------------------------------------------------
        # Historical 20-observation window
        # ----------------------------------------------------

        window = segment.iloc[
            i - WINDOW_SIZE:i
        ][FEATURE_COLUMNS]

        # ----------------------------------------------------
        # Target is stored on the current row.
        #
        # Some rows at segment boundaries have no valid
        # next-observation target, therefore target values
        # can legitimately be NaN.
        # ----------------------------------------------------

        origin_row = segment.iloc[i]

        target_value_raw = origin_row[TARGET_COLUMN]
        target_spike_raw = origin_row[SPIKE_COLUMN]

        # ----------------------------------------------------
        # FIX:
        # Do not convert NaN target values to float/int.
        # Skip them instead.
        # ----------------------------------------------------

        if (
            pd.isna(target_value_raw)
            or pd.isna(target_spike_raw)
        ):
            skipped_missing_targets += 1
            continue

        # ----------------------------------------------------
        # Check for missing feature values
        # ----------------------------------------------------

        if window.isna().any().any():
            skipped_missing_features += 1
            continue

        # ----------------------------------------------------
        # Convert target values safely
        # ----------------------------------------------------

        target_value = float(target_value_raw)
        target_spike = int(target_spike_raw)

        # ----------------------------------------------------
        # Convert the 20 × 4 dataframe to:
        #
        #     4 × 20
        #
        # This gives:
        #
        # rows    = variables
        # columns = time observations
        #
        # which is the conceptual 2D-CNN image.
        # ----------------------------------------------------

        window_matrix = window.to_numpy(
            dtype=np.float32
        ).T

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        X_windows.append(window_matrix)

        y_volatility.append(target_value)
        y_spike.append(target_spike)

        origin_dates.append(
            origin_row[DATE_COLUMN]
        )

        # The target date is the date represented by
        # the target value stored on the current row.
        target_dates.append(
            origin_row[DATE_COLUMN]
        )

        segment_window_counts[segment_id] += 1


# ============================================================
# 7. CONVERT TO NUMPY ARRAYS
# ============================================================

X = np.asarray(
    X_windows,
    dtype=np.float32
)

y_volatility = np.asarray(
    y_volatility,
    dtype=np.float32
)

y_spike = np.asarray(
    y_spike,
    dtype=np.int32
)

target_dates = pd.to_datetime(
    target_dates
)

origin_dates = pd.to_datetime(
    origin_dates
)


# ============================================================
# 8. CHECK THAT WINDOWS WERE CREATED
# ============================================================

if len(X) == 0:
    raise ValueError(
        "No valid CNN windows were created. "
        "Check the dataset, segment structure, "
        "and target columns."
    )


# ============================================================
# 9. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

train_mask = (
    target_dates <= TRAIN_END_DATE
)

test_mask = (
    target_dates > TRAIN_END_DATE
)

X_train = X[train_mask]
X_test = X[test_mask]

y_train_volatility = y_volatility[train_mask]
y_test_volatility = y_volatility[test_mask]

y_train_spike = y_spike[train_mask]
y_test_spike = y_spike[test_mask]

train_dates = target_dates[train_mask]
test_dates = target_dates[test_mask]


# ============================================================
# 10. CNN INPUT SHAPE
# ============================================================
#
# TensorFlow/Keras Conv2D expects:
#
#     samples × rows × columns × channels
#
# Therefore:
#
#     (N, 4, 20)
#
# becomes:
#
#     (N, 4, 20, 1)
#
# ============================================================

X_train_cnn = X_train[..., np.newaxis]
X_test_cnn = X_test[..., np.newaxis]


# ============================================================
# 11. PRINT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("WINDOW CONSTRUCTION COMPLETE")
print("=" * 70)

print(f"\nTotal valid windows: {len(X)}")

print(
    f"Skipped short segments: "
    f"{skipped_short_segments}"
)

print(
    f"Skipped missing targets: "
    f"{skipped_missing_targets}"
)

print(
    f"Skipped missing feature windows: "
    f"{skipped_missing_features}"
)

print("\nOriginal CNN matrix shape:")
print(
    f"X = {X.shape}"
)

print("\nCNN-ready shape:")
print(
    f"X_train = {X_train_cnn.shape}"
)

print(
    f"X_test  = {X_test_cnn.shape}"
)

print("\nTraining period:")
print(
    f"{train_dates.min()} "
    f"to "
    f"{train_dates.max()}"
)

print("\nTesting period:")
print(
    f"{test_dates.min()} "
    f"to "
    f"{test_dates.max()}"
)


# ============================================================
# 12. SPIKE DISTRIBUTION
# ============================================================

print("\n" + "-" * 70)
print("SPIKE DISTRIBUTION")
print("-" * 70)

train_spikes = int(
    y_train_spike.sum()
)

test_spikes = int(
    y_test_spike.sum()
)

train_spike_rate = (
    train_spikes / len(y_train_spike)
    if len(y_train_spike) > 0
    else np.nan
)

test_spike_rate = (
    test_spikes / len(y_test_spike)
    if len(y_test_spike) > 0
    else np.nan
)

print(
    f"Training observations: {len(y_train_spike)}"
)

print(
    f"Training spikes: {train_spikes}"
)

print(
    f"Training spike rate: "
    f"{train_spike_rate:.4%}"
)

print(
    f"\nTesting observations: {len(y_test_spike)}"
)

print(
    f"Testing spikes: {test_spikes}"
)

print(
    f"Testing spike rate: "
    f"{test_spike_rate:.4%}"
)


# ============================================================
# 13. DISPLAY FIRST WINDOW
# ============================================================

print("\n" + "-" * 70)
print("FIRST CNN WINDOW")
print("-" * 70)

print(
    "\nShape:",
    X_train[0].shape
)

print(
    "\nVariables × 20 observations:"
)

first_window_df = pd.DataFrame(
    X_train[0],
    index=FEATURE_COLUMNS,
    columns=[
        f"t-{i}"
        for i in range(WINDOW_SIZE, 0, -1)
    ]
)

print(
    first_window_df.to_string()
)


# ============================================================
# 14. FIRST TARGET
# ============================================================

print("\n" + "-" * 70)
print("FIRST TARGET")
print("-" * 70)

print(
    f"Target date: {train_dates.iloc[0] if hasattr(train_dates, 'iloc') else train_dates[0]}"
)

print(
    f"Target volatility: "
    f"{y_train_volatility[0]:.8f}"
)

print(
    f"Target spike: "
    f"{y_train_spike[0]}"
)

print(
    f"Spike threshold: "
    f"{THRESHOLD}"
)


# ============================================================
# 15. TARGET / SPIKE CONSISTENCY CHECK
# ============================================================

calculated_spikes = (
    y_volatility > THRESHOLD
).astype(int)

spike_mismatches = np.sum(
    calculated_spikes != y_spike
)

print("\n" + "-" * 70)
print("SPIKE CONSISTENCY CHECK")
print("-" * 70)

print(
    f"Spike threshold: {THRESHOLD}"
)

print(
    f"Number of mismatches: "
    f"{spike_mismatches}"
)

if spike_mismatches == 0:
    print(
        "PASS: target_spike agrees with "
        "target_volatility > threshold."
    )
else:
    print(
        "WARNING: target_spike does not fully "
        "agree with the threshold rule."
    )


# ============================================================
# 16. TEMPORAL ORDER CHECK
# ============================================================

print("\n" + "-" * 70)
print("TEMPORAL ORDER CHECK")
print("-" * 70)

if len(train_dates) > 1:

    train_dates_array = np.asarray(
        train_dates
    )

    train_ordered = np.all(
        train_dates_array[1:]
        >= train_dates_array[:-1]
    )

else:
    train_ordered = True


if len(test_dates) > 1:

    test_dates_array = np.asarray(
        test_dates
    )

    test_ordered = np.all(
        test_dates_array[1:]
        >= test_dates_array[:-1]
    )

else:
    test_ordered = True


print(
    f"Training dates chronological: "
    f"{train_ordered}"
)

print(
    f"Testing dates chronological: "
    f"{test_ordered}"
)

if len(train_dates) > 0 and len(test_dates) > 0:

    print(
        f"Last training target: "
        f"{train_dates.max()}"
    )

    print(
        f"First testing target: "
        f"{test_dates.min()}"
    )

    if train_dates.max() < test_dates.min():

        print(
            "PASS: No chronological overlap "
            "between training and testing."
        )

    else:

        print(
            "WARNING: Training and testing "
            "dates overlap."
        )


# ============================================================
# 17. SAVE WINDOWS
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "cnn_X_train.npy"
    ),
    X_train_cnn
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "cnn_X_test.npy"
    ),
    X_test_cnn
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "cnn_y_train_volatility.npy"
    ),
    y_train_volatility
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "cnn_y_test_volatility.npy"
    ),
    y_test_volatility
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "cnn_y_train_spike.npy"
    ),
    y_train_spike
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "cnn_y_test_spike.npy"
    ),
    y_test_spike
)


# ============================================================
# 18. SAVE DATE INFORMATION
# ============================================================

train_metadata = pd.DataFrame({
    "target_date": train_dates,
    "target_volatility": y_train_volatility,
    "target_spike": y_train_spike
})

test_metadata = pd.DataFrame({
    "target_date": test_dates,
    "target_volatility": y_test_volatility,
    "target_spike": y_test_spike
})

train_metadata.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cnn_train_metadata.csv"
    ),
    index=False
)

test_metadata.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cnn_test_metadata.csv"
    ),
    index=False
)


# ============================================================
# 19. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("STEP 10A FINISHED SUCCESSFULLY")
print("=" * 70)

print("\nSaved files:")

print(
    "1. data_cleaned/cnn_X_train.npy"
)

print(
    "2. data_cleaned/cnn_X_test.npy"
)

print(
    "3. data_cleaned/cnn_y_train_volatility.npy"
)

print(
    "4. data_cleaned/cnn_y_test_volatility.npy"
)

print(
    "5. data_cleaned/cnn_y_train_spike.npy"
)

print(
    "6. data_cleaned/cnn_y_test_spike.npy"
)

print(
    "7. data_cleaned/cnn_train_metadata.csv"
)

print(
    "8. data_cleaned/cnn_test_metadata.csv"
)

print("\nReady for Step 10B: CNN preprocessing/scaling.")