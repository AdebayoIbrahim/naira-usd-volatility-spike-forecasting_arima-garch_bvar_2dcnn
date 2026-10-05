# ============================================================
# STEP 09I: BVAR(8) WALK-FORWARD VOLATILITY FORECASTING
# ============================================================

import os
import numpy as np
import pandas as pd

from scipy.stats import norm


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "data_cleaned/"
    "forecasting_dataset_2006_2025_spike_gap_target_safe.csv"
)

STATE_FILE = (
    "results/"
    "bvar8_model_state_spike.npz"
)

OUTPUT_FILE = (
    "results/"
    "bvar8_spike_walk_forward_forecasts.csv"
)

EXCLUDED_OUTPUT_FILE = (
    "results/"
    "bvar8_excluded_forecast_origins.csv"
)

TRAIN_END = "2020-12-31"

SPIKE_THRESHOLD = 0.10927724

PROBABILITY_CUTOFF = 0.50

RANDOM_SEED = 42


# ============================================================
# VARIABLES
# ============================================================

VARIABLES = [
    "fx_log_return",
    "fx_return_squared",
    "brent_log_return",
    "reserves_log_change"
]

TARGET_VARIABLE = "fx_return_squared"

K = len(VARIABLES)


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)

df["date"] = pd.to_datetime(
    df["date"]
)

df["target_date"] = pd.to_datetime(
    df["target_date"]
)

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)


print("=" * 90)
print("STEP 09I: BVAR(8) WALK-FORWARD VOLATILITY FORECASTING")
print("=" * 90)

print(
    f"\nRows loaded: {len(df):,}"
)


# ============================================================
# LOAD BVAR MODEL STATE
# ============================================================

state = np.load(
    STATE_FILE,
    allow_pickle=True
)

posterior_B = state[
    "posterior_B"
]

posterior_Sigma = state[
    "posterior_Sigma"
]

means = state[
    "means"
]

stds = state[
    "stds"
]

saved_lag_order = int(
    state["lag_order"]
)

saved_train_end = str(
    state["train_end"]
)


# ============================================================
# MODEL CHECKS
# ============================================================

if saved_lag_order != 8:

    raise ValueError(
        f"Expected BVAR(8), but saved model "
        f"has lag order {saved_lag_order}."
    )


if saved_train_end != TRAIN_END:

    print(
        "\nWarning: saved training-end date is "
        f"{saved_train_end}, expected {TRAIN_END}."
    )


N_DRAWS = posterior_B.shape[0]

M = posterior_B.shape[1]


print("\n" + "=" * 90)
print("LOADED BVAR MODEL")
print("=" * 90)

print(
    f"\nPosterior draws: {N_DRAWS:,}"
)

print(
    f"Regressors per equation: {M}"
)

print(
    f"Lag order: VAR({saved_lag_order})"
)

print(
    f"Frozen spike threshold: "
    f"{SPIKE_THRESHOLD:.8f}"
)


# ============================================================
# CHECK MODEL DIMENSIONS
# ============================================================

expected_M = (
    1
    + K * saved_lag_order
)

if M != expected_M:

    raise ValueError(
        f"Unexpected number of regressors. "
        f"Expected {expected_M}, found {M}."
    )


expected_sigma_shape = (
    N_DRAWS,
    K,
    K
)

if posterior_Sigma.shape != (
    expected_sigma_shape
):

    raise ValueError(
        "Posterior covariance dimensions "
        "do not match the model."
    )


# ============================================================
# TRAINING / TEST SPLIT
# ============================================================

train = df[
    df["date"] <= TRAIN_END
].copy()

test = df[
    df["date"] > TRAIN_END
].copy()


# ------------------------------------------------------------
# Only observations with a valid next-observation target
# can be used as forecast origins.
# ------------------------------------------------------------

test = test[
    test["target_date"].notna()
].copy()


print("\n" + "=" * 90)
print("FORECAST PERIOD")
print("=" * 90)

print(
    f"\nPotential forecast origins: "
    f"{len(test):,}"
)

print(
    f"First potential origin: "
    f"{test['date'].min().date()}"
)

print(
    f"Last potential origin: "
    f"{test['date'].max().date()}"
)


# ============================================================
# INITIALIZE STORAGE
# ============================================================

forecast_rows = []

excluded_origins = []


# ============================================================
# RANDOM NUMBER GENERATOR
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)


# ============================================================
# TARGET VARIABLE INDEX
# ============================================================

target_index = VARIABLES.index(
    TARGET_VARIABLE
)


# ============================================================
# WALK-FORWARD FORECASTING
# ============================================================

print("\n" + "=" * 90)
print("GENERATING WALK-FORWARD FORECASTS")
print("=" * 90)


total_test_origins = len(test)

processed = 0


for counter, (_, origin_row) in enumerate(
    test.iterrows(),
    start=1
):

    origin_date = origin_row[
        "date"
    ]

    target_date = origin_row[
        "target_date"
    ]

    segment_id = origin_row[
        "segment_id"
    ]


    # ========================================================
    # RETRIEVE CURRENT CONTINUOUS SEGMENT
    # ========================================================

    segment = df[
        df["segment_id"] == segment_id
    ].copy()

    segment = (
        segment
        .sort_values("date")
        .reset_index(drop=True)
    )


    # ========================================================
    # FIND CURRENT ORIGIN POSITION
    # ========================================================

    matching_positions = np.where(
        segment["date"].to_numpy()
        == np.datetime64(origin_date)
    )[0]


    if len(matching_positions) != 1:

        raise ValueError(
            f"Could not uniquely locate "
            f"forecast origin {origin_date}."
        )


    position = int(
        matching_positions[0]
    )


    # ========================================================
    # CHECK SUFFICIENT WITHIN-SEGMENT HISTORY
    # ========================================================
    #
    # BVAR(8) requires:
    #
    # t-1, t-2, ..., t-8
    #
    # We cannot borrow observations from before a long gap.
    # ========================================================

    if position < saved_lag_order:

        excluded_origins.append({

            "origin_date":
                origin_date,

            "segment_id":
                segment_id,

            "position_in_segment":
                position,

            "reason":
                "Insufficient within-segment history"
        })

        continue


    # ========================================================
    # EXTRACT HISTORY
    # ========================================================

    history = segment.iloc[
        position - saved_lag_order:
        position + 1
    ][VARIABLES].to_numpy(
        dtype=float
    )


    # ========================================================
    # CHECK HISTORY FOR MISSING VALUES
    # ========================================================

    if np.isnan(history).any():

        excluded_origins.append({

            "origin_date":
                origin_date,

            "segment_id":
                segment_id,

            "position_in_segment":
                position,

            "reason":
                "Missing values in lag history"
        })

        continue


    # ========================================================
    # STANDARDIZE USING TRAINING PARAMETERS
    # ========================================================

    history_scaled = (
        history - means
    ) / stds


    # ========================================================
    # BUILD VAR(8) FORECAST VECTOR
    # ========================================================
    #
    # Structure:
    #
    # [intercept,
    #  lag 1: 4 variables,
    #  lag 2: 4 variables,
    #  ...
    #  lag 8: 4 variables]
    #
    # The current row represents time t.
    # We forecast time t+1.
    # ========================================================

    x = [1.0]


    for lag in range(
        1,
        saved_lag_order + 1
    ):

        values_at_lag = (
            history_scaled[-lag]
        )

        x.extend(
            values_at_lag
        )


    x = np.asarray(
        x,
        dtype=float
    )


    # ========================================================
    # VERIFY DESIGN DIMENSION
    # ========================================================

    if len(x) != M:

        raise ValueError(
            f"Incorrect forecast vector length "
            f"at {origin_date}. "
            f"Expected {M}, got {len(x)}."
        )


    # ========================================================
    # POSTERIOR CONDITIONAL MEANS
    # ========================================================
    #
    # For every posterior draw:
    #
    # E(Y_t | parameters, X_t)
    #
    # is obtained from:
    #
    # X_t B
    # ========================================================

    conditional_means = np.einsum(
        "m,nm->n",
        x,
        posterior_B[
            :,
            :,
            target_index
        ]
    )


    # ========================================================
    # POSTERIOR CONDITIONAL VARIANCES
    # ========================================================

    conditional_variances = (
        posterior_Sigma[
            :,
            target_index,
            target_index
        ]
    )


    conditional_variances = np.maximum(
        conditional_variances,
        1e-12
    )


    conditional_sd = np.sqrt(
        conditional_variances
    )


    # ========================================================
    # TRANSFORM SPIKE THRESHOLD TO STANDARDIZED SCALE
    # ========================================================

    threshold_scaled = (
        SPIKE_THRESHOLD
        - means[target_index]
    ) / stds[target_index]


    # ========================================================
    # POSTERIOR PREDICTIVE SPIKE PROBABILITY
    # ========================================================
    #
    # For each posterior draw:
    #
    # Y ~ N(mu, sigma^2)
    #
    # Therefore:
    #
    # P(Y > c)
    # =
    # 1 - Phi((c - mu)/sigma)
    #
    # Then average the probabilities over all
    # posterior parameter draws.
    # ========================================================

    z_values = (
        threshold_scaled
        - conditional_means
    ) / conditional_sd


    draw_spike_probabilities = (
        1.0
        - norm.cdf(z_values)
    )


    posterior_spike_probability = (
        draw_spike_probabilities.mean()
    )


    # ========================================================
    # CONTINUOUS VOLATILITY FORECAST
    # ========================================================
    #
    # Posterior predictive mean on standardized scale:
    #
    # E(Y | data)
    #
    # The innovation has zero conditional mean.
    # ========================================================

    predicted_target_scaled = (
        conditional_means.mean()
    )


    # Transform back to original scale
    predicted_volatility = (
        predicted_target_scaled
        * stds[target_index]
        + means[target_index]
    )


    # Squared return cannot theoretically be negative.
    predicted_volatility = max(
        0.0,
        predicted_volatility
    )


    # ========================================================
    # BINARY SPIKE PREDICTION
    # ========================================================

    predicted_spike = int(
        posterior_spike_probability
        >= PROBABILITY_CUTOFF
    )


    # ========================================================
    # ACTUAL TARGET
    # ========================================================
    #
    # Step 09H stores the next-observation outcome on
    # the current origin row.
    # ========================================================

    actual_volatility = float(
        origin_row[
            "target_volatility"
        ]
    )

    actual_spike = int(
        origin_row[
            "target_spike"
        ]
    )


    # ========================================================
    # TARGET DATE CHECK
    # ========================================================

    if pd.isna(target_date):

        excluded_origins.append({

            "origin_date":
                origin_date,

            "segment_id":
                segment_id,

            "position_in_segment":
                position,

            "reason":
                "No valid target date"
        })

        continue


    if target_date <= origin_date:

        raise ValueError(
            f"Invalid target date at "
            f"{origin_date}: "
            f"{target_date}"
        )


    # ========================================================
    # SAVE FORECAST
    # ========================================================

    forecast_rows.append({

        "origin_date":
            origin_date,

        "target_date":
            target_date,

        "segment_id":
            segment_id,

        "actual_volatility":
            actual_volatility,

        "forecast_volatility":
            predicted_volatility,

        "actual_spike":
            actual_spike,

        "predicted_spike_probability":
            posterior_spike_probability,

        "predicted_spike":
            predicted_spike,

        "probability_cutoff":
            PROBABILITY_CUTOFF,

        "spike_threshold":
            SPIKE_THRESHOLD
    })


    processed += 1


    # ========================================================
    # PROGRESS DISPLAY
    # ========================================================

    if counter % 250 == 0:

        print(
            f"Processed origins: "
            f"{counter:,}/{total_test_origins:,} | "
            f"Forecasts retained: "
            f"{processed:,}"
        )


# ============================================================
# CREATE FORECAST DATAFRAME
# ============================================================

forecast_df = pd.DataFrame(
    forecast_rows
)


excluded_df = pd.DataFrame(
    excluded_origins
)


# ============================================================
# CHECK THAT FORECASTS WERE GENERATED
# ============================================================

if forecast_df.empty:

    raise ValueError(
        "No BVAR forecasts were generated."
    )


# ============================================================
# TARGET CONSISTENCY CHECK
# ============================================================

invalid_target_dates = (
    forecast_df["target_date"]
    <= forecast_df["origin_date"]
).sum()


print("\n" + "=" * 90)
print("TARGET CONSISTENCY CHECK")
print("=" * 90)

print(
    f"\nInvalid target dates: "
    f"{invalid_target_dates}"
)


if invalid_target_dates > 0:

    raise ValueError(
        "Some forecast targets are not "
        "after their forecast origins."
    )


# ============================================================
# CHECK DUPLICATE FORECAST ORIGINS
# ============================================================

duplicate_origins = (
    forecast_df["origin_date"]
    .duplicated()
    .sum()
)


print(
    f"Duplicate forecast origins: "
    f"{duplicate_origins}"
)


if duplicate_origins > 0:

    raise ValueError(
        "Duplicate forecast origins detected."
    )


# ============================================================
# SAVE FORECASTS
# ============================================================

forecast_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SAVE EXCLUDED ORIGINS
# ============================================================

excluded_df.to_csv(
    EXCLUDED_OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

actual_spikes = int(
    forecast_df[
        "actual_spike"
    ].sum()
)

predicted_spikes = int(
    forecast_df[
        "predicted_spike"
    ].sum()
)

mean_probability = (
    forecast_df[
        "predicted_spike_probability"
    ].mean()
)

actual_spike_rate = (
    actual_spikes
    / len(forecast_df)
)

predicted_spike_rate = (
    predicted_spikes
    / len(forecast_df)
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("BVAR(8) WALK-FORWARD SUMMARY")
print("=" * 90)

print(
    f"\nPotential forecast origins: "
    f"{total_test_origins:,}"
)

print(
    f"Forecasts retained: "
    f"{len(forecast_df):,}"
)

print(
    f"Origins excluded: "
    f"{len(excluded_df):,}"
)

print(
    f"\nActual spikes: "
    f"{actual_spikes:,}"
)

print(
    f"Actual spike rate: "
    f"{actual_spike_rate:.4%}"
)

print(
    f"Predicted spikes: "
    f"{predicted_spikes:,}"
)

print(
    f"Predicted spike rate: "
    f"{predicted_spike_rate:.4%}"
)

print(
    f"Mean predicted spike probability: "
    f"{mean_probability:.6f}"
)

print(
    f"\nForecast period: "
    f"{forecast_df['origin_date'].min().date()} "
    f"to "
    f"{forecast_df['target_date'].max().date()}"
)


# ============================================================
# EXCLUDED ORIGIN SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("EXCLUDED FORECAST ORIGINS")
print("=" * 90)

if len(excluded_df) == 0:

    print(
        "\nNo forecast origins were excluded."
    )

else:

    print(
        f"\nTotal excluded: "
        f"{len(excluded_df):,}"
    )

    print(
        "\nReasons:"
    )

    print(
        excluded_df[
            "reason"
        ].value_counts().to_string()
    )

    print(
        "\nExcluded observations:"
    )

    print(
        excluded_df.to_string(
            index=False
        )
    )


# ============================================================
# FIRST 10 FORECASTS
# ============================================================

print("\n" + "=" * 90)
print("FIRST 10 FORECASTS")
print("=" * 90)

print(
    forecast_df.head(10).to_string(
        index=False
    )
)


# ============================================================
# LAST 10 FORECASTS
# ============================================================

print("\n" + "=" * 90)
print("LAST 10 FORECASTS")
print("=" * 90)

print(
    forecast_df.tail(10).to_string(
        index=False
    )
)


# ============================================================
# RESULT FILES
# ============================================================

print("\n" + "=" * 90)
print("RESULTS SAVED")
print("=" * 90)

print(
    f"\nForecasts:"
    f"\n  {OUTPUT_FILE}"
)

print(
    f"\nExcluded origins:"
    f"\n  {EXCLUDED_OUTPUT_FILE}"
)

print("=" * 90)