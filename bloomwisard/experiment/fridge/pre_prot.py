import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ======================================================
# CONFIG
# ======================================================

csv_path = "../../dataset2/Train_Test_IoT_Fridge.csv"

output_train = "../../dataset/fridge/2b3-"
output_test = "../../dataset/fridge/1b3-"
output_dataset = "../../dataset/fridge/"

os.makedirs(output_dataset, exist_ok=True)

# ======================================================
# LOAD CSV
# ======================================================

df = pd.read_csv(csv_path)

# ======================================================
# CLEAN SPACES
# ======================================================

df.columns = df.columns.str.strip()

for c in df.columns:
    df[c] = df[c].astype(str).str.strip()

# ======================================================
# LABEL ENCODING
# ======================================================

df["temp_condition"] = df["temp_condition"].map({
    "low": 0,
    "high": 1
})

# ======================================================
# NUMERIC TEMPERATURE
# ======================================================

df["fridge_temperature"] = pd.to_numeric(
    df["fridge_temperature"],
    errors="coerce"
)

# ======================================================
# LABEL
# ======================================================

df["label"] = pd.to_numeric(
    df["label"],
    errors="coerce"
)

# ======================================================
# FEATURE ENGINEERING
# ======================================================

# ------------------------------------------------------
# HIGH DURATION
# ------------------------------------------------------

df["high_duration"] = 0

counter = 0

for i in range(len(df)):

    if df.loc[i, "temp_condition"] == 1:
        counter += 1
    else:
        counter = 0

    df.loc[i, "high_duration"] = counter

# ------------------------------------------------------
# LOW DURATION
# ------------------------------------------------------

df["low_duration"] = 0

counter = 0

for i in range(len(df)):

    if df.loc[i, "temp_condition"] == 0:
        counter += 1
    else:
        counter = 0

    df.loc[i, "low_duration"] = counter

# ------------------------------------------------------
# TEMPERATURE DIFFERENCE
# ------------------------------------------------------

df["temp_diff"] = (
    df["fridge_temperature"]
    .diff()
    .fillna(0)
)

# ------------------------------------------------------
# ABSOLUTE TEMPERATURE CHANGE
# ------------------------------------------------------

df["temp_abs_change"] = (
    df["temp_diff"]
    .abs()
)

# ------------------------------------------------------
# CONDITION CHANGES
# ------------------------------------------------------

df["temp_condition_change"] = (
    df["temp_condition"]
    .diff()
    .abs()
    .fillna(0)
)

# ------------------------------------------------------
# RECENT TOGGLES
# ------------------------------------------------------

df["temp_toggle_10"] = (
    df["temp_condition_change"]
    .rolling(10, min_periods=1)
    .sum()
)

# ------------------------------------------------------
# TEMPERATURE MOVING AVERAGE
# ------------------------------------------------------

df["temp_ma_10"] = (
    df["fridge_temperature"]
    .rolling(10, min_periods=1)
    .mean()
)

# ------------------------------------------------------
# TEMPERATURE VARIANCE
# ------------------------------------------------------

df["temp_var_10"] = (
    df["fridge_temperature"]
    .rolling(10, min_periods=1)
    .var()
    .fillna(0)
)

# ======================================================
# REMOVE UNUSED COLUMNS
# ======================================================

df = df.drop(
    columns=[
        "date",
        "time",
        "type"
    ]
)

# ======================================================
# REMOVE NAN
# ======================================================

df = df.fillna(0)

# ======================================================
# FEATURES
# ======================================================

X = df[[
    "fridge_temperature",
    "temp_condition",

    "high_duration",
    "low_duration",

    "temp_diff",
    "temp_abs_change",

    "temp_condition_change",
    "temp_toggle_10",

    "temp_ma_10",
    "temp_var_10"
]]

X = X.astype(float).values

# ======================================================
# LABELS
# ======================================================

y = df["label"].astype(int).values

# ======================================================
# INFO
# ======================================================

print("\n====================================")
print("DATASET INFO")
print("====================================")

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nClass distribution:")
print(np.bincount(y))

print("\nFirst rows:")
print(pd.DataFrame(X).head())

# ======================================================
# SAVE FUNCTION
# ======================================================

def save_split(prefix, X, y, tag):

    np.savetxt(
        f"{prefix}{tag}.data",
        X,
        delimiter=",",
        fmt="%.6f"
    )

    np.savetxt(
        f"{prefix}{tag}.label",
        y,
        fmt="%d"
    )

# ======================================================
# TRAIN / TEST SPLIT
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ======================================================
# SAVE DATASETS
# ======================================================

save_split(
    output_dataset,
    X,
    y,
    "dataset_full"
)

save_split(
    output_train,
    X_train,
    y_train,
    "train"
)

save_split(
    output_test,
    X_test,
    y_test,
    "test"
)

# ======================================================
# DONE
# ======================================================

print("\n====================================")
print("DATASET SAVED")
print("====================================")

print("Train:", X_train.shape)
print("Test :", X_test.shape)

print("\nFiles generated:")
print(" - dataset_full.data")
print(" - dataset_full.label")
print(" - train.data")
print(" - train.label")
print(" - test.data")
print(" - test.label")