import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold

# -----------------------
# CONFIG
# -----------------------

csv_path = "../../dataset2/Train_Test_IoT_Garage_Door.csv"

output_prefix = "../../dataset/garage_door/"
output_train = "../../dataset/garage_door/2b3-"
output_test = "../../dataset/garage_door/1b3-"
output_dataset = "../../dataset/garage_door/"

os.makedirs(output_dataset, exist_ok=True)

K_FOLDS = 3

# -----------------------
# LOAD CSV
# -----------------------

df = pd.read_csv(csv_path)

# remove espaços
df.columns = df.columns.str.strip()
# Remove espaços extras
df["date"] = (
    df["date"]
    .astype(str)
    .str.strip()
)

df["time"] = (
    df["time"]
    .astype(str)
    .str.strip()
)

# Converter data
df["date"] = pd.to_datetime(
    df["date"],
    format="%d-%b-%y"
)

# Converter hora
df["time"] = pd.to_datetime(
    df["time"],
    format="%H:%M:%S"
).dt.time

# Data
df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year

# Hora
df["hour"] = pd.to_datetime(df["time"], format="%H:%M:%S").dt.hour
df["minute"] = pd.to_datetime(df["time"], format="%H:%M:%S").dt.minute
df["second"] = pd.to_datetime(df["time"], format="%H:%M:%S").dt.second
# -----------------------
# DOOR STATE
# -----------------------

df["door_state"] = (
    df["door_state"]
    .astype(str)
    .str.strip()
)

df["door_state"] = df["door_state"].map({
    "closed": 0,
    "open": 1
})

df["sphone_signal"] = (
    df["sphone_signal"]
    .astype(str)
    .str.strip()
)

df["sphone_signal"] = df["sphone_signal"].map({
    "0": 0,
    "1": 1,
    "true": 1,
    "false": 0
})

# -----------------------
# TYPE -> ONE HOT
# -----------------------

type_onehot = pd.get_dummies(
    df["type"],
    prefix="type"
)

# -----------------------
# FEATURES
# -----------------------

X_num = df[[
    "sphone_signal", "door_state", "day", "month", "year", "hour", "minute", "second"
  
]]

X = X_num

X = X.astype(float).values

# label
y = df["label"].astype(int).values

print("Shape:", X.shape)

# -----------------------
# SAVE FUNCTION
# -----------------------

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
# TRAIN / TEST
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
save_split(output_dataset, X, y, "dataset_full")
save_split(output_train, X_train, y_train, "train")
save_split(output_test, X_test, y_test, "test")

print("✔ Dataset saved successfully.")
