import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold

# -----------------------
# CONFIG
# -----------------------

csv_path = "../../dataset2/TON_IoT_Consolidado_NaNs.csv"

output_prefix = "../../dataset/concatenado/"
output_train = "../../dataset/concatenado/2b3-"
output_test = "../../dataset/concatenado/1b3-"
output_dataset = "../../dataset/concatenado/"

os.makedirs(output_dataset, exist_ok=True)



# -----------------------
# LOAD CSV
# -----------------------

df = pd.read_csv(csv_path)

# remove espaços
df.columns = df.columns.str.strip()


# -----------------------
# TYPE -> ONE HOT
# -----------------------

df["Fridge_temp_condition"] = (
    df["Fridge_temp_condition"]
    .astype(str)
    .str.strip()
)

df["Fridge_temp_condition"] = df["Fridge_temp_condition"].map({
    "low": 0,
    "high": 1
})


# -----------------------
# DOOR STATE
# -----------------------

df["Garage_Door_door_state"] = (
    df["Garage_Door_door_state"]
    .astype(str)
    .str.strip()
)

df["Garage_Door_door_state"] = df["Garage_Door_door_state"].map({
    "closed": 0,
    "open": 1
})

df["Garage_Door_sphone_signal"] = (
    df["Garage_Door_sphone_signal"]
    .astype(str)
    .str.strip()
)

df["Garage_Door_sphone_signal"] = df["Garage_Door_sphone_signal"].map({
    "0": 0,
    "1": 1,
    "true": 1,
    "false": 0
})


df["Motion_Light_light_status"] = (
    df["Motion_Light_light_status"]
    .astype(str)
    .str.strip()
)

df["Motion_Light_light_status"] = df["Motion_Light_light_status"].map({
    "on": 0,
    "off": 1
})
# -----------------------
# FEATURES
# -----------------------

X_num = df[[
   "Fridge_fridge_temperature", 
           "Fridge_temp_condition", 
           "Garage_Door_sphone_signal",
             "Garage_Door_door_state", 
             "GPS_Tracker_latitude",
               "GPS_Tracker_longitude",
    "Modbus_FC1_Read_Input_Register",
    "Modbus_FC2_Read_Discrete_Value",
    "Modbus_FC3_Read_Holding_Register",
    "Modbus_FC4_Read_Coil",
    "Motion_Light_motion_status",
    "Motion_Light_light_status",
    "Thermostat_current_temperature",
    "Thermostat_thermostat_status",
    "Weather_temperature",
    "Weather_pressure",
    "Weather_humidity"
]]

X = X_num.fillna(0)

X = X.astype(float).values

# label
y = df["Global_Attack_Label"].astype(int).values

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
