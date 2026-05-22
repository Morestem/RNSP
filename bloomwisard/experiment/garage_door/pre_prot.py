import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

# Remove espaços das colunas
df.columns = df.columns.str.strip()

# Limpeza básica de strings
df["date"] = df["date"].astype(str).str.strip()
df["time"] = df["time"].astype(str).str.strip()

# Criar um timestamp completo combinando data e hora (crucial para cálculo de tempo)
df["timestamp"] = pd.to_datetime(
    df["date"] + " " + df["time"], format="%d-%b-%y %H:%M:%S"
)

# ORDENAR POR TEMPO: Garante que o cálculo de duração faça sentido sequencial
df = df.sort_values("timestamp").reset_index(drop=True)

# Extrair componentes de data e hora a partir do timestamp unificado
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month
df["year"] = df["timestamp"].dt.year
df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute
df["second"] = df["timestamp"].dt.second
df["time"] = df["timestamp"].dt.time

# -----------------------
# DOOR STATE
# -----------------------

df["door_state"] = df["door_state"].astype(str).str.strip()

df["door_state"] = df["door_state"].map({"closed": 0, "open": 1})

df["sphone_signal"] = df["sphone_signal"].astype(str).str.strip()

df["sphone_signal"] = df["sphone_signal"].map(
    {"0": 0, "1": 1, "true": 1, "false": 0}
)

# -----------------------
# CÁLCULO DE TEMPO DO ESTADO DA PORTA
# -----------------------

# 1. Calcula a diferença de tempo (em segundos) entre a linha atual e a anterior
df["time_diff"] = df["timestamp"].diff().dt.total_seconds().fillna(0)

# 2. Identifica quando houve mudança de estado (cria um ID de bloco incremental)
df["state_block"] = (df["door_state"] != df["door_state"].shift()).cumsum()

# 3. Calcula o tempo acumulado dentro do mesmo bloco de estado
df["duration_in_state"] = df.groupby("state_block")["time_diff"].cumsum()

# 4. Separa em duas colunas específicas (segundos aberta / segundos fechada)
df["seconds_open"] = np.where(df["door_state"] == 1, df["duration_in_state"], 0)
df["seconds_closed"] = np.where(
    df["door_state"] == 0, df["duration_in_state"], 0
)

# -----------------------
# TYPE -> ONE HOT
# -----------------------

type_onehot = pd.get_dummies(df["type"], prefix="type")

# -----------------------
# FEATURES
# -----------------------

# Incluídas as novas colunas temporais nas features do modelo
X_num = df[
    [
        "sphone_signal",
        "door_state",
        "seconds_open",
        "seconds_closed",
        "day",
        "month",
        "year",
        "hour",
        "minute",
        "second",
        
    ]
]

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
        f"{prefix}{tag}.data", X, delimiter=",", fmt="%.6f"
    )  # Aumentado para float devido aos segundos acumulados

    np.savetxt(f"{prefix}{tag}.label", y, fmt="%d")


# ======================================================
# TRAIN / TEST
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
save_split(output_dataset, X, y, "dataset_full")
save_split(output_train, X_train, y_train, "train")
save_split(output_test, X_test, y_test, "test")

print("✔ Dataset saved successfully.")