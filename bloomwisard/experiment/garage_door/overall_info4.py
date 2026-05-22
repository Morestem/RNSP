import numpy as np
import pandas as pd
import sys

from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.append("../../")
from core import wnn


# =========================================================
# CONFIG
# =========================================================
base_path = "../../dataset/garage_door/"
dataset_x = base_path + "dataset_full.data"
dataset_y = base_path + "dataset_full.label"

tuple_bit = 1
num_classes = 2
sample_size = 1064


# =========================================================
# LOAD DATA
# =========================================================
df_x = pd.read_csv(
    dataset_x,
    header=None,
    names=["sphone_signal", "door_state"],
    dtype=float
)

df_y = pd.read_csv(
    dataset_y,
    header=None,
    names=["label"]
)

df = df_x.copy()
df["label"] = df_y["label"]


# =========================================================
# FEATURES TEMPORAIS
# =========================================================
df["sphone_signal_t1"] = df["sphone_signal"].shift(1)
df["door_state_t1"] = df["door_state"].shift(1)

df["sphone_signal_t2"] = df["sphone_signal"].shift(2)
df["door_state_t2"] = df["door_state"].shift(2)

df = df.dropna().reset_index(drop=True)


# =========================================================
# X / y
# =========================================================
X = df.drop(columns=["label"]).to_numpy()
y = df["label"].astype(int).to_numpy()

print("Shape X:", X.shape)
print("Shape y:", y.shape)

unique, counts = np.unique(y, return_counts=True)
dist = dict(zip(unique, counts))
print("Distribuição global:", dist)


# =========================================================
# BINARIZAÇÃO
# =========================================================
X_bin = (X > 0.5).astype(bool)


# =========================================================
# AMOSTRAGEM ALEATÓRIA (mistura classes)
# =========================================================
np.random.seed(42)

idx = np.random.choice(
    len(X_bin),
    size=sample_size,
    replace=False
)

X_small = X_bin[idx]
y_small = y[idx]

print("\n=== Subconjunto aleatório ===")
print("y_true:", y_small)

u_small, c_small = np.unique(y_small, return_counts=True)
print("Distribuição small:", dict(zip(u_small, c_small)))


# =========================================================
# MODELO
# =========================================================
entry_size = X_small.shape[1]

print("\nentry_size:", entry_size)
print("tuple_bit:", tuple_bit)

model = wnn.BloomWisard(
    entry_size,
    tuple_bit,
    num_classes,
    50000,
    error=0.5
)


# =========================================================
# TRAIN
# =========================================================
model.train(X_small, y_small)


# =========================================================
# TEST NO MESMO CONJUNTO
# =========================================================
y_pred = np.array(model.rank(X_small))

print("\n=== Predições ===")
print("y_pred:", y_pred)

acc = accuracy_score(y_small, y_pred)

print("\nAccuracy:", acc)

cm = confusion_matrix(
    y_small,
    y_pred,
    labels=[0, 1]
)

print("\nConfusion matrix:")
print(cm)


# =========================================================
# COMPARAÇÃO ITEM A ITEM
# =========================================================
print("\n=== Comparação ===")
for i in range(len(y_small)):
    print(
        f"{i:02d} | true={y_small[i]} pred={y_pred[i]}"
    )