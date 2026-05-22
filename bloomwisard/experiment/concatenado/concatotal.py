import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

# ======================================================
# CONFIG
# ======================================================

output_train = "../../dataset/concatenado/2b3-"
output_test = "../../dataset/concatenado/1b3-"
output_dataset = "../../dataset/concatenado/"

os.makedirs(output_dataset, exist_ok=True)

# ======================================================
# LOAD DATASETS
# ======================================================

df0 = pd.read_csv("../../dataset/fridge/dataset_full.data", header=None)
df1 = pd.read_csv("../../dataset/garage_door/dataset_full.data", header=None)
df2 = pd.read_csv("../../dataset/gps/dataset_full.data", header=None)
df3 = pd.read_csv("../../dataset/modbus/dataset_full.data", header=None)
df4 = pd.read_csv("../../dataset/motion_light/dataset_full.data", header=None)
df5 = pd.read_csv("../../dataset/thermostat/dataset_full.data", header=None)
df6 = pd.read_csv("../../dataset/weather/dataset_full.data", header=None)

# ======================================================
# REMOVE ÚLTIMAS 6 COLUNAS
# ======================================================

df0 = df0.iloc[:, :-6]
df1 = df1.iloc[:, :-6]
df2 = df2.iloc[:, :-6]
df3 = df3.iloc[:, :-6]
df4 = df4.iloc[:, :-6]
df5 = df5.iloc[:, :-6]
df6 = df6.iloc[:, :-6]

# ======================================================
# ORGANIZA COLUNAS
# Cada dataset ocupa um espaço diferente
# ======================================================

dfs = [df0, df1, df2, df3, df4, df5, df6]

novo_dfs = []

inicio = 0

for df in dfs:

    n_cols = df.shape[1]

    # renomeia colunas
    df.columns = range(inicio, inicio + n_cols)

    inicio += n_cols

    novo_dfs.append(df)

# quantidade total de colunas
total_cols = inicio

# ======================================================
# PREENCHE COLUNAS FALTANTES COM ZERO
# ======================================================

for i in range(len(novo_dfs)):

    novo_dfs[i] = novo_dfs[i].reindex(
        columns=range(total_cols),
        fill_value=0
    )

# ======================================================
# CONCATENA AS LINHAS
# ======================================================

df_xfinal = pd.concat(novo_dfs, ignore_index=True)

df_xfinal = df_xfinal.fillna(0)

# ======================================================
# SALVA FEATURES
# ======================================================

df_xfinal.to_csv(
    "../../dataset/concatenado/dataset_full.data",
    index=False,
    header=False
)

# ======================================================
# LABELS
# ======================================================

df0 = pd.read_csv("../../dataset/fridge/dataset_full.label", header=None)
df1 = pd.read_csv("../../dataset/garage_door/dataset_full.label", header=None)
df2 = pd.read_csv("../../dataset/gps/dataset_full.label", header=None)
df3 = pd.read_csv("../../dataset/modbus/dataset_full.label", header=None)
df4 = pd.read_csv("../../dataset/motion_light/dataset_full.label", header=None)
df5 = pd.read_csv("../../dataset/thermostat/dataset_full.label", header=None)
df6 = pd.read_csv("../../dataset/weather/dataset_full.label", header=None)

# concatena labels
df_yfinal = pd.concat(
    [df0, df1, df2, df3, df4, df5, df6],
    ignore_index=True
)

df_yfinal = df_yfinal.fillna(0)

# salva labels
df_yfinal.to_csv(
    "../../dataset/concatenado/dataset_full.label",
    index=False,
    header=False
)

# ======================================================
# TRAIN / TEST SPLIT
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

X_train, X_test, y_train, y_test = train_test_split(
    df_xfinal,
    df_yfinal,
    test_size=0.2,
    random_state=42,
    stratify=df_yfinal
)

# ======================================================
# SAVE DATASETS
# ======================================================

save_split(
    output_dataset,
    df_xfinal,
    df_yfinal,
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