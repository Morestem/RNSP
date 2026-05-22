import numpy as np
import pandas as pd
import sys
from timeit import default_timer as timer
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

sys.path.append("../../")

from core import wnn
from encoding import thermometer

# =========================================================
# CONFIG
# =========================================================

base_path = "../../dataset/weather/"

dataset_x = base_path + "dataset_full.data"
dataset_y = base_path + "dataset_full.label"


bits_encoding = 512   # Resolução alta para temperatura
tuple_bit = 32       # TUPLA MENOR = MAIOR RECALL (128/16 = 8 RAMs) bits_encoding*colunas/tuple_bit = num_rams
num_classes = 2
num_runs = 10          # runs por fold
k = 10                 # 10-fold CV

# =========================================================
# LOAD DATA
# =========================================================

columns = [
    "temperature",
    "pressure",
    "humidity"
]

df_x = pd.read_csv(
    dataset_x,
    header=None,
    names=columns,
    dtype=float
)
#print(f"X shape: {df_x.shape}")
#print(f"X head:\n{df_x.head(20)}")

df_x['pressure_diff'] = df_x['pressure'].diff().fillna(0)
df_x['humidity_diff'] = df_x['humidity'].diff().fillna(0)

df_y = pd.read_csv(
    dataset_y,
    header=None
).iloc[:, 0].astype(int).to_numpy()

X = df_x.to_numpy()
y = df_y

print(df_x.isna().sum())


# =========================================================
# ENCODING
# =========================================================

def encode(data, ths):
    out = []
    for sample in data:
        vec = np.array([], dtype=bool)
        for i, v in enumerate(sample):
            vec = np.append(vec, ths[i].binarize(v))
        out.append(vec)
    return out

# =========================================================
# CROSS VALIDATION
# =========================================================

kf = KFold(n_splits=k, shuffle=True, random_state=42)

results = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # termômetro por fold
    data_min = np.min(X_train, axis=0)
    data_max = np.max(X_train, axis=0)

    ths = []

    for i in range(X_train.shape[1]):
        mn = data_min[i]
        mx = data_max[i]

        # evita intervalo zero
        if mn == mx:
            mx = mn + 1e-9

        ths.append(
            thermometer.Thermometer(mn, mx, bits_encoding)
        )

    
    X_train_bin = encode(X_train, ths)
    X_test_bin = encode(X_test, ths)

    entry_size = len(X_train_bin[0])
    
    print(f"\nFold {fold+1}/{k} - Entry size: {entry_size} bits")

    acc_f, prec_f, rec_f, f1_f = [], [], [], []
    train_t_f, test_t_f = [], []

    for r in range(num_runs):

        model = wnn.BloomWisard(
            entry_size,
            tuple_bit,
            num_classes,
            len(y_train),
            error=0.01
        )

        # TRAIN
        start = timer()
        model.train(X_train_bin, y_train)
        train_t_f.append(timer() - start)

        # TEST
        start = timer()
        y_pred = model.rank(X_test_bin)
        test_t_f.append(timer() - start)

        # metrics
        acc_f.append(accuracy_score(y_test, y_pred))
        prec_f.append(precision_score(y_test, y_pred, zero_division=0))
        rec_f.append(recall_score(y_test, y_pred, zero_division=0))
        f1_f.append(f1_score(y_test, y_pred, zero_division=0))

    results.append({
        "fold": fold,
        "accuracy": np.mean(acc_f),
        "precision": np.mean(prec_f),
        "recall": np.mean(rec_f),
        "f1": np.mean(f1_f),
        "train_time": np.mean(train_t_f),
        "test_time": np.mean(test_t_f),
    })

# =========================================================
# CSV OUTPUT
# =========================================================

df_results = pd.DataFrame(results)

df_results.to_csv("bloom_wisard_10fold_metrics.csv", index=False)

print(df_results)

print("\nCSV salvo: bloom_wisard_10fold_metrics.csv")