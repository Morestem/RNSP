import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

# =========================================================
# CONFIGURAÇÕES
# =========================================================
base_path = "../../dataset/garage_door/"
dataset_x = base_path + "dataset_full.data"
dataset_y = base_path + "dataset_full.label"

num_runs = 10
k        = 10
K_VIZINHOS = 5  # Número de vizinhos do KNN

# =========================================================
# CARREGAMENTO E JANELA TEMPORAL
# =========================================================
df_x = pd.read_csv(dataset_x, header=None,
                   names=["sphone_signal", "door_state"], dtype=float)
df_y = pd.read_csv(dataset_y, header=None)

# Janela temporal t-1 e t-2
df_x['sphone_signal_t-1'] = df_x['sphone_signal'].shift(1)
df_x['door_state_t-1']    = df_x['door_state'].shift(1)
df_x['sphone_signal_t-2'] = df_x['sphone_signal'].shift(2)
df_x['door_state_t-2']    = df_x['door_state'].shift(2)

df_x = df_x.dropna()
y = df_y.iloc[2:, 0].astype(int).to_numpy()
X = df_x.to_numpy()

# Distribuição das classes
unique, counts = np.unique(y, return_counts=True)
print(f"Distribuição das classes: {dict(zip(unique, counts))}")
print(f"Proporção classe 1: {counts[1]/len(y):.2%}\n")

# =========================================================
# CROSS VALIDATION
# =========================================================
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
results = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    y_train, y_test         = y[train_idx], y[test_idx]

    # Normalização sem data leakage
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    acc_f, prec_f, rec_f, f1_f = [], [], [], []

    for r in range(num_runs):
        model = KNeighborsClassifier(
            n_neighbors=K_VIZINHOS,
            metric='euclidean',
            weights='distance'  # vizinhos mais próximos têm mais peso
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc_f.append(accuracy_score(y_test, y_pred))
        prec_f.append(precision_score(y_test, y_pred, zero_division=0))
        rec_f.append(recall_score(y_test, y_pred, zero_division=0))
        f1_f.append(f1_score(y_test, y_pred, zero_division=0))

    results.append({
        "fold":      fold + 1,
        "accuracy":  np.mean(acc_f),
        "precision": np.mean(prec_f),
        "recall":    np.mean(rec_f),
        "f1":        np.mean(f1_f)
    })
    print(f"Fold {fold+1}/{k} → Acc: {results[-1]['accuracy']:.3f} | "
          f"F1: {results[-1]['f1']:.3f}")

# =========================================================
# RESULTADOS
# =========================================================
df_results = pd.DataFrame(results)
print("\n=== Resultados Consolidados ===")
print(df_results.to_string(index=False))
print("\n=== Média Geral ===")
print(df_results[["accuracy","precision","recall","f1"]].mean().round(4))

# =========================================================
# BUSCA PELO MELHOR K (opcional)
# =========================================================
print("\n=== Testando diferentes valores de K ===")
best_k, best_f1 = 1, 0

for k_test in [1, 3, 5, 7, 9, 11, 15]:
    f1_scores = []
    for train_idx, test_idx in kf.split(X, y):
        scaler = MinMaxScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])

        m = KNeighborsClassifier(n_neighbors=k_test, weights='distance')
        m.fit(X_tr, y[train_idx])
        f1_scores.append(f1_score(y[test_idx], m.predict(X_te), zero_division=0))

    mean_f1 = np.mean(f1_scores)
    print(f"  K={k_test:2d} → F1 médio: {mean_f1:.4f}")
    if mean_f1 > best_f1:
        best_f1, best_k = mean_f1, k_test

print(f"\nMelhor K encontrado: {best_k} (F1={best_f1:.4f})")