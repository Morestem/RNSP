import numpy as np
import pandas as pd
import sys
from timeit import default_timer as timer

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix
)

sys.path.append("../../")

from core import wnn
from encoding import thermometer

# =========================================================
# DATASET
# =========================================================

base_path = "../../dataset/gps/"

dataset_train_x = base_path + "2b3-train.data"
dataset_train_y = base_path + "2b3-train.label"

dataset_test_x = base_path + "1b3-test.data"
dataset_test_y = base_path + "1b3-test.label"

# =========================================================
# CONFIG
# =========================================================

bits_list = [64, 128, 256]
tuple_list = [2, 4, 8, 16, 24, 32]
error_list = [0.01]

num_classes = 2
num_runs = 10
k = 4

# =========================================================
# LOAD TRAIN
# =========================================================

df_x = pd.read_csv(
    dataset_train_x,
    header=None,
    names=["latitude", "longitude", "day", "month", "year", "hour", "minute", "second"],
    dtype=float
)

df_y = pd.read_csv(
    dataset_train_y,
    header=None
).iloc[:, 0].astype(int).to_numpy()

# =========================================================
# FEATURE ENGINEERING TRAIN
# =========================================================


df_x = df_x.fillna(0)

X = df_x.to_numpy(dtype=np.float32)
y = df_y

print("Train shape:", X.shape)

# =========================================================
# LOAD FINAL TEST
# =========================================================

df_test_x = pd.read_csv(
    dataset_test_x,
    header=None,
    names=["latitude", "longitude", "day", "month", "year", "hour", "minute", "second"],
    dtype=float
)

df_test_y = pd.read_csv(
    dataset_test_y,
    header=None
).iloc[:, 0].astype(int).to_numpy()

# =========================================================
# FEATURE ENGINEERING TEST
# =========================================================

df_test_x = df_test_x.fillna(0)

X_final_test = df_test_x.to_numpy(dtype=np.float32)
y_final_test = df_test_y

print("Final test shape:", X_final_test.shape)

# =========================================================
# ENCODING
# =========================================================

def encode(data, ths):

    out = []

    for sample in data:

        vec = np.array([], dtype=bool)

        for i, v in enumerate(sample):

            vec = np.append(
                vec,
                ths[i].binarize(v)
            )

        out.append(vec)

    return np.array(out)

# =========================================================
# RESULTS
# =========================================================

all_results = []

# =========================================================
# KFOLD BATTERY
# =========================================================

for bits_encoding in bits_list:

    for tuple_bit in tuple_list:

        estimated_entry_size = X.shape[1] * bits_encoding

        if tuple_bit > estimated_entry_size:
            continue

        for error in error_list:

            print("\n========================================")
            print(f"bits_encoding : {bits_encoding}")
            print(f"tuple_bit     : {tuple_bit}")
            print(f"error         : {error}")
            print("========================================")

            skf = StratifiedKFold(
                n_splits=k,
                shuffle=True,
                random_state=42
            )

            fold_acc = []
            fold_bal_acc = []
            fold_prec = []
            fold_rec = []
            fold_f1 = []

            fold_train_time = []
            fold_test_time = []

            # =================================================
            # KFOLD LOOP
            # =================================================

            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):

                print(f"\nFold {fold+1}/{k}")

                X_train = X[train_idx]
                X_test = X[test_idx]

                y_train = y[train_idx]
                y_test = y[test_idx]

                # =============================================
                # NORMALIZATION
                # =============================================

                scaler = MinMaxScaler()

                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # =============================================
                # THERMOMETER
                # =============================================

                data_min = np.min(X_train, axis=0)
                data_max = np.max(X_train, axis=0)

                ths = []

                for i in range(X_train.shape[1]):

                    mn = data_min[i]
                    mx = data_max[i]

                    if mn == mx:
                        mx += 1e-9

                    ths.append(
                        thermometer.Thermometer(
                            mn,
                            mx,
                            bits_encoding
                        )
                    )

                X_train_bin = encode(X_train, ths)
                X_test_bin = encode(X_test, ths)

                entry_size = len(X_train_bin[0])

                print(f"Entry size: {entry_size}")

                # =============================================
                # RUNS
                # =============================================

                acc_runs = []
                bal_acc_runs = []
                prec_runs = []
                rec_runs = []
                f1_runs = []

                train_runs = []
                test_runs = []

                for run in range(num_runs):

                    print(f"Run {run+1}/{num_runs}")

                    model = wnn.BloomWisard(
                        entry_size,
                        tuple_bit,
                        num_classes,
                        len(y_train),
                        error=error
                    )

                    # =========================================
                    # TRAIN
                    # =========================================

                    start = timer()

                    model.train(
                        X_train_bin,
                        y_train
                    )

                    train_time = timer() - start

                    # =========================================
                    # TEST
                    # =========================================

                    start = timer()

                    y_pred = model.rank(X_test_bin)

                    test_time = timer() - start

                    y_pred = np.array(y_pred)

                    # =========================================
                    # METRICS
                    # =========================================

                    acc_runs.append(
                        accuracy_score(y_test, y_pred)
                    )

                    bal_acc_runs.append(
                        balanced_accuracy_score(
                            y_test,
                            y_pred
                        )
                    )

                    prec_runs.append(
                        precision_score(
                            y_test,
                            y_pred,
                            zero_division=0
                        )
                    )

                    rec_runs.append(
                        recall_score(
                            y_test,
                            y_pred,
                            zero_division=0
                        )
                    )

                    f1_runs.append(
                        f1_score(
                            y_test,
                            y_pred,
                            zero_division=0
                        )
                    )

                    train_runs.append(train_time)
                    test_runs.append(test_time)

                # =============================================
                # FOLD RESULTS
                # =============================================

                fold_acc.append(np.mean(acc_runs))
                fold_bal_acc.append(np.mean(bal_acc_runs))
                fold_prec.append(np.mean(prec_runs))
                fold_rec.append(np.mean(rec_runs))
                fold_f1.append(np.mean(f1_runs))

                fold_train_time.append(np.mean(train_runs))
                fold_test_time.append(np.mean(test_runs))

                print(f"Fold Accuracy : {np.mean(acc_runs):.4f}")
                print(f"Fold F1       : {np.mean(f1_runs):.4f}")

            # =================================================
            # SAVE CONFIG RESULT
            # =================================================

            result = {

                "bits_encoding": bits_encoding,
                "tuple_bit": tuple_bit,
                "error": error,

                "accuracy_mean": np.mean(fold_acc),
                "accuracy_std": np.std(fold_acc),

                "balanced_accuracy_mean": np.mean(fold_bal_acc),

                "precision_mean": np.mean(fold_prec),
                "precision_std": np.std(fold_prec),

                "recall_mean": np.mean(fold_rec),
                "recall_std": np.std(fold_rec),

                "f1_mean": np.mean(fold_f1),
                "f1_std": np.std(fold_f1),

                "train_time_mean": np.mean(fold_train_time),
                "test_time_mean": np.mean(fold_test_time),

                "entry_size": entry_size,
                "kfold": k,
                "runs": num_runs

            }

            all_results.append(result)

            print("\nFINAL KFOLD RESULT")
            print(result)

# =========================================================
# SAVE KFOLD CSV
# =========================================================

df_results = pd.DataFrame(all_results)

df_results = df_results.sort_values(
    by="accuracy_mean",
    ascending=False
)

df_results.to_csv(
    "bloomwisard_kfold_results.csv",
    index=False
)

print("\n========================================")
print("KFOLD RESULTS")
print(df_results.head())

# =========================================================
# BEST CONFIG
# =========================================================

best = max(
    all_results,
    key=lambda x: x["accuracy_mean"]
)

best_bits = best["bits_encoding"]
best_tuple = best["tuple_bit"]
best_error = best["error"]

print("\n========================================")
print("BEST CONFIG")
print(best)

# =========================================================
# FINAL TRAIN
# =========================================================

scaler = MinMaxScaler()

X_train_full = scaler.fit_transform(X)
X_test_final = scaler.transform(X_final_test)

data_min = np.min(X_train_full, axis=0)
data_max = np.max(X_train_full, axis=0)

ths = []

for i in range(X_train_full.shape[1]):

    mn = data_min[i]
    mx = data_max[i]

    if mn == mx:
        mx += 1e-9

    ths.append(
        thermometer.Thermometer(
            mn,
            mx,
            best_bits
        )
    )

X_train_bin = encode(X_train_full, ths)
X_test_bin = encode(X_test_final, ths)

entry_size = len(X_train_bin[0])

# =========================================================
# FINAL MODEL
# =========================================================

model = wnn.BloomWisard(
    entry_size,
    best_tuple,
    num_classes,
    len(y),
    error=best_error
)

# =========================================================
# FINAL TRAIN
# =========================================================

start = timer()

model.train(
    X_train_bin,
    y
)

final_train_time = timer() - start

# =========================================================
# FINAL TEST
# =========================================================

start = timer()

y_pred = model.rank(X_test_bin)

final_test_time = timer() - start

y_pred = np.array(y_pred)

# =========================================================
# FINAL METRICS
# =========================================================

final_accuracy = accuracy_score(
    y_final_test,
    y_pred
)

final_balanced_accuracy = balanced_accuracy_score(
    y_final_test,
    y_pred
)

final_precision = precision_score(
    y_final_test,
    y_pred,
    zero_division=0
)

final_recall = recall_score(
    y_final_test,
    y_pred,
    zero_division=0
)

final_f1 = f1_score(
    y_final_test,
    y_pred,
    zero_division=0
)

final_cm = confusion_matrix(
    y_final_test,
    y_pred
)

# =========================================================
# FINAL RESULTS
# =========================================================

print("\n========================================")
print("FINAL TEST RESULTS")
print("========================================")

print(f"Accuracy          : {final_accuracy:.6f}")
print(f"Balanced Accuracy : {final_balanced_accuracy:.6f}")
print(f"Precision         : {final_precision:.6f}")
print(f"Recall            : {final_recall:.6f}")
print(f"F1-score          : {final_f1:.6f}")

print(f"Train Time        : {final_train_time:.6f} s")
print(f"Test Time         : {final_test_time:.6f} s")

print("\nConfusion Matrix")
print(final_cm)

# =========================================================
# SAVE FINAL RESULTS
# =========================================================

df_final = pd.DataFrame([{

    "bits_encoding": best_bits,
    "tuple_bit": best_tuple,
    "error": best_error,

    "accuracy": final_accuracy,
    "balanced_accuracy": final_balanced_accuracy,
    "precision": final_precision,
    "recall": final_recall,
    "f1_score": final_f1,

    "train_time": final_train_time,
    "test_time": final_test_time

}])

df_final.to_csv(
    "bloomwisard_final_test_results.csv",
    index=False
)

print("\nCSV salvo:")
print(" - bloomwisard_kfold_results.csv")
print(" - bloomwisard_final_test_results.csv")