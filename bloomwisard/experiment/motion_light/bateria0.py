import numpy as np
import pandas as pd
import sys
from timeit import default_timer as timer

from sklearn.model_selection import StratifiedKFold
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

# =========================================================
# DATASET
# =========================================================

base_path = "../../dataset/motion_light/"

dataset_train_x = base_path + "2b3-train.data"
dataset_train_y = base_path + "2b3-train.label"

dataset_test_x = base_path + "1b3-test.data"
dataset_test_y = base_path + "1b3-test.label"

# =========================================================
# CONFIG
# =========================================================

WINDOW_SIZE = 64

tuple_list = [2, 4, 6, 8]
error_list = [0.01]

num_classes = 2
k = 4

# =========================================================
# LOAD TRAIN
# =========================================================

df_x = pd.read_csv(
    dataset_train_x,
    header=None,
    usecols=[0, 1],
    names=[
        "motion_status",
        "light_status"
    ],
    dtype=int
)

df_y = pd.read_csv(
    dataset_train_y,
    header=None
).iloc[:, 0].astype(int).to_numpy()

print("\nORIGINAL TRAIN")
print(df_x.head())

# =========================================================
# CREATE TEMPORAL WINDOW FEATURES
# =========================================================

def create_binary_window(df, window_size):

    cols = []

    for lag in range(window_size):

        shifted = df.shift(lag)

        shifted.columns = [
            f"{c}_t-{lag}"
            for c in df.columns
        ]

        cols.append(shifted)

    out = pd.concat(cols, axis=1)

    out = out.dropna()

    return out

# =========================================================
# BUILD TRAIN WINDOW
# =========================================================

df_x_window = create_binary_window(
    df_x,
    WINDOW_SIZE
)

# Align labels
df_y = df_y[WINDOW_SIZE - 1:]

# =========================================================
# FINAL TRAIN MATRICES
# =========================================================

X = df_x_window.to_numpy(dtype=bool)
y = df_y

print("\nWINDOWED TRAIN SHAPE")
print(X.shape)

print("\nENTRY SIZE")
print(X.shape[1])

# =========================================================
# LOAD TEST
# =========================================================

df_test_x = pd.read_csv(
    dataset_test_x,
    header=None,
    usecols=[0, 1],
    names=[
        "motion_status",
        "light_status"
    ],
    dtype=int
)

df_test_y = pd.read_csv(
    dataset_test_y,
    header=None
).iloc[:, 0].astype(int).to_numpy()

# =========================================================
# BUILD TEST WINDOW
# =========================================================

df_test_window = create_binary_window(
    df_test_x,
    WINDOW_SIZE
)

df_test_y = df_test_y[WINDOW_SIZE - 1:]

# =========================================================
# FINAL TEST MATRICES
# =========================================================

X_final_test = df_test_window.to_numpy(dtype=bool)
y_final_test = df_test_y

print("\nWINDOWED TEST SHAPE")
print(X_final_test.shape)

# =========================================================
# ENTRY SIZE
# =========================================================

entry_size = X.shape[1]

# =========================================================
# RESULTS
# =========================================================

all_results = []

# =========================================================
# KFOLD
# =========================================================

for tuple_bit in tuple_list:

    if tuple_bit > entry_size:
        continue

    for error in error_list:

        print("\n========================================")
        print(f"WINDOW_SIZE : {WINDOW_SIZE}")
        print(f"tuple_bit   : {tuple_bit}")
        print(f"error       : {error}")
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
            # MODEL
            # =============================================

            model = wnn.BloomWisard(
                entry_size,
                tuple_bit,
                num_classes,
                len(y_train),
                error=error
            )

            # =============================================
            # TRAIN
            # =============================================

            start = timer()

            model.train(
                X_train,
                y_train
            )

            train_time = timer() - start

            # =============================================
            # TEST
            # =============================================

            start = timer()

            y_pred = model.rank(X_test)

            test_time = timer() - start

            y_pred = np.array(y_pred)

            # =============================================
            # METRICS
            # =============================================

            acc = accuracy_score(
                y_test,
                y_pred
            )

            bal_acc = balanced_accuracy_score(
                y_test,
                y_pred
            )

            prec = precision_score(
                y_test,
                y_pred,
                zero_division=0
            )

            rec = recall_score(
                y_test,
                y_pred,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                y_pred,
                zero_division=0
            )

            fold_acc.append(acc)
            fold_bal_acc.append(bal_acc)
            fold_prec.append(prec)
            fold_rec.append(rec)
            fold_f1.append(f1)

            fold_train_time.append(train_time)
            fold_test_time.append(test_time)

            print(f"Accuracy : {acc:.4f}")
            print(f"Recall   : {rec:.4f}")
            print(f"F1-score : {f1:.4f}")

        # =================================================
        # SAVE RESULT
        # =================================================

        result = {

            "window_size": WINDOW_SIZE,
            "tuple_bit": tuple_bit,
            "error": error,

            "accuracy_mean": np.mean(fold_acc),
            "accuracy_std": np.std(fold_acc),

            "balanced_accuracy_mean": np.mean(fold_bal_acc),

            "precision_mean": np.mean(fold_prec),
            "recall_mean": np.mean(fold_rec),

            "f1_mean": np.mean(fold_f1),
            "f1_std": np.std(fold_f1),

            "train_time_mean": np.mean(fold_train_time),
            "test_time_mean": np.mean(fold_test_time),

            "entry_size": entry_size,
            "kfold": k

        }

        all_results.append(result)

        print("\nFINAL KFOLD RESULT")
        print(result)

# =========================================================
# SAVE KFOLD RESULTS
# =========================================================

df_results = pd.DataFrame(all_results)

df_results = df_results.sort_values(
    by="f1_mean",
    ascending=False
)

df_results.to_csv(
    "bloomwisard_kfold_results.csv",
    index=False
)

print("\n========================================")
print("TOP RESULTS")
print(df_results.head())

# =========================================================
# BEST CONFIG
# =========================================================

best = max(
    all_results,
    key=lambda x: x["f1_mean"]
)

best_tuple = best["tuple_bit"]
best_error = best["error"]

print("\nBEST CONFIG")
print(best)

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
    X,
    y
)

final_train_time = timer() - start

# =========================================================
# FINAL TEST
# =========================================================

start = timer()

y_pred = model.rank(X_final_test)

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

    "window_size": WINDOW_SIZE,
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