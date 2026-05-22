import numpy as np
import pandas as pd
import sys
from timeit import default_timer as timer

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

sys.path.append("../../")

from core import wnn
from encoding import thermometer

# =========================================================
# DATASET
# =========================================================

base_path = "../../dataset/concatenado/"

dataset_x = base_path + "dataset_full.data"
dataset_y = base_path + "dataset_full.label"

# =========================================================
# BATTERY CONFIG
# =========================================================

tuple_list = [4, 8, 16]
bits_list = [256]
error_list = [0.01]

num_classes = 2
num_runs = 10
k = 10

# =========================================================
# LOAD DATA
# =========================================================

df_x = pd.read_csv(
    dataset_x,
    header=None,
    names=[ "fridge_temperature", 
           "temp_condition", 
           "sphone_signal",
             "door_state", 
             "latitude",
               "longitude"
    "FC1_Read_Input_Register",
    "FC2_Read_Discrete_Value",
    "FC3_Read_Holding_Register",
    "FC4_Read_Coil",
    "motion_status",
    "light_status",
    "current_temperature",
    "thermostat_status",
    "temperature",
    "pressure",
    "humidity"],
    dtype=float
)

df_x = df_x.fillna(0)

df_y = pd.read_csv(
    dataset_y,
    header=None
).iloc[:, 0].astype(int).to_numpy()

# =========================================================
# FEATURE ENGINEERING
# =========================================================

df_x['temp_sphone_signal'] = (
    df_x['sphone_signal']
    .rolling(500)
    .mean()
    .fillna(0)
)

df_x['temp_door_state'] = (
    df_x['door_state']
    .rolling(500)
    .var()
    .fillna(0)
)

df_x['temp1_sphone_signal'] = (
    df_x['sphone_signal']
    .rolling(100)
    .mean()
    .fillna(0)
)

df_x['temp1_door_state'] = (
    df_x['door_state']
    .rolling(100)
    .var()
    .fillna(0)
)

X = df_x.to_numpy()
y = df_y

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

    return out

# =========================================================
# RESULTS
# =========================================================

all_results = []

# =========================================================
# BATTERY TESTS
# =========================================================

for bits_encoding in bits_list:

    for tuple_bit in tuple_list:

        for error in error_list:

            print("\n====================================")
            print(f"bits_encoding : {bits_encoding}")
            print(f"tuple_bit     : {tuple_bit}")
            print(f"error         : {error}")
            print("====================================")

            kf = KFold(
                n_splits=k,
                shuffle=True,
                random_state=42
            )

            fold_acc = []
            fold_prec = []
            fold_rec = []
            fold_f1 = []

            fold_train_time = []
            fold_test_time = []

            for fold, (train_idx, test_idx) in enumerate(kf.split(X)):

                print(f"\nFold {fold+1}/{k}")

                X_train = X[train_idx]
                X_test = X[test_idx]

                y_train = y[train_idx]
                y_test = y[test_idx]

                # =================================================
                # NORMALIZATION
                # =================================================

                scaler = MinMaxScaler()

                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # =================================================
                # THERMOMETER ENCODING
                # =================================================

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

                # =================================================
                # RUNS
                # =================================================

                acc_runs = []
                prec_runs = []
                rec_runs = []
                f1_runs = []

                train_runs = []
                test_runs = []

                for run in range(num_runs):

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
                        X_train_bin,
                        y_train
                    )

                    train_time = timer() - start

                    # =============================================
                    # TEST
                    # =============================================

                    start = timer()

                    y_pred = model.rank(X_test_bin)

                    test_time = timer() - start

                    # =============================================
                    # METRICS
                    # =============================================

                    acc_runs.append(
                        accuracy_score(y_test, y_pred)
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
                # FOLD AVERAGE
                # =============================================

                fold_acc.append(np.mean(acc_runs))
                fold_prec.append(np.mean(prec_runs))
                fold_rec.append(np.mean(rec_runs))
                fold_f1.append(np.mean(f1_runs))

                fold_train_time.append(np.mean(train_runs))
                fold_test_time.append(np.mean(test_runs))

            # =====================================================
            # FINAL CONFIG RESULTS
            # =====================================================

            all_results.append({

                "bits_encoding": bits_encoding,
                "tuple_bit": tuple_bit,
                "error": error,

                "accuracy_mean": np.mean(fold_acc),
                "accuracy_std": np.std(fold_acc),

                "precision_mean": np.mean(fold_prec),
                "recall_mean": np.mean(fold_rec),
                "f1_mean": np.mean(fold_f1),

                "train_time_mean": np.mean(fold_train_time),
                "test_time_mean": np.mean(fold_test_time)

            })

# =========================================================
# SAVE CSV
# =========================================================

df_results = pd.DataFrame(all_results)

df_results.to_csv(
    "bloomwisard_battery_results.csv",
    index=False
)

print("\n====================================")
print(df_results)
print("\nCSV salvo: bloomwisard_battery_results.csv")