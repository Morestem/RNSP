import os
import pandas as pd

# =========================================================
# PASTAS
# =========================================================

folders = [
    "fridge/",
    "garage_door/",
    "gps_tracker/",
    "modbus/",
    "motion_light/",
    "thermostat/",
    "weather/"
]

OUTPUT_FILE = "resultado_final.csv"

# =========================================================
# COLUNAS DESEJADAS
# =========================================================

desired_columns = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "train_time",
    "test_time"
]

dfs = []

# =========================================================
# LEITURA
# =========================================================

for folder in folders:
    try:
                # Lê CSV
        df = pd.read_csv(folder+"bloomwisard_final_test_results.csv")

                # Mantém apenas colunas desejadas
        df = df[desired_columns]

                # Adiciona nome do dataset na primeira coluna
        df.insert(0, "Dataset", folder[:-1])  # Remove a barra do final

        dfs.append(df)

        print(f"[OK] {folder}")

    except Exception as e:
                print(f"[ERRO] {folder} -> {e}")

# =========================================================
# CONCATENA
# =========================================================

if len(dfs) > 0:

    final_df = pd.concat(dfs, ignore_index=True)

    # Salva CSV final
    final_df.to_csv(OUTPUT_FILE, index=False)

    print("\nArquivo salvo:")
    print(OUTPUT_FILE)

    print("\nPreview:")
    print(final_df.head(7))

else:
    print("Nenhum CSV encontrado.")