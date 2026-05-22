import argparse
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

try:
    import wisardpkg as wp
except ImportError as exc:
    raise ImportError("wisardpkg is required to run this script. Install it in your environment.") from exc


# --- CLASSE PARA SALVAR O LOG NO TXT E EXIBIR NO CONSOLE ---
class Logger:
    def __init__(self, filename="resultados.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Garante que o conteúdo seja escrito imediatamente

    def flush(self):
        self.terminal.flush()
        self.log.flush()
# -----------------------------------------------------------


DEFAULT_DATA_FILE = os.path.join(
    "ToN_IoT",
    "Processed_datasets",
    "Processed_IoT_dataset"
)

DATASET_NAMES = [
    "IoT_Fridge.csv", 
    "IoT_Garage_Door.csv", 
    "IoT_GPS_Tracker.csv", 
    "IoT_Motion_Light.csv", 
    "IoT_Modbus.csv", 
    "IoT_Thermostat.csv"
]

EXCLUDE_COLUMNS = ["ts", "date", "time", "label", "type"]


def load_dataset(filepath: str) -> pd.DataFrame:
    print(f"Loading dataset from: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    df = pd.read_csv(filepath, low_memory=False)
    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")
    df["label"] = df["label"].astype(str)
    return df


def pipeline_especifico_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    cols = set(df.columns)
    df_clean = df.copy()
    
    # Resolvido o Pandas4Warning explicitando os tipos textuais
    for col in df_clean.select_dtypes(include=['object', 'string', 'category']).columns:
        df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()

    # 1. Tabela 2: IoT Fridge
    if "fridge_temperature" in cols and "temp_condition" in cols:
        print("Dataset identificado: IoT Fridge (Tabela 2)")
        feature_cols = ["fridge_temperature", "temp_condition"]
        df_clean["fridge_temperature"] = pd.to_numeric(df_clean["fridge_temperature"], errors="coerce")
        df_clean["temp_condition"] = df_clean["temp_condition"].map({"high": 1.0, "low": 0.0})

    # 2. Tabela 3: IoT Garage Door
    elif "door_state" in cols and "sphone_signal" in cols:
        print("Dataset identificado: IoT Garage Door (Tabela 3)")
        feature_cols = ["door_state", "sphone_signal"]
        df_clean["door_state"] = df_clean["door_state"].map({"open": 1.0, "closed": 0.0})
        df_clean["sphone_signal"] = df_clean["sphone_signal"].map({"true": 1.0, "false": 0.0})

    # 3. Tabela 4: IoT GPS Tracker
    elif "latitude" in cols and "longitude" in cols:
        print("Dataset identificado: IoT GPS Tracker (Tabela 4)")
        feature_cols = ["latitude", "longitude"]
        df_clean["latitude"] = pd.to_numeric(df_clean["latitude"], errors="coerce")
        df_clean["longitude"] = pd.to_numeric(df_clean["longitude"], errors="coerce")

    # 4. Tabela 5: IoT Motion Light
    elif "motion_status" in cols and "light_status" in cols:
        print("Dataset identificado: IoT Motion Light (Tabela 5)")
        feature_cols = ["motion_status", "light_status"]
        df_clean["motion_status"] = pd.to_numeric(df_clean["motion_status"], errors="coerce")
        df_clean["light_status"] = df_clean["light_status"].map({"on": 1.0, "off": 0.0})

    # 5. Tabela 6: IoT Modbus
    elif any(c.startswith("FC1_") or c.startswith("FC2_") for c in cols):
        print("Dataset identificado: IoT Modbus (Tabela 6)")
        feature_cols = [c for c in df.columns if c.startswith("FC")]
        for c in feature_cols:
            df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")

    # 6. Tabela 7: IoT Thermostat
    elif "current_temperature" in cols and "thermostat_status" in cols:
        print("Dataset identificado: IoT Thermostat (Tabela 7)")
        feature_cols = ["current_temperature", "thermostat_status"]
        df_clean["current_temperature"] = pd.to_numeric(df_clean["current_temperature"], errors="coerce")
        # Trata caso venha como "on"/"off" ou já venha como 1/0
        if df_clean["thermostat_status"].astype(str).str.isalpha().any():
            df_clean["thermostat_status"] = df_clean["thermostat_status"].map({"on": 1.0, "off": 0.0})
        else:
            df_clean["thermostat_status"] = pd.to_numeric(df_clean["thermostat_status"], errors="coerce")

    # 7. Tabela 8: IoT Weather
    elif "pressure" in cols and "humidity" in cols:
        print("Dataset identificado: IoT Weather (Tabela 8)")
        feature_cols = ["temperature", "pressure", "humidity"]
        for c in feature_cols:
            df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")
            
    else:
        print("Dataset identificado: Combinado ou Formato Não Padrão. Aplicando binarização automática.")
        feature_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS]
        for c in feature_cols:
            if df_clean[c].dtype == object:
                df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce").fillna(0.0)

    feature_df = df_clean[feature_cols].copy()
    for col in feature_df.columns:
        if feature_df[col].isnull().any():
            median_val = feature_df[col].median()
            feature_df[col] = feature_df[col].fillna(median_val if not pd.isna(median_val) else 0.0)

    labels = df["label"].astype(str).tolist()
    print(f"Matrix de Features gerada: {feature_df.shape}")
    return feature_df, labels


def thermometer_encoding(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    resolution: int = 64,
) -> Tuple[List[List[int]], List[List[int]]]:
    train_encoded_parts = []
    test_encoded_parts = []
    
    for col in X_train.columns:
        col_train = X_train[col].to_numpy(dtype=float)
        col_test = X_test[col].to_numpy(dtype=float)
        
        unique_vals = np.unique(col_train[~np.isnan(col_train)])
        is_binary = set(unique_vals).issubset({0.0, 1.0})
        
        if is_binary:
            train_encoded_parts.append(col_train[:, None].astype(int))
            test_encoded_parts.append(col_test[:, None].astype(int))
        else:
            val_min, val_max = np.nanmin(col_train), np.nanmax(col_train)
            denominator = 1.0 if val_max == val_min else (val_max - val_min)
            
            clipped_train = np.clip((col_train - val_min) / denominator, 0.0, 1.0)
            clipped_test = np.clip((col_test - val_min) / denominator, 0.0, 1.0)
            
            bins = np.linspace(0, 1, resolution + 1)[1:]
            bin_train = (clipped_train[:, None] >= bins).astype(int)
            bin_test = (clipped_test[:, None] >= bins).astype(int)
            
            train_encoded_parts.append(bin_train)
            test_encoded_parts.append(bin_test)
            
    X_train_bin = np.hstack(train_encoded_parts)
    X_test_bin = np.hstack(test_encoded_parts)
    
    return X_train_bin.tolist(), X_test_bin.tolist()


def pad_bit_vectors(X: List[List[int]], address_size: int) -> List[List[int]]:
    if not X:
        return X
    bit_length = len(X[0])
    remainder = bit_length % address_size
    if remainder == 0:
        return X
    pad = address_size - remainder
    return [row + [0] * pad for row in X]


def undersample_majority(X: List[List[int]], y: List[str]) -> Tuple[List[List[int]], List[str]]:
    labels = sorted(set(y))
    counts = {label: y.count(label) for label in labels}
    minority_label = min(counts, key=counts.get)
    minority_count = counts[minority_label]

    indices_by_label = {label: [i for i, v in enumerate(y) if v == label] for label in labels}
    sampled_indices = []
    for label, indices in indices_by_label.items():
        if label == minority_label:
            sampled_indices.extend(indices)
        else:
            sampled_indices.extend(np.random.choice(indices, minority_count, replace=False).tolist())

    np.random.shuffle(sampled_indices)
    X_balanced = [X[i] for i in sampled_indices]
    y_balanced = [y[i] for i in sampled_indices]
    return X_balanced, y_balanced


def grid_search_cluwisard(
    X_train_raw: List[List[int]],
    X_test_raw: List[List[int]],
    y_train: List[str],
    y_test: List[str],
    address_sizes: List[int],
    min_scores: List[float],
    thresholds: List[int],
    discriminator_limits: List[int],
) -> Tuple[wp.ClusWisard, dict, str]:
    """
    Grid Search Robusto: O padding e os DataSets do wisardpkg são recriados 
    dinamicamente dentro do loop para permitir a variação do address_size.
    """
    best_model = None
    best_score = -1.0
    best_config = {}
    best_report = ""

    for size in address_sizes:
        # O tamanho do vetor binário muda dependendo do tamanho do endereço!
        X_train_padded = pad_bit_vectors(X_train_raw, size)
        X_test_padded = pad_bit_vectors(X_test_raw, address_size=size)
        
        train_ds = wp.DataSet(X_train_padded, y_train)
        test_ds = wp.DataSet(X_test_padded, y_test)
        
        for min_score in min_scores:
            for threshold in thresholds:
                for limit in discriminator_limits:
                    print(f"    Testing address_size={size}, min_score={min_score}, threshold={threshold}, limit={limit}...")
                    
                    model = wp.ClusWisard(size, min_score, threshold, limit)
                    model.train(train_ds)
                    y_pred = model.classify(test_ds)
                    
                    score = f1_score(y_test, y_pred, pos_label="1", zero_division=0)
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_config = {
                            "address_size": size,
                            "min_score": min_score,
                            "threshold": threshold,
                            "discriminator_limit": limit,
                        }
                        best_report = classification_report(y_test, y_pred, zero_division=0)

    if best_model is None:
        raise RuntimeError("Grid search did not produce any model.")

    return best_model, best_config, best_report


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline WiSARD Otimizado ToN_IoT.")
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE, help="Caminho base do diretório.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporção de teste (0.2 do artigo).")
    parser.add_argument("--random-state", type=int, default=42, help="Semente aleatória.")
    parser.add_argument("--balance", action="store_true", help="Aplica subamostragem no treino.")
    parser.add_argument("--grid-search", action="store_true", help="Executa busca de hiperparâmetros.")
    
    # Novo argumento para definir o nome do arquivo de saída
    parser.add_argument("--output", type=str, default="resultados_wisard.txt", help="Nome do arquivo txt para salvar as saídas.")

    # Hiperparâmetros mapeados para o Grid Search abrangente
    parser.add_argument("--address-sizes", nargs="*", type=int, default=[4, 6, 8, 12, 16], help="Tamanhos de endereço para testar.")
    parser.add_argument("--min-scores", nargs="*", type=float, default=[0.05, 0.1, 0.15], help="Scores mínimos para testar.")
    parser.add_argument("--thresholds", nargs="*", type=int, default=[5, 10], help="Limiares de bleach para testar.")
    parser.add_argument("--discriminator-limits", nargs="*", type=int, default=[5, 10, 20], help="Limites de sub-clusters.")
    parser.add_argument("--thermometer-resolution", nargs="*", type=int, default=[64, 128, 256], help="Resolução do termômetro.")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Inicia o redirecionamento dos prints para o arquivo e para a tela simultaneamente
    sys.stdout = Logger(args.output)
    
    print(f"As saídas desta execução estão sendo salvas em: {args.output}\n")
    
    for dataset_name in DATASET_NAMES:
        print(f"\n=== Processando dataset: {dataset_name} ===")
        current_dataset = os.path.join(args.data_file, dataset_name)
        
        try:
            df = load_dataset(current_dataset)
        except FileNotFoundError:
            print(f"Arquivo {dataset_name} não encontrado. Pulando...")
            continue
            
        X_df, y = pipeline_especifico_dataset(df)

        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X_df, y, test_size=args.test_size, random_state=args.random_state, stratify=y
        )

        for resolution in args.thermometer_resolution:
            print(f"\n🔧 Aplicando codificação termômetro com resolução {resolution}...")
            X_train_bin, X_test_bin = thermometer_encoding(X_train_df, X_test_df, resolution)
            X_train_bin = [list(map(int, row)) for row in X_train_bin]
            X_test_bin = [list(map(int, row)) for row in X_test_bin]

            if args.balance:
                X_train_bin, y_train = undersample_majority(X_train_bin, y_train)

            if args.grid_search:
                model, best_config, best_report = grid_search_cluwisard(
                    X_train_bin, X_test_bin, y_train, y_test,
                    args.address_sizes, args.min_scores, args.thresholds, args.discriminator_limits
                )
                print(f"\n🏆 Melhor configuração para {dataset_name}: {best_config}")
                print(best_report)
            else:
                # Caso não use grid search, adota endereço padrão 8 (melhor para vetores enxutos)
                X_train_padded = pad_bit_vectors(X_train_bin, 8)
                X_test_padded = pad_bit_vectors(X_test_bin, 8)
                train_ds = wp.DataSet(X_train_padded, y_train)
                test_ds = wp.DataSet(X_test_padded, y_test)
                
                model = wp.ClusWisard(8, 0.1, 10, 10)
                model.train(train_ds)
                y_pred = model.classify(test_ds)
                print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    main()