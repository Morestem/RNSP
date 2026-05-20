#!/usr/bin/env python3
"""
Train only the WORKING models: ULEEN and ClusWiSaRD
(torchwnn and BTHOWeN have a fundamental bug causing single-class predictions)

This script:
- Tests multiple hyperparameter configurations for each model
- Trains on 7 individual datasets + 1 combined dataset
- Saves comprehensive results with metrics and execution times
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import sys
import os
import time
import json
from datetime import datetime
from itertools import product

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'ULEEN', 'software_model'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'torchwnn'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'BTHOWeN', 'software_model'))

try:
    from ULEEN.software_model.model import BackpropWiSARD as UleenWisard
except ImportError as e:
    print(f"Error importing ULEEN: {e}")
    sys.exit(1)

try:
    import wisardpkg as wp
except ImportError:
    print("wisardpkg not available - ClusWiSaRD will be skipped")
    wp = None

try:
    from torchwnn.classifiers import Wisard as TorchWNNWisard
    from torchwnn.classifiers import BloomWisard as BloomWisard
    print("✓ torchwnn imported successfully")
except ImportError as e:
    print(f"torchwnn not available: {e}")
    TorchWNNWisard = None
    BloomWisard = None

try:
    from wisard import WiSARD as BTHOWeNWisard
    print("✓ BTHOWeN imported successfully")
except ImportError as e:
    print(f"BTHOWeN not available: {e}")
    BTHOWeNWisard = None

# Create output directory
os.makedirs('_results', exist_ok=True)
os.makedirs('_results/binarized_datasets', exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ========== BINARIZATION CONFIGURATION ==========
TEXT_STATE_MAPPING = {'off': 0, 'on': 1, 'closed': 0, 'open': 1, 'low': 0, 'high': 1}
CONTROL_COLS = {'label', 'type', 'date', 'time', 'device_origin'}
BITS_RESOLUTION = 16  # Bits for numerical features


# ========== BINARIZATION FUNCTIONS ==========
def encode_circular_thermometer(values, max_val, num_bits=8):
    """Encodes circular continuous values into a circular thermometer (vectorized)."""
    values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)
    start_bits = ((values / max_val) * num_bits).astype(np.int32) % num_bits
    
    thermometer = np.zeros((len(values), num_bits), dtype=np.uint8)
    window = num_bits // 2 
    
    for w in range(window):
        bit_pos = (start_bits + w) % num_bits
        thermometer[np.arange(len(values)), bit_pos] = 1
        
    return thermometer


def binarize_dataframe_densely(df, resolution, cat_resolution=8):
    """Applies heterogeneous thermometers and presence flags to dataframe."""
    df_clean = df.copy()
    final_bin_parts = []
    
    # 1. Circular Temporal Extraction
    if 'time' in df_clean.columns:
        presence_time = df_clean['time'].notna().astype(np.uint8).values.reshape(-1, 1)
        hours = pd.to_datetime(df_clean['time'], errors='coerce', format='mixed').dt.hour.fillna(0).values
        therm_hours = encode_circular_thermometer(hours, max_val=24, num_bits=12)
        final_bin_parts.append(np.hstack([presence_time, therm_hours]))
        
    if 'date' in df_clean.columns:
        presence_date = df_clean['date'].notna().astype(np.uint8).values.reshape(-1, 1)
        dow = pd.to_datetime(df_clean['date'], errors='coerce', format='mixed').dt.dayofweek.fillna(0).values
        therm_dow = encode_circular_thermometer(dow, max_val=7, num_bits=8)
        final_bin_parts.append(np.hstack([presence_date, therm_dow]))

    # 2. Binary Textual State Standardization
    for col in df_clean.columns:
        if col not in CONTROL_COLS and df_clean[col].dtype == 'object':
            unique_vals = df_clean[col].dropna().astype(str).str.strip().str.lower().unique()
            if any(val in TEXT_STATE_MAPPING for val in unique_vals):
                df_clean[col] = df_clean[col].astype(str).str.strip().str.lower().map(TEXT_STATE_MAPPING)

    # 3. Structural Attribute Processing
    features_to_process = [c for c in df_clean.columns if c not in CONTROL_COLS]
    
    for col in features_to_process:
        presence_flag = df_clean[col].notna().astype(np.uint8).values.reshape(-1, 1)
        
        # Categorical columns
        if df_clean[col].dtype == 'object':
            filled_cat = df_clean[col].fillna('missing').astype('category')
            cats = filled_cat.cat.codes.values
            norm_cats = cats / cats.max() if cats.max() > 0 else np.zeros_like(cats)
            
            thermometer_cat = np.zeros((len(norm_cats), cat_resolution), dtype=np.uint8)
            for i in range(cat_resolution):
                threshold = (i + 1) / (cat_resolution + 1)
                thermometer_cat[norm_cats >= threshold, i] = 1
            final_bin_parts.append(np.hstack([presence_flag, thermometer_cat]))
            continue
            
        # Numeric columns
        val = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).values
        unique_vals = np.unique(val)
        
        # Binary case (only 0 and 1)
        if np.all(np.isin(unique_vals, [0, 1])):
            bi_part = val.astype(np.uint8).reshape(-1, 1)
            final_bin_parts.append(np.hstack([presence_flag, bi_part]))
            continue
            
        # Log scale for high-range values
        if val.max() > 1000 or (val.max() / (val.min() + 1e-5) > 500):
            val = np.log1p(val)
            
        # Normalize
        min_v, max_v = val.min(), val.max()
        norm = np.zeros_like(val) if max_v - min_v == 0 else (val - min_v) / (max_v - min_v)
        
        # Thermometer encoding
        thermometer_num = np.zeros((len(norm), resolution), dtype=np.uint8)
        for i in range(resolution):
            threshold = (i + 1) / (resolution + 1)
            thermometer_num[norm >= threshold, i] = 1
            
        final_bin_parts.append(np.hstack([presence_flag, thermometer_num]))
        
    return np.hstack(final_bin_parts) if final_bin_parts else np.empty((len(df), 0), dtype=np.uint8)


def load_and_preprocess_dataset(csv_path, max_samples=None):
    """Load dataset and preprocess with dense binarization."""
    print(f"📂 Loading {os.path.basename(csv_path)}...")
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()
    
    print(f"   Raw samples: {len(df)}")
    
    # Keep rows as long as label is present; preserve feature-level NaNs for presence flags
    label_col = None
    for col in df.columns:
        if col.lower() == 'label':
            label_col = col
            break
    
    if label_col is None:
        print(f"   ⚠️  No 'label' column found!")
        return None, None
    
    df = df[df[label_col].notna()].copy()
    print(f"   Samples with label: {len(df)}")
    
    # Extract labels
    y = df[label_col].values
    
    # Convert label to numeric
    unique_labels = np.unique(y)
    if len(unique_labels) == 0:
        print(f"   ⚠️  No labels found!")
        return None, None
    
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_numeric = np.array([label_map[label] for label in y])
    
    print(f"   Classes: {dict(zip(unique_labels, np.unique(y_numeric, return_counts=True)[1]))}")
    
    # Binarize with dense encoding
    X_binarized = binarize_dataframe_densely(df, BITS_RESOLUTION)
    
    if X_binarized.shape[1] == 0:
        print(f"   ⚠️  No valid features after binarization!")
        return None, None
    
    print(f"   Binarized features: {X_binarized.shape}")
    
    unique_classes, class_counts = np.unique(y_numeric, return_counts=True)
    print(f"   Class distribution: {dict(zip(unique_classes, class_counts))}")
    
    if len(class_counts) == 0:
        print(f"   ⚠️  No valid classes found!")
        return None, None
    
    # If max_samples is provided, apply balancing/subsampling; otherwise keep full dataset
    if max_samples is None:
        return X_binarized, y_numeric
    
    min_class_count = min(class_counts.min(), max_samples // len(unique_classes))
    balanced_indices = []
    for cls in unique_classes:
        cls_indices = np.where(y_numeric == cls)[0]
        selected = np.random.choice(cls_indices, min_class_count, replace=False)
        balanced_indices.extend(selected)
    
    balanced_indices = np.array(balanced_indices)
    np.random.shuffle(balanced_indices)
    
    X_balanced = X_binarized[balanced_indices]
    y_balanced = y_numeric[balanced_indices]
    
    print(f"   Balanced samples: {len(X_balanced)}")
    unique_classes, class_counts = np.unique(y_balanced, return_counts=True)
    print(f"   Balanced class distribution: {dict(zip(unique_classes, class_counts))}")
    
    return X_balanced, y_balanced


def train_uleen(X_train, X_test, y_train, y_test, **kwargs):
    """Train ULEEN with custom parameters"""
    filter_inputs = kwargs.get('filter_inputs', 3)
    filter_entries = kwargs.get('filter_entries', 16)
    learning_rate = kwargs.get('learning_rate', 0.01)
    epochs = kwargs.get('epochs', 10)
    
    model = UleenWisard(
        inputs=X_train.shape[1],
        classes=2,
        filter_inputs=filter_inputs,
        filter_entries=filter_entries,
        filter_hash_functions=2
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    Xt = torch.tensor(X_train, dtype=torch.long)
    yt = torch.tensor(y_train, dtype=torch.long)
    Xt_test = torch.tensor(X_test, dtype=torch.long)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(Xt)
        loss = criterion(outputs, yt)
        loss.backward()
        optimizer.step()
        model.clamp()
    
    model.eval()
    with torch.no_grad():
        outputs = model(Xt_test)
        preds = torch.argmax(outputs, dim=1).numpy()
    
    return preds


def train_cluswisard(X_train, X_test, y_train, y_test, **kwargs):
    """Train ClusWiSaRD with custom parameters (wisardpkg 2.0.0a7 API)"""
    address_size = kwargs.get('address_size', 3)
    default_bleach = kwargs.get('default_bleach', 0.1)
    
    model = wp.ClusWisard(address_size, default_bleach, 10, 5)
    
    # Convert to integer binary vectors
    X_train_int = [[int(x) for x in row] for row in X_train]
    y_train_str = [str(int(y)) for y in y_train]  # Labels as STRINGS!
    X_test_int = [[int(x) for x in row] for row in X_test]
    
    # Create DataSet objects with string labels
    train_dataset = wp.DataSet(X_train_int, y_train_str)
    test_dataset = wp.DataSet(X_test_int)
    
    # Train
    model.train(train_dataset)
    
    # Predict - result format is "prediction::confidence"
    preds = []
    for i in range(len(X_test)):
        pred_str = model.classify(test_dataset[i])
        pred_int = int(pred_str.split("::")[0])
        preds.append(pred_int)
    
    return np.array(preds)


def train_torchwnn_wisard(X_train, X_test, y_train, y_test, **kwargs):
    """Train torchwnn WiSARD model"""
    if TorchWNNWisard is None:
        raise ImportError("torchwnn not available")
    
    address_size = kwargs.get('address_size', 3)
    
    # Convert to torch tensors (long for boolean values)
    X_train_t = torch.tensor(X_train, dtype=torch.long)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.long)
    
    # Create and train model
    model = TorchWNNWisard(
        entry_size=X_train.shape[1],
        n_classes=len(np.unique(y_train)),
        tuple_size=address_size,
        bleaching=False
    )
    
    model.fit(X_train_t, y_train_t)
    
    # Predict
    with torch.no_grad():
        preds_t = model.predict(X_test_t)
    
    return preds_t.numpy().astype(int)


def train_torchwnn_bloom(X_train, X_test, y_train, y_test, **kwargs):
    """Train torchwnn BloomWiSARD model"""
    if BloomWisard is None:
        raise ImportError("torchwnn BloomWiSARD not available")
    
    address_size = kwargs.get('address_size', 3)
    filter_size = kwargs.get('filter_size', 256)
    
    # Convert to torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.long)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.long)
    
    # Create and train model
    model = BloomWisard(
        entry_size=X_train.shape[1],
        n_classes=len(np.unique(y_train)),
        tuple_size=address_size,
        bleaching=False,
        filter_size=filter_size
    )
    
    model.fit(X_train_t, y_train_t)
    
    # Predict
    with torch.no_grad():
        preds_t = model.predict(X_test_t)
    
    return preds_t.numpy().astype(int)


def train_bethown_wisard(X_train, X_test, y_train, y_test, **kwargs):
    """Train BTHOWeN WiSARD model"""
    if BTHOWeNWisard is None:
        raise ImportError("BTHOWeN not available")
    
    address_size = kwargs.get('address_size', 3)
    unit_entries = kwargs.get('unit_entries', 256)  # Must be power of 2
    unit_hashes = kwargs.get('unit_hashes', 1)
    
    # Convert to int (for BTHOWeN which works with numpy)
    X_train_int = X_train.astype(int)
    y_train_int = y_train.astype(int)
    X_test_int = X_test.astype(int)
    
    # Create and train model
    model = BTHOWeNWisard(
        num_inputs=X_train.shape[1],
        num_classes=len(np.unique(y_train)),
        unit_inputs=address_size,
        unit_entries=unit_entries,
        unit_hashes=unit_hashes
    )
    
    # Train
    for sample, label in zip(X_train_int, y_train_int):
        model.train(sample, int(label))
    
    # Predict
    preds = []
    for sample in X_test_int:
        pred_indices = model.predict(sample)
        # If multiple classes have max response, take the first (lowest index)
        preds.append(int(pred_indices[0]))
    
    return np.array(preds)


def train_and_evaluate(model_name, X_train, X_test, y_train, y_test, params):
    """Train model and evaluate"""
    try:
        start_time = time.time()
        
        if model_name == 'uleen':
            preds = train_uleen(X_train, X_test, y_train, y_test, **params)
        elif model_name == 'cluswisard':
            preds = train_cluswisard(X_train, X_test, y_train, y_test, **params)
        elif model_name == 'torchwnn_wisard':
            preds = train_torchwnn_wisard(X_train, X_test, y_train, y_test, **params)
        elif model_name == 'torchwnn_bloom':
            preds = train_torchwnn_bloom(X_train, X_test, y_train, y_test, **params)
        elif model_name == 'bethown':
            preds = train_bethown_wisard(X_train, X_test, y_train, y_test, **params)
        else:
            return None
        
        exec_time = time.time() - start_time
        
        # Metrics
        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average='weighted', zero_division=0)
        recall = recall_score(y_test, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, preds)
        
        unique_preds, pred_counts = np.unique(preds, return_counts=True)
        pred_dist = {str(int(k)): int(v) for k, v in zip(unique_preds, pred_counts)}
        
        return {
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'confusion_matrix': cm.tolist(),
            'prediction_distribution': pred_dist,
            'execution_time': float(exec_time),
            'parameters': {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in params.items()},
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'status': 'failed',
            'error': str(e)
        }


def save_binarized_dataset(X, y, identifier, tags=None):
    """Save binarized dataset to CSV for inspection"""
    output_dir = '_results/binarized_datasets'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataframe from binarized data
    feature_names = [f"bit_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y
    
    if tags is not None:
        df['device_origin'] = tags
    
    # Save CSV
    output_file = os.path.join(output_dir, f'binarized_{identifier}.csv')
    df.to_csv(output_file, index=False)
    print(f"    📊 Binarized dataset saved: {output_file} ({df.shape})")
    
    # Save sample
    sample_file = os.path.join(output_dir, f'binarized_{identifier}_sample.csv')
    df.head(100).to_csv(sample_file, index=False)
    print(f"    📊 Sample saved: {sample_file}")
    
    return df


def main():
    print(f"\n{'='*70}")
    print(f"TRAINING ALL MODELS WITH DENSE BINARIZATION: {timestamp}")
    print(f"{'='*70}")
    
    available_models = ['uleen', 'cluswisard', 'torchwnn_wisard', 'torchwnn_bloom', 'bethown']
    if not wp:
        available_models.remove('cluswisard')
    if TorchWNNWisard is None:
        available_models.remove('torchwnn_wisard')
        available_models.remove('torchwnn_bloom')
    if BTHOWeNWisard is None:
        available_models.remove('bethown')
    
    print(f"Available models: {', '.join(available_models)}")
    
    # Get datasets
    datasets = sorted([f for f in os.listdir('_dataset') if f.startswith('IoT_') and f.endswith('.csv')])
    print(f"Found {len(datasets)} datasets: {', '.join([d.replace('IoT_', '').replace('.csv', '') for d in datasets])}")
    
    max_samples_per_dataset = None
    results = {
        'timestamp': timestamp,
        'available_models': available_models,
        'datasets': {},
        'combined': {}
    }
    
    # ========== PHASE 1: Individual datasets ==========
    print(f"\n{'='*70}")
    print("PHASE 1: Training on Individual Datasets")
    print(f"{'='*70}")
    
    all_X_train = []
    all_X_test = []
    all_y_train = []
    all_y_test = []
    all_X_full = []
    all_y_full = []
    dataset_origins = []
    dataset_max_features = 0  # Track maximum features
    
    # First pass: determine maximum features and save datasets
    for dataset_file in datasets:
        dataset_path = os.path.join('_dataset', dataset_file)
        dataset_name = dataset_file.replace('.csv', '').replace('IoT_', '')
        
        print(f"\n{'─'*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'─'*70}")
        
        X, y = load_and_preprocess_dataset(dataset_path, max_samples=max_samples_per_dataset)
        
        if X is None:
            print(f"   ⚠️  Skipping - failed to load")
            continue
        
        # Track maximum features
        dataset_max_features = max(dataset_max_features, X.shape[1])
        
        # Save full binarized dataset
        print(f"\n  💾 Saving binarized dataset...")
        device_tags = np.array([dataset_name] * len(X))
        save_binarized_dataset(X, y, dataset_name, tags=device_tags)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        all_X_train.append(X_train)
        all_X_test.append(X_test)
        all_y_train.append(y_train)
        all_y_test.append(y_test)
        all_X_full.append(X)
        all_y_full.append(y)
        dataset_origins.append([dataset_name] * len(X))
        
        results['datasets'][dataset_name] = {'models': {}, 'binarized_shape': X.shape}
    
    # Pad all datasets to have the same number of features
    if all_X_full:
        print(f"\n📐 Padding datasets to {dataset_max_features} features...")
        for i in range(len(all_X_train)):
            if all_X_train[i].shape[1] < dataset_max_features:
                pad_size = dataset_max_features - all_X_train[i].shape[1]
                all_X_train[i] = np.pad(all_X_train[i], ((0, 0), (0, pad_size)))
                all_X_test[i] = np.pad(all_X_test[i], ((0, 0), (0, pad_size)))
                all_X_full[i] = np.pad(all_X_full[i], ((0, 0), (0, pad_size)))
    
    # Second pass: train all models on each dataset
    print(f"\n{'='*70}")
    print("PHASE 1: Training Models on Individual Datasets")
    print(f"{'='*70}")
    
    # Second pass: train all models on each dataset
    print(f"\n{'='*70}")
    print("PHASE 1: Training Models on Individual Datasets")
    print(f"{'='*70}")
    
    dataset_names_list = list(results['datasets'].keys())
    
    for idx, dataset_name in enumerate(dataset_names_list):
        print(f"\n{'─'*70}")
        print(f"Dataset: {dataset_name} ({idx+1}/{len(dataset_names_list)})")
        print(f"{'─'*70}")
        
        X_train = all_X_train[idx]
        X_test = all_X_test[idx]
        y_train = all_y_train[idx]
        y_test = all_y_test[idx]
        
        # Train all available models
        for model_name in available_models:
            print(f"\n  🔍 Testing {model_name.upper()}...")
            best_result = None
            best_params = None
            
            # Generate parameter configurations for each model
            if model_name == 'uleen':
                configs = list(product([2, 3], [8, 16], [0.001, 0.01, 0.1], [5, 10]))[:10]
                config_params = [
                    {'filter_inputs': c[0], 'filter_entries': c[1], 'learning_rate': c[2], 'epochs': c[3]}
                    for c in configs
                ]
                config_names = [
                    f"f_in={c[0]}, f_ent={c[1]}, lr={c[2]}, ep={c[3]}"
                    for c in configs
                ]
            
            elif model_name == 'cluswisard':
                configs = list(product([2, 3, 4], [0.05, 0.1, 0.2]))[:9]
                config_params = [
                    {'address_size': c[0], 'default_bleach': c[1]}
                    for c in configs
                ]
                config_names = [
                    f"addr_sz={c[0]}, bleach={c[1]}"
                    for c in configs
                ]
            
            elif model_name == 'torchwnn_wisard':
                configs = [[2], [3], [4]]
                config_params = [{'address_size': c[0]} for c in configs]
                config_names = [f"addr_sz={c[0]}" for c in configs]
            
            elif model_name == 'torchwnn_bloom':
                configs = list(product([2, 3], [128, 256]))[:4]
                config_params = [
                    {'address_size': c[0], 'filter_size': c[1]}
                    for c in configs
                ]
                config_names = [
                    f"addr_sz={c[0]}, filt_sz={c[1]}"
                    for c in configs
                ]
            
            elif model_name == 'bethown':
                configs = list(product([2, 3], [128, 256]))[:4]
                config_params = [
                    {'address_size': c[0], 'unit_entries': c[1]}
                    for c in configs
                ]
                config_names = [
                    f"addr_sz={c[0]}, ent={c[1]}"
                    for c in configs
                ]
            
            else:
                continue
            
            for i, params in enumerate(config_params):
                try:
                    result = train_and_evaluate(model_name, X_train, X_test, y_train, y_test, params)
                    
                    if result and result['status'] == 'success':
                        acc = result['accuracy']
                        f1 = result['f1']
                        print(f"    ✓ {config_names[i]}: Acc={acc:.4f}, F1={f1:.4f}, Time={result['execution_time']:.2f}s")
                        
                        if best_result is None or result['f1'] > best_result['f1']:
                            best_result = result
                            best_params = params
                    else:
                        print(f"    ✗ {config_names[i]}: Failed - {result.get('error', 'Unknown error')}")
                except Exception as e:
                    print(f"    ✗ {config_names[i]}: Exception - {str(e)}")
            
            if best_result:
                results['datasets'][dataset_name]['models'][model_name] = best_result
    
    # ========== PHASE 2: Combined dataset ==========
    if all_X_train:
        print(f"\n{'='*70}")
        print("PHASE 2: Training on Combined Dataset")
        print(f"{'='*70}")
        
        # Combine all full datasets (already padded)
        X_combined_full = np.vstack(all_X_full)
        y_combined_full = np.hstack(all_y_full)
        origins_combined = np.hstack(dataset_origins)
        
        print(f"Combined dataset shape: {X_combined_full.shape}")
        
        # Save combined binarized dataset
        print(f"\n  💾 Saving combined binarized dataset...")
        save_binarized_dataset(X_combined_full, y_combined_full, 'combined', tags=origins_combined)
        
        # Combine train/test splits (already padded)
        X_train_combined = np.vstack(all_X_train)
        X_test_combined = np.vstack(all_X_test)
        y_train_combined = np.hstack(all_y_train)
        y_test_combined = np.hstack(all_y_test)
        
        print(f"Combined train: {X_train_combined.shape}")
        print(f"Combined test: {X_test_combined.shape}")
        
        results['combined']['binarized_shape'] = X_combined_full.shape
        
        # Train all models on combined
        for model_name in available_models:
            print(f"\n  🔍 Testing {model_name.upper()} on combined...")
            best_result = None
            best_params = None
            
            # Generate parameter configurations
            if model_name == 'uleen':
                configs = list(product([2, 3], [8, 16], [0.001, 0.01, 0.1], [5, 10]))[:10]
                config_params = [
                    {'filter_inputs': c[0], 'filter_entries': c[1], 'learning_rate': c[2], 'epochs': c[3]}
                    for c in configs
                ]
                config_names = [
                    f"f_in={c[0]}, f_ent={c[1]}, lr={c[2]}, ep={c[3]}"
                    for c in configs
                ]
            
            elif model_name == 'cluswisard':
                configs = list(product([2, 3, 4], [0.05, 0.1, 0.2]))[:9]
                config_params = [
                    {'address_size': c[0], 'default_bleach': c[1]}
                    for c in configs
                ]
                config_names = [
                    f"addr_sz={c[0]}, bleach={c[1]}"
                    for c in configs
                ]
            
            elif model_name == 'torchwnn_wisard':
                configs = [[2], [3], [4]]
                config_params = [{'address_size': c[0]} for c in configs]
                config_names = [f"addr_sz={c[0]}" for c in configs]
            
            elif model_name == 'torchwnn_bloom':
                configs = list(product([2, 3], [128, 256]))[:4]
                config_params = [
                    {'address_size': c[0], 'filter_size': c[1]}
                    for c in configs
                ]
                config_names = [
                    f"addr_sz={c[0]}, filt_sz={c[1]}"
                    for c in configs
                ]
            
            elif model_name == 'bethown':
                configs = list(product([2, 3], [128, 256]))[:4]
                config_params = [
                    {'address_size': c[0], 'unit_entries': c[1]}
                    for c in configs
                ]
                config_names = [
                    f"addr_sz={c[0]}, ent={c[1]}"
                    for c in configs
                ]
            
            else:
                continue
            
            for i, params in enumerate(config_params):
                try:
                    result = train_and_evaluate(model_name, X_train_combined, X_test_combined, 
                                               y_train_combined, y_test_combined, params)
                    
                    if result and result['status'] == 'success':
                        acc = result['accuracy']
                        f1 = result['f1']
                        print(f"    ✓ {config_names[i]}: Acc={acc:.4f}, F1={f1:.4f}, Time={result['execution_time']:.2f}s")
                        
                        if best_result is None or result['f1'] > best_result['f1']:
                            best_result = result
                            best_params = params
                    else:
                        print(f"    ✗ {config_names[i]}: Failed - {result.get('error', 'Unknown error')}")
                except Exception as e:
                    print(f"    ✗ {config_names[i]}: Exception - {str(e)}")
            
            if best_result:
                results['combined']['models'] = results['combined'].get('models', {})
                results['combined']['models'][model_name] = best_result
    
    # ========== Save results ==========
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    results_file = f'_results/all_models_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {results_file}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n📊 Binarized datasets saved to: _results/binarized_datasets/")
    
    for dataset_name, dataset_results in results['datasets'].items():
        print(f"\n{dataset_name} (shape: {dataset_results.get('binarized_shape', 'N/A')}):")
        for model_name, model_result in dataset_results['models'].items():
            if 'accuracy' in model_result:
                print(f"  {model_name}: Acc={model_result['accuracy']:.4f}, F1={model_result['f1']:.4f}, Time={model_result['execution_time']:.2f}s")
    
    if 'combined' in results and 'models' in results['combined']:
        print(f"\nCombined (shape: {results['combined'].get('binarized_shape', 'N/A')}):")
        for model_name, model_result in results['combined']['models'].items():
            if 'accuracy' in model_result:
                print(f"  {model_name}: Acc={model_result['accuracy']:.4f}, F1={model_result['f1']:.4f}, Time={model_result['execution_time']:.2f}s")


if __name__ == '__main__':
    main()
