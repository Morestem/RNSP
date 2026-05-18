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
import pickle
from datetime import datetime
import glob

# Add paths to import models
sys.path.append(os.path.join(os.path.dirname(__file__), 'BTHOWeN', 'software_model'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'ULEEN', 'software_model'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'torchwnn'))

try:
    from torchwnn.classifiers import Wisard as TorchWisard
    from torchwnn.classifiers import BloomWisard as TorchBloomWisard
except ImportError as e:
    print(f"Error importing torchwnn: {e}")

try:
    from BTHOWeN.software_model.wisard import WiSARD as BTHOWenWisard
except ImportError as e:
    print(f"Error importing BTHOWeN: {e}")

try:
    from ULEEN.software_model.model import BackpropWiSARD as UleenWisard
except ImportError as e:
    print(f"Error importing ULEEN: {e}")

try:
    import wisardpkg as wp
except ImportError:
    print("wisardpkg is not installed or available.")
    wp = None

# Create output directory
os.makedirs('_results', exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def log_and_print(message):
    """Log message to both console and file"""
    print(message)

def load_and_preprocess_dataset(csv_path, max_samples=None):
    """Load a dataset and preprocess it with balanced class distribution"""
    print(f"\n📂 Loading {os.path.basename(csv_path)}...")
    df = pd.read_csv(csv_path, low_memory=False)
    
    print(f"   Raw samples: {len(df)}")
    
    df = df.dropna()
    print(f"   After dropping NaN: {len(df)}")
    
    # Find label column
    label_col = None
    for col in df.columns:
        if col.lower() == 'label':
            label_col = col
            break
    
    if label_col is None:
        print(f"   ⚠️  No 'label' column found!")
        return None, None
    
    # Extract labels
    y = df[label_col].values
    
    # Convert label to numeric
    unique_labels = np.unique(y)
    if len(unique_labels) == 0:
        print(f"   ⚠️  No labels found!")
        return None, None
    
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_numeric = np.array([label_map[label] for label in y])
    
    # Select feature columns (skip non-feature columns)
    skip_cols = {label_col, 'date', 'time', 'type', 'Type'}
    feature_cols = [col for col in df.columns if col not in skip_cols]
    
    print(f"   Feature columns: {feature_cols}")
    
    # Convert all features to useful binary features (0 or 1)
    X_processed = []
    for col in feature_cols:
        col_data = df[col].values
        
        # Convert to string first to handle mixed types
        col_str = np.array([str(x).strip() for x in col_data])
        
        # Try to convert to numeric
        try:
            col_numeric = pd.to_numeric(col_str, errors='coerce')
            valid_mask = ~np.isnan(col_numeric)
            valid_vals = col_numeric[valid_mask]
            
            # If enough numeric values
            if valid_mask.sum() > len(col_numeric) * 0.5 and len(np.unique(valid_vals)) > 1:
                # Use min/max normalization then quartile thresholding
                col_min, col_max = np.nanmin(col_numeric), np.nanmax(col_numeric)
                
                # If range is zero, make arbitrary binary split
                if col_max == col_min:
                    f1 = np.zeros_like(col_numeric, dtype=float)
                    f1[~np.isnan(col_numeric)] = 1.0
                    X_processed.append(np.nan_to_num(f1, nan=0.0))
                else:
                    # Normalize to [0, 1]
                    col_norm = (col_numeric - col_min) / (col_max - col_min)
                    
                    # Create 3 binary features using percentile thresholds
                    q1 = np.nanpercentile(col_norm, 33)
                    q2 = np.nanpercentile(col_norm, 67)
                    
                    f1 = (col_norm > q1).astype(float)
                    f2 = (col_norm > q2).astype(float)
                    f3 = (col_norm > 0.5).astype(float)
                    
                    X_processed.append(np.nan_to_num(f1, nan=0.0))
                    X_processed.append(np.nan_to_num(f2, nan=0.0))
                    X_processed.append(np.nan_to_num(f3, nan=0.0))
                continue
        except:
            pass
        
        # Fall back to categorical: create multiple binary features
        unique_vals = np.unique(col_str)
        if len(unique_vals) > 1:
            # Use top 2 unique values as separators for 2 binary features
            for i, uv in enumerate(unique_vals[:2]):
                binary_col = (col_str == uv).astype(float)
                X_processed.append(binary_col)
        else:
            # Single unique value - add dummy feature
            X_processed.append(np.zeros(len(col_str)))
    
    if len(X_processed) == 0:
        print(f"   ⚠️  No valid features found!")
        return None, None
    
    # Normalize to max 12 features per dataset
    if len(X_processed) > 12:
        X_processed = X_processed[:12]
    
    X = np.column_stack(X_processed).astype(float)
    print(f"   Processed features: {X.shape} (reduced from {len(feature_cols)})")
    
    # Remove rows with NaN
    valid_idx = ~np.isnan(X).any(axis=1)
    X = X[valid_idx]
    y_numeric = y_numeric[valid_idx]
    
    print(f"   After removing NaN: {len(X)}")
    
    y = y_numeric
    
    # Balance dataset by taking equal samples from each class
    unique_classes, class_counts = np.unique(y, return_counts=True)
    print(f"   Original class distribution: {dict(zip(unique_classes, class_counts))}")
    
    if len(class_counts) == 0:
        print(f"   ⚠️  No valid classes found!")
        return None, None
    
    min_class_count = class_counts.min()
    
    # If max_samples specified, use smaller of max_samples or balanced size
    if max_samples is not None:
        min_class_count = min(min_class_count, max_samples // len(unique_classes))
    
    # Create balanced dataset
    balanced_indices = []
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        selected = np.random.choice(cls_indices, min_class_count, replace=False)
        balanced_indices.extend(selected)
    
    balanced_indices = np.array(balanced_indices)
    np.random.shuffle(balanced_indices)
    
    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]
    
    print(f"   Balanced samples: {len(X_balanced)}")
    unique_classes, class_counts = np.unique(y_balanced, return_counts=True)
    print(f"   Balanced class distribution: {dict(zip(unique_classes, class_counts))}")
    
    return X_balanced, y_balanced

def train_model(model_name, X_train, X_test, y_train, y_test, models_dir, dataset_name):
    """Train a single model and return metrics"""
    try:
        start_time = time.time()
        
        if model_name == 'torchwnn_wisard':
            device = torch.device("cpu")
            Xt = torch.tensor(X_train, dtype=torch.long).to(device)
            yt = torch.tensor(y_train, dtype=torch.long).to(device)
            Xt_test = torch.tensor(X_test, dtype=torch.long).to(device)
            yt_test = torch.tensor(y_test, dtype=torch.long).to(device)
            
            model = TorchWisard(entry_size=X_train.shape[1], n_classes=2, tuple_size=4)
            model.fit(Xt, yt)
            preds = model.predict(Xt_test).cpu().numpy()
            yt_test = yt_test.cpu().numpy()
            
        elif model_name == 'torchwnn_bloomwisard':
            device = torch.device("cpu")
            Xt = torch.tensor(X_train, dtype=torch.long).to(device)
            yt = torch.tensor(y_train, dtype=torch.long).to(device)
            Xt_test = torch.tensor(X_test, dtype=torch.long).to(device)
            yt_test = torch.tensor(y_test, dtype=torch.long).to(device)
            
            model = TorchBloomWisard(entry_size=X_train.shape[1], n_classes=2, tuple_size=4, filter_size=128, n_hashes=2)
            model.fit(Xt, yt)
            preds = model.predict(Xt_test).cpu().numpy()
            yt_test = yt_test.cpu().numpy()
            
        elif model_name == 'bthowen':
            model = BTHOWenWisard(num_inputs=X_train.shape[1], num_classes=2, unit_inputs=4, unit_entries=16, unit_hashes=2)
            for i in range(len(X_train)):
                model.train(X_train[i].astype(bool), y_train[i])
            
            preds = []
            for i in range(len(X_test)):
                p = model.predict(X_test[i].astype(bool))
                preds.append(p[0])
            preds = np.array(preds)
            
        elif model_name == 'uleen':
            model = UleenWisard(inputs=X_train.shape[1], classes=2, filter_inputs=4, filter_entries=16, filter_hash_functions=2)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()
            
            Xt = torch.tensor(X_train, dtype=torch.long)
            yt = torch.tensor(y_train, dtype=torch.long)
            Xt_test = torch.tensor(X_test, dtype=torch.long)
            
            model.train()
            for epoch in range(5):
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
                
        elif model_name == 'cluswisard':
            if wp is None:
                return None
            
            X_train_list = X_train.astype(int).tolist()
            y_train_list = y_train.astype(str).tolist()
            X_test_list = X_test.astype(int).tolist()
            y_test_list = y_test.astype(str).tolist()
            
            # Use smaller address size based on number of features
            addressSize = max(2, min(4, X_train.shape[1]))
            model = wp.ClusWisard(addressSize, 0.1, 10, 5)
            model.train(X_train_list, y_train_list)
            preds_raw = model.classify(X_test_list)
            preds = np.array([int(p) for p in preds_raw])
            y_test = np.array([int(y) for y in y_test_list])
        else:
            return None
        
        execution_time = time.time() - start_time
        
        # Debug: Check predictions
        unique_preds, pred_counts = np.unique(preds, return_counts=True)
        pred_dist = {int(label): int(count) for label, count in zip(unique_preds, pred_counts)}
        unique_true, true_counts = np.unique(y_test, return_counts=True)
        true_dist = {int(label): int(count) for label, count in zip(unique_true, true_counts)}
        
        # Calculate metrics
        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average='weighted', zero_division=0)
        recall = recall_score(y_test, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, preds)
        
        return {
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'confusion_matrix': cm.tolist(),
            'prediction_distribution': pred_dist,
            'execution_time': execution_time,
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'status': 'failed',
            'error': str(e)
        }

# Main execution
if __name__ == '__main__':
    # Get all datasets
    datasets = sorted(glob.glob('_dataset/IoT_*.csv'))
    
    print(f"\n{'='*70}")
    print(f"MULTI-DATASET TRAINING RUN: {timestamp}")
    print(f"{'='*70}")
    print(f"Found {len(datasets)} datasets")
    
    # Dataset parameters
    max_samples_per_dataset = 25000  # Balanced subset size (5K per class)
    
    # Models to train
    models = ['torchwnn_wisard', 'torchwnn_bloomwisard', 'bthowen', 'uleen', 'cluswisard']
    
    # Results structure
    all_results = {
        'timestamp': timestamp,
        'datasets': {},
        'combined': {}
    }
    
    # 1. Train on individual datasets
    print(f"\n{'='*70}")
    print("PHASE 1: Training on Individual Datasets")
    print(f"{'='*70}")
    
    all_X_train_combined = []
    all_X_test_combined = []
    all_y_train_combined = []
    all_y_test_combined = []
    
    for dataset_path in datasets:
        dataset_name = os.path.basename(dataset_path).replace('.csv', '').replace('IoT_', '')
        print(f"\n{'─'*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'─'*70}")
        
        # Load and preprocess
        X_bin, y = load_and_preprocess_dataset(dataset_path, max_samples=max_samples_per_dataset)
        
        # Skip if loading failed
        if X_bin is None or y is None:
            print(f"   ⚠️  Skipping {dataset_name} - failed to load")
            continue
        
        # Debug: Show feature statistics
        print(f"   📊 Feature stats:")
        print(f"      Shape: {X_bin.shape}, Data type: {X_bin.dtype}")
        print(f"      Feature mins: {X_bin.min(axis=0)[:5]}")
        print(f"      Feature maxs: {X_bin.max(axis=0)[:5]}")
        print(f"      Feature means: {X_bin.mean(axis=0)[:5].round(3)}")
        print(f"      Feature stds: {X_bin.std(axis=0)[:5].round(3)}")
        
        X_train, X_test, y_train, y_test = train_test_split(X_bin, y, test_size=0.3, random_state=42)
        
        all_results['datasets'][dataset_name] = {
            'n_samples': len(X_bin),
            'n_features': X_bin.shape[1],
            'models': {}
        }
        
        # Collect for combined dataset
        all_X_train_combined.append(X_train)
        all_X_test_combined.append(X_test)
        all_y_train_combined.append(y_train)
        all_y_test_combined.append(y_test)
        
        # Train each model
        for model_name in models:
            print(f"  Training {model_name}...", end=' ', flush=True)
            result = train_model(model_name, X_train, X_test, y_train, y_test, 
                               f'_results/models_{timestamp}', dataset_name)
            
            if result is None:
                print("⊘ Skipped")
                continue
            
            if result['status'] == 'success':
                acc = result['accuracy']
                f1 = result['f1']
                exec_time = result['execution_time']
                print(f"✓ Acc: {acc:.4f}, F1: {f1:.4f}, Time: {exec_time:.2f}s")
                all_results['datasets'][dataset_name]['models'][model_name] = result
            else:
                print(f"✗ Failed: {result['error'][:50]}")
    
    # 2. Train on combined dataset
    print(f"\n{'='*70}")
    print("PHASE 2: Training on Combined Dataset")
    print(f"{'='*70}")
    
    # Find max features across all datasets
    max_features = max([X.shape[1] for X in all_X_train_combined])
    print(f"\nMax features across datasets: {max_features}")
    
    # Pad all datasets to have same feature count
    X_train_padded = []
    X_test_padded = []
    for X_train, X_test in zip(all_X_train_combined, all_X_test_combined):
        if X_train.shape[1] < max_features:
            # Pad with zeros
            pad_train = np.pad(X_train, ((0, 0), (0, max_features - X_train.shape[1])), mode='constant')
            pad_test = np.pad(X_test, ((0, 0), (0, max_features - X_test.shape[1])), mode='constant')
        else:
            pad_train = X_train
            pad_test = X_test
        X_train_padded.append(pad_train)
        X_test_padded.append(pad_test)
    
    X_train_combined = np.vstack(X_train_padded)
    X_test_combined = np.vstack(X_test_padded)
    y_train_combined = np.hstack(all_y_train_combined)
    y_test_combined = np.hstack(all_y_test_combined)
    
    print(f"\nCombined dataset statistics:")
    print(f"  Total train samples: {len(X_train_combined)}")
    print(f"  Total test samples: {len(X_test_combined)}")
    print(f"  Features: {X_train_combined.shape[1]}")
    unique, counts = np.unique(y_train_combined, return_counts=True)
    print(f"  Class distribution (train): {dict(zip(unique, counts))}")
    
    all_results['combined'] = {
        'n_train_samples': len(X_train_combined),
        'n_test_samples': len(X_test_combined),
        'n_features': X_train_combined.shape[1],
        'models': {}
    }
    
    print(f"\n{'─'*70}")
    for model_name in models:
        print(f"Training {model_name}...", end=' ', flush=True)
        result = train_model(model_name, X_train_combined, X_test_combined, 
                           y_train_combined, y_test_combined,
                           f'_results/models_{timestamp}', 'combined')
        
        if result is None:
            print("⊘ Skipped")
            continue
        
        if result['status'] == 'success':
            acc = result['accuracy']
            f1 = result['f1']
            exec_time = result['execution_time']
            print(f"✓ Acc: {acc:.4f}, F1: {f1:.4f}, Time: {exec_time:.2f}s")
            all_results['combined']['models'][model_name] = result
        else:
            print(f"✗ Failed: {result['error'][:50]}")
    
    # 3. Save results
    print(f"\n{'='*70}")
    print("PHASE 3: Saving Results")
    print(f"{'='*70}")
    
    results_file = f'_results/multi_dataset_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    # 4. Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    print("\n📊 Individual Dataset Results:")
    for dataset_name, dataset_results in all_results['datasets'].items():
        print(f"\n  {dataset_name}:")
        for model_name, model_results in dataset_results['models'].items():
            acc = model_results['accuracy']
            f1 = model_results['f1']
            time_val = model_results['execution_time']
            print(f"    {model_name}: Acc={acc:.4f}, F1={f1:.4f}, Time={time_val:.2f}s")
    
    print(f"\n📊 Combined Dataset Results:")
    for model_name, model_results in all_results['combined']['models'].items():
        acc = model_results['accuracy']
        f1 = model_results['f1']
        time_val = model_results['execution_time']
        print(f"  {model_name}: Acc={acc:.4f}, F1={f1:.4f}, Time={time_val:.2f}s")
    
    print(f"\n{'='*70}\n")
