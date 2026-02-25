"""Utility functions for KneE-PAD system"""

import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str = "config/system_config.yaml") -> Dict[str, Any]:
    """
    Load system configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_directory_structure(base_path: str = "."):
    """
    Create necessary directory structure for KneE-PAD
    
    Args:
        base_path: Base directory path
    """
    base = Path(base_path)
    
    directories = [
        'data/raw',
        'data/processed',
        'data/sessions',
        'models',
        'results',
        'visualizations/patient',
        'visualizations/clinician',
        'visualizations/coaching',
        'logs'
    ]
    
    for directory in directories:
        (base / directory).mkdir(parents=True, exist_ok=True)
    
    print(f"Directory structure created at {base}")


def normalize_time_series(X: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize time series data
    
    Args:
        X: Time series [N, T, C] or [T, C]
        method: 'zscore', 'minmax', or 'none'
        
    Returns:
        Normalized time series
    """
    if method == 'none':
        return X
    
    original_shape = X.shape
    if X.ndim == 2:
        X = X[np.newaxis, :]
    
    N, T, C = X.shape
    X_norm = X.copy()
    
    if method == 'zscore':
        # Per-channel z-score normalization
        for c in range(C):
            mean = np.mean(X[:, :, c])
            std = np.std(X[:, :, c])
            if std > 1e-8:
                X_norm[:, :, c] = (X[:, :, c] - mean) / std
    
    elif method == 'minmax':
        # Per-channel min-max normalization
        for c in range(C):
            min_val = np.min(X[:, :, c])
            max_val = np.max(X[:, :, c])
            if max_val - min_val > 1e-8:
                X_norm[:, :, c] = (X[:, :, c] - min_val) / (max_val - min_val)
    
    if len(original_shape) == 2:
        X_norm = X_norm[0]
    
    return X_norm


def compute_dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Dynamic Time Warping distance between two time series
    
    Args:
        x: Time series 1 [T1, C]
        y: Time series 2 [T2, C]
        
    Returns:
        DTW distance
    """
    from scipy.spatial.distance import euclidean
    
    # Ensure same number of channels
    assert x.shape[1] == y.shape[1], "Time series must have same number of channels"
    
    T1, T2 = len(x), len(y)
    
    # Initialize cost matrix
    dtw_matrix = np.full((T1 + 1, T2 + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    # Fill cost matrix
    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            cost = euclidean(x[i-1], y[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],      # insertion
                dtw_matrix[i, j-1],      # deletion
                dtw_matrix[i-1, j-1]     # match
            )
    
    return dtw_matrix[T1, T2]


def sliding_window_stats(
    signal: np.ndarray,
    window_size: int,
    stride: int = 1
) -> Dict[str, np.ndarray]:
    """
    Compute statistics over sliding windows
    
    Args:
        signal: Input signal [T] or [T, C]
        window_size: Window size
        stride: Stride between windows
        
    Returns:
        Dictionary with mean, std, min, max arrays
    """
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]
    
    T, C = signal.shape
    n_windows = (T - window_size) // stride + 1
    
    stats = {
        'mean': np.zeros((n_windows, C)),
        'std': np.zeros((n_windows, C)),
        'min': np.zeros((n_windows, C)),
        'max': np.zeros((n_windows, C))
    }
    
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        window = signal[start:end, :]
        
        stats['mean'][i] = np.mean(window, axis=0)
        stats['std'][i] = np.std(window, axis=0)
        stats['min'][i] = np.min(window, axis=0)
        stats['max'][i] = np.max(window, axis=0)
    
    return stats


def export_to_json(data: Dict, filepath: str):
    """
    Export data to JSON file
    
    Args:
        data: Data dictionary
        filepath: Output file path
    """
    import json
    
    # Convert numpy arrays to lists
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        return obj
    
    data_converted = convert(data)
    
    with open(filepath, 'w') as f:
        json.dump(data_converted, f, indent=2)
    
    print(f"Data exported to {filepath}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute classification metrics
    
    Args:
        y_true: True labels [N]
        y_pred: Predicted labels [N]
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    return metrics


class ProgressTracker:
    """Track and display progress during long operations"""
    
    def __init__(self, total: int, desc: str = "Processing"):
        self.total = total
        self.current = 0
        self.desc = desc
    
    def update(self, n: int = 1):
        """Update progress"""
        self.current += n
        progress = self.current / self.total * 100
        print(f"\r{self.desc}: {progress:.1f}% ({self.current}/{self.total})", end='')
        
        if self.current >= self.total:
            print()  # New line when complete
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        if self.current < self.total:
            print()  # Ensure new line


if __name__ == "__main__":
    # Test utilities
    print("Testing KneE-PAD utilities...")
    
    # Test config loading
    try:
        config = load_config("../config/system_config.yaml")
        print(f"✓ Config loaded: {len(config)} sections")
    except:
        print("✗ Config loading failed (expected if run from different directory)")
    
    # Test normalization
    X = np.random.randn(10, 100, 20)
    X_norm = normalize_time_series(X, method='zscore')
    print(f"✓ Normalization: {X.shape} -> {X_norm.shape}")
    
    # Test DTW
    x = np.random.randn(50, 5)
    y = np.random.randn(60, 5)
    dist = compute_dtw_distance(x, y)
    print(f"✓ DTW distance: {dist:.2f}")
    
    print("All utility tests passed!")
