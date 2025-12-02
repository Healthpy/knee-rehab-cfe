"""
Utility functions for the XAI counterfactual analysis framework.
"""

import logging
import random
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import yaml
import json

# Import centralized constants
from .config import (
    LEFT_INJURY_SUBJECTS, RIGHT_INJURY_SUBJECTS, get_injury_side,
    SensorSpecifications, SensorGroups, samples_to_time
)


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )
    
    return logging.getLogger()


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_data_shape(
    data: np.ndarray,
    expected_shape: Optional[tuple] = None,
    min_dims: int = 2,
    max_dims: int = 3
) -> None:
    """
    Validate data array shape.
    
    Args:
        data: Data array to validate
        expected_shape: Expected shape tuple
        min_dims: Minimum number of dimensions
        max_dims: Maximum number of dimensions
        
    Raises:
        ValueError: If data shape is invalid
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array")
    
    if data.ndim < min_dims or data.ndim > max_dims:
        raise ValueError(
            f"Data must have {min_dims}-{max_dims} dimensions, got {data.ndim}"
        )
    
    if expected_shape is not None:
        if data.shape != expected_shape:
            raise ValueError(
                f"Expected shape {expected_shape}, got {data.shape}"
            )


def normalize_data(
    data: np.ndarray,
    method: str = "standardize",
    axis: Optional[Union[int, tuple]] = None
) -> np.ndarray:
    """
    Normalize data using specified method.
    
    Args:
        data: Data to normalize
        method: Normalization method ('standardize', 'minmax', 'l2')
        axis: Axis along which to normalize
        
    Returns:
        Normalized data array
    """
    if method == "standardize":
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        return (data - mean) / (std + 1e-8)
    
    elif method == "minmax":
        min_val = np.min(data, axis=axis, keepdims=True)
        max_val = np.max(data, axis=axis, keepdims=True)
        return (data - min_val) / (max_val - min_val + 1e-8)
    
    elif method == "l2":
        norm = np.linalg.norm(data, axis=axis, keepdims=True)
        return data / (norm + 1e-8)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_statistics(
    data: np.ndarray,
    axis: Optional[Union[int, tuple]] = None
) -> Dict[str, np.ndarray]:
    """
    Compute basic statistics for data array.
    
    Args:
        data: Data array
        axis: Axis along which to compute statistics
        
    Returns:
        Dictionary of statistics
    """
    return {
        'mean': np.mean(data, axis=axis),
        'std': np.std(data, axis=axis),
        'min': np.min(data, axis=axis),
        'max': np.max(data, axis=axis),
        'median': np.median(data, axis=axis),
        'q25': np.percentile(data, 25, axis=axis),
        'q75': np.percentile(data, 75, axis=axis)
    }


def create_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """
    Create correlation matrix for data.
    
    Args:
        data: Data array (samples x features)
        
    Returns:
        Correlation matrix
    """
    if data.ndim != 2:
        raise ValueError("Data must be 2D for correlation matrix")
    
    return np.corrcoef(data.T)


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays and other objects."""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_results(
    results: Dict[str, Any],
    output_path: Union[str, Path],
    format: str = "json"
) -> None:
    """
    Save results to file.
    
    Args:
        results: Results dictionary
        output_path: Output file path
        format: Output format ('json', 'yaml', 'npz')
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    
    if format == "json":
        with open(output_path, 'w') as f:
            json.dump(results, f, cls=CustomJSONEncoder, indent=2)
    
    elif format == "yaml":
        # Convert numpy arrays to lists for YAML serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(results)
        with open(output_path, 'w') as f:
            yaml.dump(serializable_results, f, default_flow_style=False, indent=2)
    
    elif format == "npz":
        # Only save numpy arrays
        arrays = {k: v for k, v in results.items() if isinstance(v, np.ndarray)}
        np.savez_compressed(output_path, **arrays)
    
    else:
        raise ValueError(f"Unknown format: {format}")


def load_results(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load results from file.
    
    Args:
        file_path: Input file path
        
    Returns:
        Results dictionary
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            return json.load(f)
    
    elif file_path.suffix in ['.yaml', '.yml']:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    
    elif file_path.suffix == '.npz':
        data = np.load(file_path)
        return {key: data[key] for key in data.files}
    
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def get_device() -> str:
    """
    Get the best available computing device.
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def memory_usage() -> Dict[str, float]:
    """
    Get current memory usage information.
    
    Returns:
        Dictionary with memory usage stats
    """
    import psutil
    
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        'percent': process.memory_percent()
    }


# Additional functions for counterfactual explainers
# def tv_norm(x: torch.Tensor, beta: int = 3) -> torch.Tensor:
#     """
#     Compute total variation norm for regularization.
    
#     Args:
#         x: Input tensor
#         beta: TV norm parameter
        
#     Returns:
#         TV norm value
#     """
#     if x.dim() == 2:
#         # For 2D tensors (channels, time)
#         diff_time = torch.abs(x[:, 1:] - x[:, :-1])
#         tv = torch.sum(torch.pow(diff_time, beta / beta))
#     elif x.dim() == 3:
#         # For 3D tensors (batch, channels, time)
#         diff_time = torch.abs(x[:, :, 1:] - x[:, :, :-1])
#         tv = torch.sum(torch.pow(diff_time, beta / beta))
#     else:
#         # Fallback for other dimensions
#         tv = torch.sum(torch.pow(torch.abs(x), beta / beta))
    
#     return tv

def tv_norm(signal, tv_beta):
    signal = signal.flatten()
    signal_grad = torch.mean(torch.abs(signal[:-1] - signal[1:]).pow(tv_beta))
    return signal_grad


def save_timeseries_data(
    mask: np.ndarray,
    time_series: np.ndarray,
    perturbated_output: np.ndarray,
    save_dir: str,
    filename_prefix: str = "cf_result"
) -> None:
    """
    Save time series counterfactual results.
    
    Args:
        mask: Explanation mask
        time_series: Original time series
        perturbated_output: Counterfactual time series
        save_dir: Directory to save results
        filename_prefix: Prefix for saved files
    """
    import os
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save arrays
    np.save(os.path.join(save_dir, f"{filename_prefix}_mask.npy"), mask)
    np.save(os.path.join(save_dir, f"{filename_prefix}_original.npy"), time_series)
    np.save(os.path.join(save_dir, f"{filename_prefix}_counterfactual.npy"), perturbated_output)
    
    print(f"Counterfactual results saved to {save_dir}")


class CustomJsonEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling non-serializable objects."""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
