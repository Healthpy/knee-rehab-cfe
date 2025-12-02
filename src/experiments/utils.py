"""
Experiment utilities for reproducibility and setup.
"""

import random
import numpy as np
import torch
from typing import Optional


def set_global_seed(seed: int) -> None:
    """
    Set global random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For reproducible behavior on CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Global random seed set to: {seed}")


def create_experiment_directory(base_dir: str, experiment_name: str) -> str:
    """
    Create directory for experiment results.
    
    Args:
        base_dir: Base directory path
        experiment_name: Name of the experiment
        
    Returns:
        Path to created experiment directory
    """
    import os
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    return exp_dir


def save_experiment_config(config: dict, save_path: str) -> None:
    """
    Save experiment configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration file
    """
    import json
    import os
    
    config_path = os.path.join(save_path, 'experiment_config.json')
    
    # Convert non-serializable objects to strings
    serializable_config = {}
    for key, value in config.items():
        try:
            json.dumps(value)
            serializable_config[key] = value
        except (TypeError, ValueError):
            serializable_config[key] = str(value)
    
    with open(config_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)
    
    print(f"Experiment configuration saved to: {config_path}")


def log_system_info() -> dict:
    """
    Log system information for experiment tracking.
    
    Returns:
        Dictionary with system information
    """
    import platform
    import torch
    
    system_info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        system_info['gpu_name'] = torch.cuda.get_device_name(0)
        system_info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory
    
    return system_info


def validate_experiment_args(args) -> bool:
    """
    Validate experiment arguments for consistency.
    
    Args:
        args: Parsed arguments object
        
    Returns:
        True if arguments are valid
    """
    valid = True
    
    # Check movement type
    if args.movement_type not in ['squat', 'extension', 'gait']:
        print(f"Invalid movement type: {args.movement_type}")
        valid = False
    
    # Check algorithm
    if args.algo not in ['cf', 'cf_adaptive']:
        print(f"Invalid algorithm: {args.algo}")
        valid = False
    
    # Check learning rate
    if args.lr <= 0:
        print(f"Learning rate must be positive: {args.lr}")
        valid = False
    
    # Check iterations
    if args.max_itr <= 0:
        print(f"Max iterations must be positive: {args.max_itr}")
        valid = False
    
    # Check coefficients are non-negative
    coefficients = ['l_budget_coeff', 'l_tv_norm_coeff', 'l_max_coeff', 'l_sparse_coeff']
    for coeff in coefficients:
        if hasattr(args, coeff) and getattr(args, coeff) < 0:
            print(f"Coefficient {coeff} must be non-negative: {getattr(args, coeff)}")
            valid = False
    
    return valid
