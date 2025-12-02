"""
Data loading and preprocessing utilities for IMU and EMG data.

This module provides comprehensive data loading capabilities for movement analysis,
including support for different movement types and data formats.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from src.core.utils import validate_data_shape, ensure_dir


class MovementDataLoader:
    """
    Comprehensive data loader for movement analysis data.
    
    Supports loading IMU and EMG data for different movement types
    with proper validation and preprocessing.
    """
    
    SUPPORTED_MOVEMENTS = ["squat", "extension", "gait"]
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing movement data files
            config: Configuration dictionary
        """
        self.data_dir = Path(data_dir)
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Data specifications
        self.imu_channels = self.config.get('imu_channels', 48)
        self.imu_sensors = self.config.get('imu_sensors', 8)
        self.channels_per_sensor = self.config.get('channels_per_sensor', 6)
        
        # Validate data directory
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
    
    def load_movement_data(
        self,
        movement_type: str,
        include_emg: bool = True,
        validate: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Load train and test data for a specific movement type.
        
        Args:
            movement_type: Type of movement ('squat', 'extension', 'gait')
            include_emg: Whether to include EMG data
            validate: Whether to validate data shapes
            
        Returns:
            Dictionary containing loaded data arrays
            
        Raises:
            ValueError: If movement type is not supported
            FileNotFoundError: If data files are not found
        """
        if movement_type not in self.SUPPORTED_MOVEMENTS:
            raise ValueError(
                f"Unsupported movement type: {movement_type}. "
                f"Supported types: {self.SUPPORTED_MOVEMENTS}"
            )
        
        # Construct file paths
        train_path = self.data_dir / f"{movement_type}_train.npz"
        test_path = self.data_dir / f"{movement_type}_test.npz"
        
        # Check file existence
        missing_files = []
        if not train_path.exists():
            missing_files.append(str(train_path))
        if not test_path.exists():
            missing_files.append(str(test_path))
        
        if missing_files:
            raise FileNotFoundError(
                f"Missing data files: {missing_files}. "
                f"Please ensure {movement_type} data is available in {self.data_dir}"
            )
        
        # Load data
        self.logger.info(f"Loading {movement_type} movement data...")
        
        train_data = np.load(train_path)
        test_data = np.load(test_path)
        
        # Extract data arrays
        data_dict = {
            'X_train': train_data['X_train_imu'],
            'y_train': train_data['y_train'],
            'subjects_train': train_data['subjects'],
            'X_test': test_data['X_test_imu'],
            'y_test': test_data['y_test'],
            'subjects_test': test_data['subjects']
        }
        
        # Add EMG data if requested and available
        if include_emg:
            if 'X_train_emg' in train_data:
                data_dict['X_train_emg'] = train_data['X_train_emg']
            if 'X_test_emg' in test_data:
                data_dict['X_test_emg'] = test_data['X_test_emg']
        
        # Validate data if requested
        if validate:
            self._validate_loaded_data(data_dict, movement_type)
        
        # Log data information
        self._log_data_info(data_dict, movement_type)
        
        return data_dict
    
    def _validate_loaded_data(
        self,
        data_dict: Dict[str, np.ndarray],
        movement_type: str
    ) -> None:
        """
        Validate loaded data arrays.
        
        Args:
            data_dict: Dictionary of loaded data
            movement_type: Movement type for context
            
        Raises:
            ValueError: If data validation fails
        """
        # Check required keys
        required_keys = ['X_train', 'y_train', 'subjects_train', 
                        'X_test', 'y_test', 'subjects_test']
        
        missing_keys = [key for key in required_keys if key not in data_dict]
        if missing_keys:
            raise ValueError(f"Missing required data keys: {missing_keys}")
        
        # Validate IMU data shapes
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        
        if X_train.shape[1] != self.imu_channels:
            raise ValueError(
                f"Expected {self.imu_channels} IMU channels, "
                f"got {X_train.shape[1]} in training data"
            )
        
        if X_test.shape[1] != self.imu_channels:
            raise ValueError(
                f"Expected {self.imu_channels} IMU channels, "
                f"got {X_test.shape[1]} in test data"
            )
        
        # Validate label consistency
        train_samples = len(data_dict['X_train'])
        test_samples = len(data_dict['X_test'])
        
        if len(data_dict['y_train']) != train_samples:
            raise ValueError("Training data and labels have different lengths")
        
        if len(data_dict['y_test']) != test_samples:
            raise ValueError("Test data and labels have different lengths")
        
        if len(data_dict['subjects_train']) != train_samples:
            raise ValueError("Training data and subjects have different lengths")
        
        if len(data_dict['subjects_test']) != test_samples:
            raise ValueError("Test data and subjects have different lengths")
        
        self.logger.info(f"Data validation passed for {movement_type}")
    
    def _log_data_info(
        self,
        data_dict: Dict[str, np.ndarray],
        movement_type: str
    ) -> None:
        """
        Log information about loaded data.
        
        Args:
            data_dict: Dictionary of loaded data
            movement_type: Movement type
        """
        train_samples = len(data_dict['X_train'])
        test_samples = len(data_dict['X_test'])
        num_classes = len(np.unique(data_dict['y_train']))
        
        self.logger.info(f"Movement: {movement_type}")
        self.logger.info(f"Classes: {num_classes}")
        self.logger.info(f"Train samples: {train_samples}")
        self.logger.info(f"Test samples: {test_samples}")
        
        if 'X_train_emg' in data_dict:
            self.logger.info(f"EMG data included")
        
        # Log class distribution
        unique_train, counts_train = np.unique(data_dict['y_train'], return_counts=True)
        unique_test, counts_test = np.unique(data_dict['y_test'], return_counts=True)
        
        self.logger.info(f"Train class distribution: {dict(zip(unique_train, counts_train))}")
        self.logger.info(f"Test class distribution: {dict(zip(unique_test, counts_test))}")
    
    def get_subject_data(
        self,
        data_dict: Dict[str, np.ndarray],
        subject_id: int,
        split: str = 'train'
    ) -> Dict[str, np.ndarray]:
        """
        Extract data for a specific subject.
        
        Args:
            data_dict: Dictionary of loaded data
            subject_id: Subject identifier
            split: Data split ('train' or 'test')
            
        Returns:
            Dictionary containing subject-specific data
        """
        if split not in ['train', 'test']:
            raise ValueError("Split must be 'train' or 'test'")
        
        # Get subject indices
        subjects_key = f'subjects_{split}'
        if subjects_key not in data_dict:
            raise KeyError(f"Subject data not found: {subjects_key}")
        
        subject_mask = data_dict[subjects_key] == subject_id
        
        if not np.any(subject_mask):
            raise ValueError(f"Subject {subject_id} not found in {split} data")
        
        # Extract subject data
        result = {}
        for key, value in data_dict.items():
            if key.startswith(f'X_{split}') or key.startswith(f'y_{split}'):
                result[key] = value[subject_mask]
        
        result[subjects_key] = data_dict[subjects_key][subject_mask]
        
        return result
    
    def get_class_data(
        self,
        data_dict: Dict[str, np.ndarray],
        class_label: int,
        split: str = 'train'
    ) -> Dict[str, np.ndarray]:
        """
        Extract data for a specific class.
        
        Args:
            data_dict: Dictionary of loaded data
            class_label: Class label
            split: Data split ('train' or 'test')
            
        Returns:
            Dictionary containing class-specific data
        """
        if split not in ['train', 'test']:
            raise ValueError("Split must be 'train' or 'test'")
        
        # Get class indices
        labels_key = f'y_{split}'
        if labels_key not in data_dict:
            raise KeyError(f"Label data not found: {labels_key}")
        
        class_mask = data_dict[labels_key] == class_label
        
        if not np.any(class_mask):
            raise ValueError(f"Class {class_label} not found in {split} data")
        
        # Extract class data
        result = {}
        for key, value in data_dict.items():
            if key.endswith(f'_{split}'):
                result[key] = value[class_mask]
        
        return result
    
    def list_available_movements(self) -> List[str]:
        """
        List available movement types in the data directory.
        
        Returns:
            List of available movement types
        """
        available = []
        
        for movement in self.SUPPORTED_MOVEMENTS:
            train_path = self.data_dir / f"{movement}_train.npz"
            test_path = self.data_dir / f"{movement}_test.npz"
            
            if train_path.exists() and test_path.exists():
                available.append(movement)
        
        return available
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of available data.
        
        Returns:
            Dictionary with data summary
        """
        summary = {
            'data_directory': str(self.data_dir),
            'available_movements': self.list_available_movements(),
            'supported_movements': self.SUPPORTED_MOVEMENTS,
            'imu_channels': self.imu_channels,
            'imu_sensors': self.imu_sensors
        }
        
        # Get detailed info for each available movement
        movement_details = {}
        for movement in summary['available_movements']:
            try:
                data = self.load_movement_data(movement, validate=False)
                movement_details[movement] = {
                    'train_samples': len(data['X_train']),
                    'test_samples': len(data['X_test']),
                    'num_classes': len(np.unique(data['y_train'])),
                    'has_emg': 'X_train_emg' in data
                }
            except Exception as e:
                movement_details[movement] = {'error': str(e)}
        
        summary['movement_details'] = movement_details
        
        return summary


class DataPreprocessor:
    """Preprocessor for IMU and EMG data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def normalize_imu_data(
        self,
        data: np.ndarray,
        method: str = 'standardize',
        per_channel: bool = True
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Normalize IMU data.
        
        Args:
            data: IMU data array (samples, channels, time)
            method: Normalization method ('standardize', 'minmax')
            per_channel: Whether to normalize per channel
            
        Returns:
            Tuple of (normalized_data, normalization_params)
        """
        if method == 'standardize':
            axis = (0, 2) if per_channel else None
            mean = np.mean(data, axis=axis, keepdims=True)
            std = np.std(data, axis=axis, keepdims=True)
            
            normalized = (data - mean) / (std + 1e-8)
            params = {'mean': mean, 'std': std, 'method': method}
        
        elif method == 'minmax':
            axis = (0, 2) if per_channel else None
            min_val = np.min(data, axis=axis, keepdims=True)
            max_val = np.max(data, axis=axis, keepdims=True)
            
            normalized = (data - min_val) / (max_val - min_val + 1e-8)
            params = {'min': min_val, 'max': max_val, 'method': method}
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized, params
    
    def apply_normalization(
        self,
        data: np.ndarray,
        params: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Apply previously computed normalization parameters.
        
        Args:
            data: Data to normalize
            params: Normalization parameters
            
        Returns:
            Normalized data
        """
        method = params['method']
        
        if method == 'standardize':
            return (data - params['mean']) / (params['std'] + 1e-8)
        elif method == 'minmax':
            return (data - params['min']) / (params['max'] - params['min'] + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def segment_data(
        self,
        data: np.ndarray,
        window_size: int,
        overlap: float = 0.5
    ) -> np.ndarray:
        """
        Segment time series data into overlapping windows.
        
        Args:
            data: Input data (samples, channels, time)
            window_size: Size of each window
            overlap: Overlap fraction between windows
            
        Returns:
            Segmented data array
        """
        if data.ndim != 3:
            raise ValueError("Data must be 3D (samples, channels, time)")
        
        step_size = int(window_size * (1 - overlap))
        segments = []
        
        for sample in data:
            sample_segments = []
            for start in range(0, sample.shape[1] - window_size + 1, step_size):
                end = start + window_size
                segment = sample[:, start:end]
                sample_segments.append(segment)
            
            if sample_segments:
                segments.extend(sample_segments)
        
        return np.array(segments)


# Convenience function for backward compatibility with main_copy.py
def load_movement_data(movement_type: str, data_dir: str = "data") -> Dict[str, np.ndarray]:
    """
    Load movement data for a specific movement type.
    
    Args:
        movement_type: Type of movement ("squat", "extension", "gait")
        data_dir: Directory containing movement data
        
    Returns:
        Dictionary containing train/test data and labels
    """
    data_path = Path(data_dir)
    
    # File paths
    train_path = data_path / f"{movement_type}_train.npz"
    test_path = data_path / f"{movement_type}_test.npz"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")
    
    # Load data
    train_data = np.load(train_path)
    test_data = np.load(test_path)
    
    return {
        'X_train': train_data['X_train_imu'],
        'y_train': train_data['y_train'],
        'subjects_train': train_data['subjects'],
        'X_test': test_data['X_test_imu'],
        'y_test': test_data['y_test'],
        'subjects_test': test_data['subjects'],
        'X_train_emg': train_data['X_train_emg'],
        'X_test_emg': test_data['X_test_emg']
    }
