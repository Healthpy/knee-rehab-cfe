"""
Base explainer abstract class for counterfactual explanation methods.

This module provides the foundation for all explainer implementations in the XAI framework.
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
from pathlib import Path

# Import centralized configuration
from .config import (
    get_injury_side, SensorSpecifications, SensorGroups,
    LEFT_INJURY_SUBJECTS, RIGHT_INJURY_SUBJECTS
)


class BaseExplainer(ABC):
    """
    Abstract base class for all counterfactual explainers.
    
    This class defines the common interface and shared functionality for all
    explanation methods in the framework.
    """
    
    def __init__(
        self,
        model_path: str,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the base explainer.
        
        Args:
            model_path: Path to the trained model
            config: Configuration dictionary
            device: Computing device ('cpu', 'cuda', etc.)
        """
        self.model_path = Path(model_path)
        self.config = config or {}
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Model and data attributes
        self.model = None
        self.input_size = self.config.get('input_size', 48)
        self.num_classes = self.config.get('num_classes', 3)
        
        # Load model
        self._load_model()
    
    @abstractmethod
    def _load_model(self) -> None:
        """Load the machine learning model."""
        pass
    
    @abstractmethod
    def explain(
        self,
        data: np.ndarray,
        target_class: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate counterfactual explanations for the given data.
        
        Args:
            data: Input data to explain
            target_class: Target class for counterfactual generation
            **kwargs: Additional parameters specific to the explainer
            
        Returns:
            Dictionary containing explanation results
        """
        pass
    
    def predict(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get model predictions for the given data.
        
        Args:
            data: Input data
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        self.model.eval()
        with torch.no_grad():
            if isinstance(data, np.ndarray):
                data = torch.FloatTensor(data).to(self.device)
            
            if data.dim() == 2:
                data = data.unsqueeze(0)
            
            outputs = self.model(data)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            return predictions.cpu().numpy(), probabilities.cpu().numpy()
    
    def validate_input(self, data: np.ndarray) -> None:
        """
        Validate input data format and shape.
        
        Args:
            data: Input data to validate
            
        Raises:
            ValueError: If data format is invalid
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array")
        
        if data.ndim not in [2, 3]:
            raise ValueError(f"Data must be 2D or 3D, got {data.ndim}D")
        
        if data.shape[-2] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} channels, got {data.shape[-2]}"
            )
    
    def compute_feature_importance(
        self,
        original_data: np.ndarray,
        counterfactual_data: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute feature importance based on counterfactual changes.
        
        Args:
            original_data: Original input data
            counterfactual_data: Generated counterfactual data
            
        Returns:
            Dictionary with importance scores
        """
        # Compute absolute differences
        differences = np.abs(counterfactual_data - original_data)
        
        # Channel-wise importance
        channel_importance = np.mean(differences, axis=-1)
        
        # Sensor-wise importance (8 sensors, 6 channels each)
        sensor_importance = np.mean(
            channel_importance.reshape(-1, 8, 6), axis=-1
        )
        
        return {
            'channel_importance': channel_importance,
            'sensor_importance': sensor_importance,
            'total_change': np.sum(differences)
        }
    
    def get_sensor_names(self) -> List[str]:
        """
        Get standardized sensor names.
        
        Returns:
            List of sensor names
        """
        return [
            'LTh_sensor', 'LSh_sensor', 'LF_sensor', 'LT_sensor',
            'RTh_sensor', 'RSh_sensor', 'RF_sensor', 'RT_sensor'
        ]
    
    def get_channel_names(self) -> List[str]:
        """
        Get standardized channel names.
        
        Returns:
            List of channel names (48 total)
        """
        sensors = self.get_sensor_names()
        axes = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        
        channel_names = []
        for sensor in sensors:
            for axis in axes:
                channel_names.append(f"{sensor}_{axis}")
        
        return channel_names
    
    def save_explanation(
        self,
        explanation: Dict[str, Any],
        output_path: str
    ) -> None:
        """
        Save explanation results to file.
        
        Args:
            explanation: Explanation results
            output_path: Output file path
        """
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_explanation = convert_numpy(explanation)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_explanation, f, indent=2)
        
        self.logger.info(f"Explanation saved to {output_path}")


class CounterfactualMixin:
    """Mixin class providing common counterfactual generation utilities."""
    
    def generate_perturbations(
        self,
        data: np.ndarray,
        perturbation_magnitude: float = 0.1,
        num_perturbations: int = 100
    ) -> np.ndarray:
        """
        Generate random perturbations of the input data.
        
        Args:
            data: Original data
            perturbation_magnitude: Magnitude of perturbations
            num_perturbations: Number of perturbations to generate
            
        Returns:
            Array of perturbed data samples
        """
        perturbations = []
        
        for _ in range(num_perturbations):
            noise = np.random.normal(0, perturbation_magnitude, data.shape)
            perturbed = data + noise
            perturbations.append(perturbed)
        
        return np.array(perturbations)
    
    def optimize_counterfactual(
        self,
        original_data: np.ndarray,
        target_class: int,
        max_iterations: int = 1000,
        learning_rate: float = 0.01
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Optimize counterfactual using gradient-based methods.
        
        Args:
            original_data: Original input data
            target_class: Target class for counterfactual
            max_iterations: Maximum optimization iterations
            learning_rate: Learning rate for optimization
            
        Returns:
            Tuple of (counterfactual_data, optimization_info)
        """
        # This is a template - specific implementations should override
        raise NotImplementedError("Subclasses must implement this method")
