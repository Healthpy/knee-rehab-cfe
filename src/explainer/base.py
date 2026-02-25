"""
Base Saliency Model Interface

Abstract base class for all explainer methods.
"""

from abc import ABCMeta, abstractmethod
import torch
import numpy as np
from typing import Any, Dict, Optional


class Saliency(metaclass=ABCMeta):
    """
    Abstract base class for saliency/explainability methods
    
    This provides a common interface for different explanation techniques
    including counterfactual generation, saliency maps, and attribution methods.
    """
    
    def __init__(
        self,
        background_data: np.ndarray,
        background_label: np.ndarray,
        predict_fn: callable
    ):
        """
        Initialize saliency explainer
        
        Args:
            background_data: Reference dataset [N, C, T] or [N, T, C]
            background_label: Labels for background data [N]
            predict_fn: Model prediction function
        """
        self.background_data = background_data
        self.background_label = background_label
        self.predict_fn = predict_fn
        self.perturbation_manager = None
    
    @abstractmethod
    def generate_saliency(
        self,
        data: torch.Tensor,
        label: int,
        **kwargs
    ) -> tuple:
        """
        Generate saliency/explanation for input data
        
        Args:
            data: Input time series [C, T]
            label: True label
            **kwargs: Additional arguments
            
        Returns:
            mask: Saliency/importance mask
            perturbated: Perturbed/counterfactual data
            confidence: Confidence or probability
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(background_data={self.background_data.shape})"
