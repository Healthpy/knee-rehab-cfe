"""
Loss Functions for MCELS (Mask-based Counterfactual Explanation with Local Search).

This module implements the specialized loss functions required for the MCELS method:
- Maximize Loss (L_maximize) - Validity loss for target class
- Budget Loss (L_budget) - Sparsity constraint on mask
- Total Variation Loss (L_tv) - Smoothness constraint on mask

Reference: MCELS method for counterfactual explanations in time-series data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class MCELSLoss(nn.Module):
    """
    Combined loss function for MCELS counterfactual generation.
    
    Loss = λ_max * L_maximize + λ_budget * L_budget + λ_tv * L_tv_norm
    """
    
    def __init__(
        self,
        lambda_max: float = 0.7,
        lambda_budget: float = 0.6,
        lambda_tv: float = 0.5,
        tv_beta: int = 3,
        enable_budget: bool = True,
        enable_tvnorm: bool = True,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize MCELS loss function.
        
        Args:
            lambda_max: Weight for maximize loss (validity)
            lambda_budget: Weight for budget loss (sparsity)
            lambda_tv: Weight for total variation loss (smoothness)
            tv_beta: Power parameter for TV norm computation
            enable_budget: Whether to include budget loss
            enable_tvnorm: Whether to include TV norm loss
            confidence_threshold: Minimum confidence threshold for validity
        """
        super().__init__()
        self.lambda_max = lambda_max
        self.lambda_budget = lambda_budget
        self.lambda_tv = lambda_tv
        self.tv_beta = tv_beta
        self.enable_budget = enable_budget
        self.enable_tvnorm = enable_tvnorm
        self.confidence_threshold = confidence_threshold
        
        logger.info(f"MCELS Loss initialized with λ=[{lambda_max}, {lambda_budget}, {lambda_tv}], tv_beta={tv_beta}")
    
    def forward(
        self,
        mask: torch.Tensor,
        model_output: torch.Tensor,
        target_class: int
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined MCELS loss.
        
        Args:
            mask: Mask tensor for perturbation (batch_size, channels, timesteps)
            model_output: Model predictions (batch_size, num_classes) 
            target_class: Target class index
            
        Returns:
            Dictionary containing individual loss components and total loss
        """
        # Apply softmax to get probabilities
        probs = F.softmax(model_output, dim=1)
        
        # Compute individual loss components
        l_maximize = self.maximize_loss(probs, target_class)
        l_budget = self.budget_loss(mask) if self.enable_budget else torch.tensor(0.0, device=mask.device)
        l_tv_norm = self.tv_norm_loss(mask) if self.enable_tvnorm else torch.tensor(0.0, device=mask.device)
        
        # Combined loss
        total_loss = (
            self.lambda_max * l_maximize +
            self.lambda_budget * l_budget +
            self.lambda_tv * l_tv_norm
        )
        
        return {
            'total': total_loss,
            'maximize': l_maximize,
            'budget': l_budget,
            'tv_norm': l_tv_norm
        }
    
    def maximize_loss(self, probs: torch.Tensor, target_class: int) -> torch.Tensor:
        """
        Compute maximize loss: L_maximize = 1 - P(target_class).
        
        This loss encourages the model to predict the target class with high confidence.
        
        Args:
            probs: Model output probabilities (batch_size, num_classes)
            target_class: Target class index
            
        Returns:
            Maximize loss scalar
        """
        # Extract target class probability
        target_prob = probs[:, target_class]
        
        # Maximize loss: minimize (1 - target_probability)
        maximize_loss = 1.0 - target_prob.mean()
        
        return maximize_loss
    
    def budget_loss(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute budget loss: L_budget = mean(|mask|).
        
        This loss encourages sparsity by penalizing large mask values.
        
        Args:
            mask: Mask tensor (batch_size, channels, timesteps)
            
        Returns:
            Budget loss scalar
        """
        # L1 norm of mask values encourages sparsity
        budget_loss = torch.mean(torch.abs(mask))
        
        return budget_loss
    
    def tv_norm_loss(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute total variation norm loss for mask smoothness.
        
        This loss encourages temporal smoothness in the mask by penalizing
        large differences between consecutive timesteps.
        
        Args:
            mask: Mask tensor (batch_size, channels, timesteps)
            
        Returns:
            TV norm loss scalar
        """
        # Flatten mask to 1D for each sample
        batch_size = mask.shape[0]
        tv_losses = []
        
        for i in range(batch_size):
            mask_flat = mask[i].flatten()
            
            if len(mask_flat) <= 1:
                tv_losses.append(torch.tensor(0.0, device=mask.device))
                continue
                
            # Compute differences between consecutive elements
            mask_grad = torch.abs(mask_flat[:-1] - mask_flat[1:])
            
            # Apply power and take mean
            tv_loss = torch.mean(mask_grad.pow(self.tv_beta))
            tv_losses.append(tv_loss)
        
        # Average over batch
        return torch.stack(tv_losses).mean()
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get current loss weights."""
        return {
            'maximize': self.lambda_max,
            'budget': self.lambda_budget, 
            'tv_norm': self.lambda_tv
        }
    
    def set_loss_weights(self, weights: Dict[str, float]):
        """Update loss weights."""
        self.lambda_max = weights.get('maximize', self.lambda_max)
        self.lambda_budget = weights.get('budget', self.lambda_budget)
        self.lambda_tv = weights.get('tv_norm', self.lambda_tv)
        
        logger.info(f"Updated MCELS loss weights: {self.get_loss_weights()}")
    
    def get_configuration(self) -> Dict[str, Union[float, int, bool]]:
        """Get current configuration."""
        return {
            'lambda_max': self.lambda_max,
            'lambda_budget': self.lambda_budget,
            'lambda_tv': self.lambda_tv,
            'tv_beta': self.tv_beta,
            'enable_budget': self.enable_budget,
            'enable_tvnorm': self.enable_tvnorm,
            'confidence_threshold': self.confidence_threshold
        }


class MCELSGuideBasedLoss(MCELSLoss):
    """
    Extended MCELS loss that incorporates guide-based perturbation.
    
    This version computes losses based on the perturbated input created
    by blending original data with guide examples using the mask.
    """
    
    def __init__(
        self,
        lambda_max: float = 0.7,
        lambda_budget: float = 0.6,
        lambda_tv: float = 0.5,
        lambda_guide: float = 0.1,
        tv_beta: int = 3,
        enable_budget: bool = True,
        enable_tvnorm: bool = True,
        enable_guide_loss: bool = True,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize extended MCELS loss with guide-based components.
        
        Args:
            lambda_guide: Weight for guide consistency loss
            enable_guide_loss: Whether to include guide consistency loss
            **kwargs: Arguments passed to parent MCELSLoss
        """
        super().__init__(
            lambda_max=lambda_max,
            lambda_budget=lambda_budget,
            lambda_tv=lambda_tv,
            tv_beta=tv_beta,
            enable_budget=enable_budget,
            enable_tvnorm=enable_tvnorm,
            confidence_threshold=confidence_threshold
        )
        
        self.lambda_guide = lambda_guide
        self.enable_guide_loss = enable_guide_loss
        
        logger.info(f"Extended MCELS Loss initialized with guide weight: {lambda_guide}")
    
    def forward(
        self,
        mask: torch.Tensor,
        model_output: torch.Tensor,
        target_class: int,
        original_data: Optional[torch.Tensor] = None,
        guide_data: Optional[torch.Tensor] = None,
        perturbated_data: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute extended MCELS loss with guide-based components.
        
        Args:
            mask: Mask tensor (batch_size, channels, timesteps)
            model_output: Model predictions (batch_size, num_classes)
            target_class: Target class index
            original_data: Original input data (batch_size, channels, timesteps)
            guide_data: Guide example data (batch_size, channels, timesteps)
            perturbated_data: Perturbated input data (batch_size, channels, timesteps)
            
        Returns:
            Dictionary containing all loss components and total loss
        """
        # Get base MCELS losses
        base_losses = super().forward(mask, model_output, target_class)
        
        # Add guide consistency loss if enabled and data is provided
        l_guide = torch.tensor(0.0, device=mask.device)
        
        if (self.enable_guide_loss and 
            original_data is not None and 
            guide_data is not None and 
            perturbated_data is not None):
            l_guide = self.guide_consistency_loss(mask, original_data, guide_data, perturbated_data)
        
        # Update total loss
        total_loss = (
            base_losses['total'] +
            self.lambda_guide * l_guide
        )
        
        # Add guide loss to results
        base_losses['guide'] = l_guide
        base_losses['total'] = total_loss
        
        return base_losses
    
    def guide_consistency_loss(
        self,
        mask: torch.Tensor,
        original_data: torch.Tensor,
        guide_data: torch.Tensor,
        perturbated_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute guide consistency loss.
        
        This loss ensures that the perturbated data is consistent with the
        mask-based blending of original and guide data.
        
        Args:
            mask: Mask tensor (batch_size, channels, timesteps)
            original_data: Original input (batch_size, channels, timesteps)
            guide_data: Guide example (batch_size, channels, timesteps)
            perturbated_data: Actual perturbated input (batch_size, channels, timesteps)
            
        Returns:
            Guide consistency loss scalar
        """
        # Expected perturbated data based on mask
        expected_perturbated = original_data * (1 - mask) + guide_data * mask
        
        # L2 distance between expected and actual perturbated data
        consistency_loss = torch.mean((perturbated_data - expected_perturbated) ** 2)
        
        return consistency_loss
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get current loss weights including guide weight."""
        weights = super().get_loss_weights()
        weights['guide'] = self.lambda_guide
        return weights
    
    def set_loss_weights(self, weights: Dict[str, float]):
        """Update loss weights including guide weight."""
        super().set_loss_weights(weights)
        self.lambda_guide = weights.get('guide', self.lambda_guide)


# Standalone utility functions
def compute_maximize_loss(model_output: torch.Tensor, target_class: int) -> torch.Tensor:
    """
    Standalone function to compute maximize loss.
    
    Args:
        model_output: Model predictions (batch_size, num_classes)
        target_class: Target class index
        
    Returns:
        Maximize loss scalar
    """
    probs = F.softmax(model_output, dim=1)
    target_prob = probs[:, target_class]
    return 1.0 - target_prob.mean()


def compute_budget_loss(mask: torch.Tensor) -> torch.Tensor:
    """
    Standalone function to compute budget loss.
    
    Args:
        mask: Mask tensor (batch_size, channels, timesteps)
        
    Returns:
        Budget loss scalar
    """
    return torch.mean(torch.abs(mask))


def compute_tv_norm_loss(mask: torch.Tensor, tv_beta: int = 3) -> torch.Tensor:
    """
    Standalone function to compute TV norm loss.
    
    Args:
        mask: Mask tensor (batch_size, channels, timesteps)
        tv_beta: Power parameter for TV norm
        
    Returns:
        TV norm loss scalar
    """
    batch_size = mask.shape[0]
    tv_losses = []
    
    for i in range(batch_size):
        mask_flat = mask[i].flatten()
        
        if len(mask_flat) <= 1:
            tv_losses.append(torch.tensor(0.0, device=mask.device))
            continue
            
        mask_grad = torch.abs(mask_flat[:-1] - mask_flat[1:])
        tv_loss = torch.mean(mask_grad.pow(tv_beta))
        tv_losses.append(tv_loss)
    
    return torch.stack(tv_losses).mean()


def create_perturbated_input(
    original_data: torch.Tensor,
    guide_data: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Create perturbated input using mask-based blending.
    
    Args:
        original_data: Original input (batch_size, channels, timesteps)
        guide_data: Guide example (batch_size, channels, timesteps)
        mask: Mask tensor (batch_size, channels, timesteps)
        
    Returns:
        Perturbated input tensor
    """
    return original_data * (1 - mask) + guide_data * mask


def evaluate_counterfactual(
    model_output: torch.Tensor,
    target_class: int,
    confidence_threshold: float = 0.5
) -> Dict[str, Union[bool, float, int]]:
    """
    Evaluate counterfactual quality.
    
    Args:
        model_output: Model predictions (batch_size, num_classes)
        target_class: Target class index
        confidence_threshold: Minimum confidence for success
        
    Returns:
        Dictionary with evaluation metrics
    """
    probs = F.softmax(model_output, dim=1)
    predicted_class = torch.argmax(probs, dim=1)
    target_prob = probs[:, target_class]
    
    # Check if target class is predicted with sufficient confidence
    success = (predicted_class == target_class).all() and (target_prob >= confidence_threshold).all()
    
    return {
        'success': success.item(),
        'predicted_class': predicted_class[0].item() if len(predicted_class) == 1 else predicted_class.tolist(),
        'target_probability': target_prob.mean().item(),
        'max_probability': probs.max(dim=1)[0].mean().item(),
        'confidence_met': (target_prob >= confidence_threshold).all().item()
    }


class MCELSLossTracker:
    """
    Utility class for tracking MCELS loss components during optimization.
    """
    
    def __init__(self):
        """Initialize loss tracker."""
        self.losses = {
            'total': [],
            'maximize': [],
            'budget': [],
            'tv_norm': [],
            'guide': [],
            'target_prob': [],
            'iteration': []
        }
    
    def update(
        self,
        loss_dict: Dict[str, torch.Tensor],
        target_prob: float,
        iteration: int
    ):
        """
        Update tracked losses.
        
        Args:
            loss_dict: Dictionary of loss components
            target_prob: Target class probability
            iteration: Current iteration number
        """
        for key, value in loss_dict.items():
            if key in self.losses:
                if isinstance(value, torch.Tensor):
                    self.losses[key].append(value.item())
                else:
                    self.losses[key].append(value)
        
        self.losses['target_prob'].append(target_prob)
        self.losses['iteration'].append(iteration)
    
    def get_latest(self) -> Dict[str, float]:
        """Get latest loss values."""
        return {key: values[-1] if values else 0.0 for key, values in self.losses.items()}
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get loss summary statistics."""
        summary = {}
        for key, values in self.losses.items():
            if values:
                summary[key] = {
                    'final': values[-1],
                    'min': min(values),
                    'max': max(values),
                    'mean': sum(values) / len(values)
                }
            else:
                summary[key] = {'final': 0.0, 'min': 0.0, 'max': 0.0, 'mean': 0.0}
        
        return summary
    
    def reset(self):
        """Reset all tracked losses."""
        for key in self.losses:
            self.losses[key].clear()
    