"""
Shapley Ranking Utilities for Channel Importance.

This module implements channel importance ranking methods for the Adaptive-MO algorithm:
- DeepSHAP-based ranking (faster than KernelSHAP for neural networks)
- Gradient-based ranking baseline
- Random ranking baseline
- Channel selection utilities

Reference: Adaptive-MO method for counterfactual explanations in time-series data
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
from itertools import combinations
import random
from sklearn.metrics import accuracy_score
import shap

logger = logging.getLogger(__name__)


class ShapleyChannelRanker:
    """
    Channel importance ranking using Shapley values for Adaptive-MO.
    
    This class implements channel ranking methods using SHAP's DeepExplainer
    to determine which IMU channels are most important for model predictions,
    enabling incremental group-based counterfactual search.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        background_data: Optional[torch.Tensor] = None,
        max_evals: int = 1000,
        random_seed: int = 42
    ):
        """
        Initialize Shapley channel ranker.
        
        Args:
            model: Trained PyTorch model for evaluation
            background_data: Background dataset for KernelSHAP (batch_size, channels, timesteps)
            max_evals: Maximum evaluations for Shapley computation
            random_seed: Random seed for reproducibility
        """
        self.model = model
        self.background_data = background_data
        self.max_evals = max_evals
        self.random_seed = random_seed
        
        # Set random seeds
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Cache for computed Shapley values
        self._shapley_cache = {}
        
        logger.info(f"ShapleyChannelRanker initialized with DeepSHAP, max_evals={max_evals}")
    
    def rank_channels_shapley(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
        use_cache: bool = True
    ) -> Tuple[List[int], np.ndarray]:
        """
        Rank channels by importance using DeepSHAP.
        
        Args:
            x: Input sample (channels, timesteps)
            target_class: Target class for explanation (if None, use predicted class)
            use_cache: Whether to use cached Shapley values
            
        Returns:
            Tuple of (ranked_channel_indices, shapley_values)
        """
        # Convert to numpy for SHAP compatibility
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = x
        
        # Create cache key
        cache_key = self._create_cache_key(x_np, target_class)
        
        if use_cache and cache_key in self._shapley_cache:
            logger.debug("Using cached Shapley values")
            return self._shapley_cache[cache_key]
        

        # Compute Shapley values using DeepExplainer
        shapley_values = self._compute_deep_shap(x_np, target_class)
        
        # Aggregate temporal Shapley values per channel
        channel_importance = self._aggregate_channel_importance(shapley_values)
        
        # Rank channels by importance (descending)
        ranked_indices = np.argsort(channel_importance)[::-1].tolist()
        
        # Cache results
        if use_cache:
            self._shapley_cache[cache_key] = (ranked_indices, channel_importance)
        
        # logger.info(f"DeepSHAP ranking completed. Top 5 channels: {ranked_indices[:5]}")
        return ranked_indices, channel_importance
            
    
    def _compute_deep_shap(self, x: np.ndarray, target_class: Optional[int]) -> np.ndarray:
        """
        Compute DeepSHAP values for the input sample using SHAP's DeepExplainer.
        
        Args:
            x: Input sample (channels, timesteps)
            target_class: Target class index
            
        Returns:
            Shapley values array (channels, timesteps)
        """
        # Convert to tensor format
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        x_tensor = x_tensor.to(device)
        
        # Prepare background data for DeepExplainer
        if self.background_data is not None:
            bg_data = self.background_data.to(device)
            # Subsample background data for efficiency (DeepExplainer is faster but still needs reasonable size)
            if bg_data.shape[0] > 100:
                indices = torch.randperm(bg_data.shape[0])[:100]
                bg_data = bg_data[indices]
        else:
            # Create zero background with same shape as input
            bg_data = torch.zeros(1, x.shape[0], x.shape[1], dtype=torch.float32).to(device)
        
        # Initialize DeepSHAP explainer
        explainer = shap.DeepExplainer(self.model, bg_data)
        
        # Compute Shapley values
        self.model.eval()
        # with torch.no_grad():
        shap_values = explainer.shap_values(x_tensor)
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            if target_class is not None:
                # Use specified target class
                shap_values = shap_values[target_class]
            else:
                # Use predicted class
                outputs = self.model(x_tensor)
                predicted_class = torch.argmax(outputs, dim=1).item()
                shap_values = shap_values[predicted_class]
        
        # Convert back to numpy and remove batch dimension
        if isinstance(shap_values, torch.Tensor):
            shap_values = shap_values.cpu().numpy()
        
        # Remove batch dimension: (1, channels, timesteps) -> (channels, timesteps)
        shap_values = shap_values.squeeze(0)
        
        return shap_values
    
    def _aggregate_channel_importance(self, shapley_values: np.ndarray) -> np.ndarray:
        """
        Aggregate temporal Shapley values to get per-channel importance.
        
        Args:
            shapley_values: Shapley values (channels, timesteps)
            
        Returns:
            Channel importance scores (channels,)
        """
        # Use absolute values to capture both positive and negative contributions
        abs_shapley = np.abs(shapley_values)
        
        # Aggregate over time using mean
        channel_importance = np.mean(abs_shapley, axis=-1)
        
        return channel_importance
    
    def rank_channels_random(self, x: torch.Tensor) -> Tuple[List[int], np.ndarray]:
        """
        Random channel ranking baseline.
        
        Args:
            x: Input sample (channels, timesteps)
            
        Returns:
            Tuple of (random_channel_indices, uniform_importance_scores)
        """
        if isinstance(x, torch.Tensor):
            channels = x.shape[0]
        else:
            channels = x.shape[0]
        
        # Create random permutation of channel indices
        channel_indices = list(range(channels))
        random.shuffle(channel_indices)
        
        # Uniform importance scores
        importance_scores = np.ones(channels) / channels
        
        logger.info(f"Random ranking generated. First 5 channels: {channel_indices[:5]}")
        return channel_indices, importance_scores
    
    def rank_channels_variance(self, x: torch.Tensor) -> Tuple[List[int], np.ndarray]:
        """
        Rank channels by signal variance (simple baseline).
        
        Args:
            x: Input sample (channels, timesteps)
            
        Returns:
            Tuple of (variance_ranked_indices, variance_scores)
        """
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = x
        
        # Compute variance per channel
        channel_variances = np.var(x_np, axis=1)
        
        # Rank by variance (descending)
        ranked_indices = np.argsort(channel_variances)[::-1].tolist()
        
        # Normalize variance scores
        variance_scores = channel_variances / np.sum(channel_variances)
        
        logger.info(f"Variance ranking completed. Top 5 channels: {ranked_indices[:5]}")
        return ranked_indices, variance_scores
    
    def select_top_channels(
        self,
        ranked_channels: List[int],
        top_k: int,
        importance_scores: Optional[np.ndarray] = None
    ) -> List[int]:
        """
        Select top-k channels based on ranking.
        
        Args:
            ranked_channels: Channel indices ranked by importance
            top_k: Number of top channels to select
            importance_scores: Channel importance scores (for logging purposes)
            
        Returns:
            List of selected channel indices
        """
        # Select top-k channels
        selected_channels = ranked_channels[:top_k]
        logger.info(f"Selected top-{top_k} channels")
        
        return selected_channels
    
    def create_channel_groups(
        self,
        ranked_channels: List[int],
        group_size: int = 6,
        sensor_groups: Optional[Dict[str, List[int]]] = None
    ) -> List[List[int]]:
        """
        Create channel groups for incremental search.
        
        Args:
            ranked_channels: Channel indices ranked by importance
            group_size: Default group size
            sensor_groups: Predefined sensor groups (if available)
            
        Returns:
            List of channel groups (each group is a list of channel indices)
        """
        if sensor_groups is not None:
            # Use predefined sensor groups, ordered by importance
            groups = []
            used_channels = set()
            
            # Sort sensor groups by average rank of their channels
            group_priorities = []
            for group_name, channels in sensor_groups.items():
                avg_rank = np.mean([ranked_channels.index(ch) if ch in ranked_channels else len(ranked_channels) 
                                 for ch in channels])
                group_priorities.append((avg_rank, group_name, channels))
            
            group_priorities.sort()  # Sort by average rank (ascending)
            
            # Add groups in order of importance
            for _, group_name, channels in group_priorities:
                # Filter out already used channels
                available_channels = [ch for ch in channels if ch not in used_channels]
                if available_channels:
                    groups.append(available_channels)
                    used_channels.update(available_channels)
            
            logger.info(f"Created {len(groups)} sensor-based groups")
        else:
            # Create groups of fixed size from ranked channels
            groups = []
            for i in range(0, len(ranked_channels), group_size):
                group = ranked_channels[i:i + group_size]
                groups.append(group)
            
            logger.info(f"Created {len(groups)} fixed-size groups of size {group_size}")
        
        return groups
    
    def _create_cache_key(self, x: np.ndarray, target_class: Optional[int]) -> str:
        """Create cache key for Shapley computation."""
        x_hash = hash(x.tobytes())
        return f"{x_hash}_{target_class}"
    
    def clear_cache(self):
        """Clear Shapley values cache."""
        self._shapley_cache.clear()
        logger.info("Shapley cache cleared")
    
    def get_cache_size(self) -> int:
        """Get current cache size."""
        return len(self._shapley_cache)


def rank_channels_by_gradient(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_class: int
) -> Tuple[List[int], np.ndarray]:
    """
    Rank channels by gradient-based importance (fast baseline).
    
    Args:
        model: Trained PyTorch model
        x: Input sample (channels, timesteps)
        target_class: Target class index
        
    Returns:
        Tuple of (gradient_ranked_indices, gradient_importance_scores)
    """
    # Get model device
    device = next(model.parameters()).device
    
    # Ensure input requires gradient and is on correct device
    x = x.clone().detach().to(device).requires_grad_(True)
    
    # Add batch dimension if needed
    if len(x.shape) == 2:
        x = x.unsqueeze(0)
    
    # Retain gradients for non-leaf tensor
    x.retain_grad()
    
    try:
        # Forward pass
        model.eval()
        output = model(x)
        
        # Compute gradient with respect to target class
        target_score = output[0, target_class]
        target_score.backward()
        
        # Get gradients
        if x.grad is not None:
            gradients = x.grad.detach().cpu().numpy()
        else:
            logger.warning("No gradients computed, using zero gradients")
            gradients = np.zeros_like(x.detach().cpu().numpy())
        
        # Remove batch dimension and compute channel importance
        gradients = gradients.squeeze(0)  # (channels, timesteps)
        channel_importance = np.mean(np.abs(gradients), axis=1)  # (channels,)
        
        # Rank channels by importance (descending)
        ranked_indices = np.argsort(channel_importance)[::-1].tolist()
        
        return ranked_indices, channel_importance
        
    except Exception as e:
        logger.warning(f"Gradient computation failed: {e}. Using variance-based ranking.")
        # Fallback to variance-based ranking
        x_np = x.detach().cpu().numpy().squeeze(0)
        channel_variances = np.var(x_np, axis=1)
        ranked_indices = np.argsort(channel_variances)[::-1].tolist()
        variance_scores = channel_variances / (np.sum(channel_variances) + 1e-8)
        return ranked_indices, variance_scores


def create_default_sensor_groups(group_level: str = 'sensor') -> Dict[str, List[int]]:
    """
    Create default sensor or modality groups for 48-channel IMU data.
    
    Args:
        group_level: Either 'sensor' (8 groups) or 'modality' (16 groups)
    
    Returns:
        Dictionary mapping group names to channel indices
    """
    if group_level == 'sensor':
        # 8 sensor-level groups (6 channels each)
        return {
            'R_RF': list(range(0, 6)),       # Right Rectus Femoris
            'R_HAM': list(range(6, 12)),     # Right Hamstring  
            'R_TA': list(range(12, 18)),     # Right Tibialis Anterior
            'R_GAS': list(range(18, 24)),    # Right Gastrocnemius
            'L_RF': list(range(24, 30)),     # Left Rectus Femoris
            'L_HAM': list(range(30, 36)),    # Left Hamstring
            'L_TA': list(range(36, 42)),     # Left Tibialis Anterior
            'L_GAS': list(range(42, 48)),    # Left Gastrocnemius
        }
    elif group_level == 'modality':
        # 16 modality-level groups (3 channels each)
        groups = {}
        sensor_names = ['R_RF', 'R_HAM', 'R_TA', 'R_GAS', 'L_RF', 'L_HAM', 'L_TA', 'L_GAS']
        # modality_names = ['acc', 'gyr']
        
        for i, sensor in enumerate(sensor_names):
            base_idx = i * 6
            groups[f'{sensor}_acc'] = list(range(base_idx, base_idx + 3))      # Accelerometer
            groups[f'{sensor}_gyr'] = list(range(base_idx + 3, base_idx + 6))   # Gyroscope
        
        return groups
    else:
        raise ValueError(f"Invalid group_level: {group_level}. Must be 'sensor' or 'modality'")


def evaluate_channel_ranking(
    model: torch.nn.Module,
    x: torch.Tensor,
    y_true: int,
    ranked_channels: List[int],
    incremental_steps: List[int] = None
) -> Dict[str, List[float]]:
    """
    Evaluate channel ranking quality by incremental feature addition.
    
    Args:
        model: Trained PyTorch model
        x: Input sample (channels, timesteps)
        y_true: True class label
        ranked_channels: Channel indices ranked by importance
        incremental_steps: List of channel counts to evaluate
        
    Returns:
        Dictionary with evaluation metrics at each step
    """
    if incremental_steps is None:
        incremental_steps = [1, 2, 4, 8, 12, 16, 24, 32, 48]
    
    # Ensure we don't exceed available channels
    max_channels = min(len(ranked_channels), x.shape[0])
    incremental_steps = [s for s in incremental_steps if s <= max_channels]
    
    # Add batch dimension if needed
    if len(x.shape) == 2:
        x = x.unsqueeze(0)
    
    results = {
        'num_channels': [],
        'accuracy': [],
        'confidence': [],
        'prediction': []
    }
    
    model.eval()
    with torch.no_grad():
        for num_channels in incremental_steps:
            # Select top channels
            selected_channels = ranked_channels[:num_channels]
            
            # Create masked input (zero out non-selected channels)
            x_masked = x.clone()
            all_channels = set(range(x.shape[1]))
            excluded_channels = list(all_channels - set(selected_channels))
            
            if excluded_channels:
                x_masked[:, excluded_channels, :] = 0
            
            # Get prediction
            output = model(x_masked)
            probs = torch.softmax(output, dim=1)
            
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, predicted_class].item()
            accuracy = 1.0 if predicted_class == y_true else 0.0
            
            # Store results
            results['num_channels'].append(num_channels)
            results['accuracy'].append(accuracy)
            results['confidence'].append(confidence)
            results['prediction'].append(predicted_class)
    
    return results