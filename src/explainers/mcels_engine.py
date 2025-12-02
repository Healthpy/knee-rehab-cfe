"""
MCELS Counterfactual Explainer with Standardized Configuration.

This module implements the MCELS (Mask-based Counterfactual Explanation with Local Search)
algorithm for time series data with consistent configuration management.
"""

import logging
from collections import defaultdict
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR
from tslearn.neighbors import KNeighborsTimeSeries

from ..core.config import (
    AlgorithmConstants, 
    SensorSpecifications, 
    VisualizationConfig
)
from ..core.utils import tv_norm
from .saliency import Saliency

logger = logging.getLogger(__name__)


class MCELSExplainer(Saliency):
    """
    MCELS Counterfactual Explainer for time series data.
    
    This explainer generates counterfactual explanations using mask optimization
    with guide retrieval from background data. It follows the MCELS algorithm
    which learns a mask to transform input data toward a target class.
    """
    
    def __init__(
        self,
        background_data: np.ndarray,
        background_label: np.ndarray,
        predict_fn,
        enable_wandb: bool = False,
        use_cuda: bool = False,
        args: Optional[object] = None
    ):
        """
        Initialize MCELS explainer with standardized configuration.
        
        Args:
            background_data: Background dataset for guide retrieval (N, C, T)
            background_label: Labels for background data (N,)
            predict_fn: Model prediction function
            enable_wandb: Whether to enable Weights & Biases logging
            use_cuda: Whether to use CUDA acceleration
            args: Arguments object with training parameters
            
        Raises:
            ValueError: If background data dimensions are invalid
        """
        super(MCELSExplainer, self).__init__(
            background_data=background_data,
            background_label=background_label,
            predict_fn=predict_fn,
        )
        
        # Validate input dimensions
        if background_data.ndim != 3:
            raise ValueError(f"Expected 3D background data (N, C, T), got {background_data.ndim}D")
        
        if background_data.shape[1] != SensorSpecifications.TOTAL_CHANNELS:
            logger.warning(
                f"Background data has {background_data.shape[1]} channels, "
                f"expected {SensorSpecifications.TOTAL_CHANNELS}"
            )
        
        self.enable_wandb = enable_wandb
        self.use_cuda = use_cuda
        self.args = args
        
        # Initialize PyTorch components
        self.softmax_fn = torch.nn.Softmax(dim=-1)
        self.perturbation_manager = None
        
        # Use centralized algorithm constants
        self.conf_threshold = AlgorithmConstants.CONFIDENCE_THRESHOLD
        self.eps = AlgorithmConstants.INITIAL_EPS
        self.eps_decay = AlgorithmConstants.EPS_DECAY
        self.min_target_prob = AlgorithmConstants.MIN_TARGET_PROBABILITY
        self.mask_threshold = AlgorithmConstants.MASK_THRESHOLD
        
        # Set random seeds for reproducibility
        if hasattr(args, 'seed'):
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
        
        logger.info(f"Initialized MCELSExplainer with {len(background_data)} background samples")
    
    def native_guide_retrieval(
        self,
        query: np.ndarray,
        target_label: int,
        distance: str = "euclidean",
        n_neighbors: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve guide examples from background data for target class.
        
        Args:
            query: Query time series of shape (C, T)
            target_label: Target class label
            distance: Distance metric for KNN search
            n_neighbors: Number of nearest neighbors to retrieve
            
        Returns:
            Tuple of (distances, indices) of nearest neighbors
            
        Raises:
            ValueError: If no examples found for target label
        """
        if query.ndim != 2:
            raise ValueError(f"Expected 2D query (C, T), got {query.ndim}D")
        
        dim_nums, ts_length = query.shape[0], query.shape[1]
        df = pd.DataFrame(self.background_label, columns=['label'])
        
        # Get examples of target class
        target_indices = df[df['label'] == target_label].index.values
        if len(target_indices) == 0:
            raise ValueError(f"No examples found for target label {target_label}")
        
        target_data = self.background_data[target_indices]
        
        try:
            knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric=distance)
            knn.fit(target_data)
            
            dist, ind = knn.kneighbors(
                query.reshape(1, dim_nums, ts_length), 
                return_distance=True
            )
            
            # Map back to original indices
            original_indices = target_indices[ind[0][:]]
            
            logger.debug(
                f"Retrieved {len(original_indices)} guides for target {target_label}, "
                f"min distance: {dist[0][0]:.4f}"
            )
            
            return dist, original_indices
            
        except Exception as e:
            logger.error(f"Guide retrieval failed: {e}")
            raise
    
    def cf_label_fun(self, instance: torch.Tensor) -> int:
        """
        Determine counterfactual target label (second highest probability class).
        
        Args:
            instance: Input time series tensor of shape (C, T)
            
        Returns:
            Target class for counterfactual (second most likely class)
        """
        if instance.dim() != 2:
            raise ValueError(f"Expected 2D instance (C, T), got {instance.dim()}D")
        
        with torch.no_grad():
            output = self.softmax_fn(
                self.predict_fn(instance.reshape(1, instance.shape[0], instance.shape[1]).float())
            )
            target = torch.argsort(output, descending=True)[0, 1].item()
        
        logger.debug(f"Determined counterfactual target: {target}")
        return target
    
    def generate_saliency(
        self,
        data: np.ndarray,
        label: int,
        target_class: Optional[int] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, float, list, Dict[str, Any]]:
        """
        Generate counterfactual explanation for input data.
        
        Args:
            data: Input time series data of shape (C, T)
            label: Original class label
            target_class: Target counterfactual label (if None, auto-determined)
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (mask, counterfactual_input, target_probability, selected_channels, metrics)
            
        Raises:
            ValueError: If optimization fails or invalid inputs
        """
        logger.info(f"Starting MCELS explanation for label {label}")
        
        # Validate inputs
        if isinstance(data, np.ndarray) and data.ndim != 2:
            raise ValueError(f"Expected 2D data (C, T), got {data.ndim}D")
        
        self.mode = 'Explore'
        query = data.copy()
        
        # Convert to tensor
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)
        
        # Determine target label
        if target_class is None:
            # Try from kwargs first, then config, then auto-determine
            target_class = kwargs.get('target', SensorSpecifications.TARGET_CLASS)
            if target_class is None:
                target_class = self.cf_label_fun(data)
        
        logger.info(f"Target class for counterfactual: {target_class}")
        
        # Guide retrieval
        try:
            distances, indices = self.native_guide_retrieval(query, target_class, "euclidean", 1)
            guide_example = self.background_data[indices.item()]
            logger.debug(f"Selected guide with distance: {distances[0][0]:.4f}")
        except Exception as e:
            logger.error(f"Guide retrieval failed: {e}")
            return self._create_empty_result()
        
        # Initialize optimization
        self.eps = AlgorithmConstants.INITIAL_EPS
        
        # Initialize mask with reproducible random values
        mask_init = np.random.uniform(size=data.shape, low=0, high=1)
        mask = Variable(torch.from_numpy(mask_init), requires_grad=True)
        
        # Setup optimizer
        optimizer = torch.optim.Adam([mask], lr=self.args.lr)
        scheduler = None
        
        if self.args.enable_lr_decay:
            scheduler = ExponentialLR(optimizer, gamma=self.args.lr_decay)
        
        logger.info(f"Starting optimization with max_iterations={self.args.max_itr}")
        
        # Initialize metrics tracking
        metrics = defaultdict(list)
        
        # Early stopping parameters
        max_iterations_without_improvement = AlgorithmConstants.MAX_ITERATIONS_WITHOUT_IMPROVEMENT
        best_loss = float('inf')
        counter = 0
        
        # Optimization loop
        for i in range(self.args.max_itr + 1):
            guide_tensor = torch.tensor(guide_example, dtype=torch.float32)
            
            # Apply mask to create perturbation
            perturbated_input = data.mul(1 - mask) + guide_tensor.mul(mask)
            
            # Get prediction
            pred_outputs = self.softmax_fn(
                self.predict_fn(
                    perturbated_input.reshape(
                        1, perturbated_input.shape[0], perturbated_input.shape[1]
                    ).float()
                )
            )
            target_prob = float(pred_outputs[0][target_class].item())
            
            # Compute loss components
            l_maximize = 1 - pred_outputs[0][target_class]
            l_budget_loss = torch.mean(torch.abs(mask)) * float(self.args.enable_budget)
            l_tv_norm_loss = tv_norm(mask, self.args.tv_beta) * float(self.args.enable_tvnorm)
            
            total_loss = (
                self.args.l_budget_coeff * l_budget_loss +
                self.args.l_tv_norm_coeff * l_tv_norm_loss +
                self.args.l_max_coeff * l_maximize
            )
            
            # Early stopping logic
            current_loss = total_loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                counter = 0
            else:
                counter += 1
            
            # Store metrics (reduced frequency for performance)
            if i % 10 == 0:
                metrics['L_Maximize'].append(float(l_maximize.item()))
                metrics['L_Budget'].append(float(l_budget_loss.item()))
                metrics['L_TV_Norm'].append(float(l_tv_norm_loss.item()))
                metrics['L_Total'].append(float(total_loss.item()))
                metrics['CF_Prob'].append(float(target_prob))
                metrics['iteration'].append(i)
                
                logger.debug(
                    f"Iter {i}: Loss={current_loss:.4f}, "
                    f"TargetProb={target_prob:.3f}, Counter={counter}"
                )
            
            # Optimization step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            # Clamp mask values to [0, 1]
            mask.data.clamp_(0, 1)
            
            # Log to wandb if enabled
            if self.enable_wandb and i % 10 == 0:
                try:
                    import wandb
                    wandb_metrics = {k: v[-1] for k, v in metrics.items() if k != "epoch"}
                    wandb.log(wandb_metrics)
                except ImportError:
                    logger.warning("wandb not available for logging")
            
            # Early stopping checks
            if counter >= max_iterations_without_improvement:
                logger.info(
                    f"Early stopping at iteration {i}: "
                    f"No improvement for {max_iterations_without_improvement} iterations"
                )
                break
            
            # Success check
            if target_prob >= self.min_target_prob:
                logger.info(
                    f"Target probability threshold reached at iteration {i}: {target_prob:.3f}"
                )
                break
        
        # Post-processing
        final_mask, final_input, final_prob, success = self._process_final_result(
            data, mask, guide_example, target_class
        )
        
        # MCELS uses all channels (no channel selection or grouping)
        all_channels = list(range(SensorSpecifications.TOTAL_CHANNELS))
        
        # Add final metrics
        metrics['final_target_prob'] = final_prob
        metrics['predicted_class'] = int(torch.argmax(
            self.softmax_fn(
                self.predict_fn(
                    torch.tensor(final_input).reshape(1, final_input.shape[0], final_input.shape[1]).float()
                )
            )[0]
        ).item())
        metrics['success'] = success
        metrics['total_iterations'] = i + 1
        metrics['best_loss'] = best_loss
        metrics['num_channels_used'] = SensorSpecifications.TOTAL_CHANNELS
        metrics['algorithm'] = 'MCELS'
        metrics['uses_grouping'] = False
        metrics['channel_selection'] = False
        
        logger.info(
            f"Optimization completed. Final target probability: {final_prob:.3f}, "
            f"Success: {success}, Iterations: {i + 1}, All {SensorSpecifications.TOTAL_CHANNELS} channels used"
        )
        
        return final_mask, final_input, final_prob, all_channels, dict(metrics)
    
    def _process_final_result(
        self,
        original_data: torch.Tensor,
        mask: Variable,
        guide_example: np.ndarray,
        target_class: int
    ) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """
        Process final optimization result and apply thresholding.
        
        Args:
            original_data: Original input tensor
            mask: Optimized mask variable
            guide_example: Guide example used
            target_class: Target class
            
        Returns:
            Tuple of (final_mask, final_input, final_probability, success)
        """
        # Convert mask to numpy and apply threshold
        mask_numpy = mask.cpu().detach().numpy()
        converted_mask = np.where(mask_numpy >= self.mask_threshold, mask_numpy, 0)
        
        # Apply final mask
        guide_tensor = torch.tensor(guide_example, dtype=torch.float32)
        converted_mask_tensor = torch.tensor(converted_mask, dtype=torch.float32)
        
        perturbated_input = (
            original_data.mul(1 - converted_mask_tensor) +
            guide_tensor.mul(converted_mask_tensor)
        )
        
        # Final prediction
        with torch.no_grad():
            pred_outputs = self.softmax_fn(
                self.predict_fn(
                    perturbated_input.reshape(
                        1, perturbated_input.shape[0], perturbated_input.shape[1]
                    ).float()
                )
            )
            final_target_prob = float(pred_outputs[0][target_class].item())
            predicted_class = torch.argmax(pred_outputs[0]).item()
        
        # Determine success
        success = predicted_class == target_class and final_target_prob >= self.min_target_prob
        
        # Convert to numpy arrays
        final_mask = converted_mask.flatten()
        final_input = perturbated_input.cpu().detach().numpy()
        
        return final_mask, final_input, final_target_prob, success
    
    def _create_empty_result(self) -> Tuple[np.ndarray, np.ndarray, float, list, Dict[str, Any]]:
        """Create empty result for failed explanations."""
        logger.warning("Creating empty result due to failure")
        return (
            np.array([]),
            np.array([]),
            0.0,
            [],  # Empty channel list for failed cases
            {'success': False, 'error': 'Guide retrieval failed', 'algorithm': 'MCELS'}
        )
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Get model predictions for input data.
        
        Args:
            x: Input data of shape (N, C, T) or (C, T)
            
        Returns:
            Predicted class labels of shape (N,)
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            logits = self.predict_fn(x)
            predictions = torch.argmax(logits, dim=1)
        
        return predictions.cpu().numpy()
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """
        Get summary of current configuration.
        
        Returns:
            Dictionary with configuration parameters
        """
        return {
            'algorithm': 'MCELS',
            'confidence_threshold': self.conf_threshold,
            'mask_threshold': self.mask_threshold,
            'min_target_probability': self.min_target_prob,
            'initial_eps': self.eps,
            'eps_decay': self.eps_decay,
            'background_samples': len(self.background_data),
            'num_channels': SensorSpecifications.TOTAL_CHANNELS,
            'enable_wandb': self.enable_wandb,
            'use_cuda': self.use_cuda
        }
    
    def save_results(self, save_path: str) -> None:
        """Save MCELS results to file."""
        if not hasattr(self, 'results') or not self.results:
            logger.error("No results to save.")
            return
        
        # Prepare serializable results
        save_data = {}
        for key, value in self.results.items():
            if isinstance(value, np.ndarray):
                save_data[key] = value.tolist()
            elif isinstance(value, torch.Tensor):
                save_data[key] = value.cpu().numpy().tolist()
            elif isinstance(value, (tuple, list)):
                save_data[key] = list(value)
            elif hasattr(value, '__dict__'):
                # Convert objects with attributes to dictionary
                save_data[key] = vars(value)
            else:
                save_data[key] = value
        
        # Save to JSON
        import json
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)

        logger.info(f"MCELS results saved to {save_path}")