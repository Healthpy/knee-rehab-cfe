"""
M-CELS Explainer Implementation

Counterfactual Explanation for Multivariate Time Series Data
Guided by Learned Saliency Maps.

Reference:
    M-CELS: Counterfactual Explanation for Multivariate Time Series Data
    Guided by Learned Saliency Maps
    https://ieeexplore.ieee.org/document/10903326
    https://github.com/Luckilyeee/M-CELS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Tuple, Optional, Any
from torch.optim.lr_scheduler import ExponentialLR
from tslearn.neighbors import KNeighborsTimeSeries
from torch.autograd import Variable

from src.explainer.base import Saliency
from src.explainer.perturbation_manager import PerturbationManager
from src.explainer.utils import normalize, save_timeseries_mul


def tv_norm(signal, tv_beta):
    """Calculate TV norm with power function
    
    Args:
        signal: Input signal tensor
        tv_beta: Power parameter for TV norm
        
    Returns:
        TV norm value
    """
    signal = signal.flatten()
    signal_grad = torch.mean(torch.abs(signal[:-1] - signal[1:]).pow(tv_beta))
    return signal_grad


class MCELSExplainer(Saliency):
    """
    M-CELS: Counterfactual Explanation with Learned Saliency Maps
    
    Generates counterfactual explanations for multivariate time series
    by learning saliency masks that guide the perturbation of input
    towards a target class.
    
    Key Features:
    - Native guide retrieval from background data
    - Learned saliency masks for interpretability
    - TV norm and budget constraints for sparsity
    - Early stopping based on confidence threshold
    """
    
    def __init__(
        self,
        background_data: np.ndarray,
        background_label: np.ndarray,
        predict_fn: callable,
        enable_wandb: bool = False,
        args: Optional[Any] = None,
        use_cuda: bool = False
    ):
        """
        Initialize M-CELS explainer
        
        Args:
            background_data: Reference dataset [N, C, T]
            background_label: Labels for background data [N]
            predict_fn: Model prediction function
            enable_wandb: Enable Weights & Biases logging
            args: Additional arguments (hyperparameters)
            use_cuda: Use CUDA acceleration
        """
        super().__init__(background_data, background_label, predict_fn)
        
        self.enable_wandb = enable_wandb
        self.args = args
        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.enable_lr_decay = getattr(args, 'enable_lr_decay', False) if args else False
        self.lr_decay = getattr(args, 'lr_decay', 0.99) if args else 0.99
        # Default hyperparameters
        self.max_itr = getattr(args, 'max_itr', 5000) if args else 5000
        self.l_tv_norm_coeff = getattr(args, 'l_tv_norm_coeff', 0.6) if args else 0.6
        self.l_budget_coeff = getattr(args, 'l_budget_coeff', 0.5) if args else 0.5
        self.l_max_coeff = getattr(args, 'l_max_coeff', 0.7) if args else 0.7
        self.enable_tvnorm = getattr(args, 'enable_tvnorm', True) if args else True
        self.enable_budget = getattr(args, 'enable_budget', True) if args else True
        self.learning_rate = getattr(args, 'learning_rate', 0.1) if args else 0.1
        self.tv_beta = getattr(args, 'tv_beta', 3) if args else 3
        
        self.softmax_fn = nn.Softmax(dim=1)
    
    def native_guide_retrieval(
        self,
        query: torch.Tensor,
        target_label: int,
        distance: str = 'euclidean',
        n_neighbors: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve nearest neighbors from background data of target class
        
        Args:
            query: Query time series [C, T]
            target_label: Target class to retrieve from
            distance: Distance metric ('dtw' or 'euclidean')
            n_neighbors: Number of neighbors to retrieve
            
        Returns:
            dist: Distances to neighbors
            ind: Indices of neighbors in background data
        """
        dim_nums, ts_length = query.shape[0], query.shape[1]
        df = pd.DataFrame(self.background_label, columns=['label'])
        
        # Get samples from target class
        target_indices = list(df[df['label'] == target_label].index.values)
        target_data = self.background_data[target_indices]
        
        # k-NN search
        knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric=distance)
        knn.fit(target_data)
        
        dist, ind = knn.kneighbors(
            query.reshape(1, dim_nums, ts_length),
            return_distance=True
        )
        
        # Map back to original indices
        original_indices = df[df['label'] == target_label].index[ind[0][:]]
        
        return dist, original_indices
    
    def cf_label_fun(self, instance: torch.Tensor) -> int:
        """
        Get counterfactual target label (second most likely class)
        
        Args:
            instance: Time series [C, T]
            
        Returns:
            Target class label
        """
        output = self.softmax_fn(
            self.predict_fn(instance.reshape(1, instance.shape[0], instance.shape[1]).float())
        )
        target = torch.argsort(output, descending=True)[0, 1].item()
        return target
    
    def generate_saliency(
        self,
        data: torch.Tensor,
        label: int,
        target_class: Optional[int] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, float, int]:
        """
        Generate counterfactual explanation with learned saliency mask
        
        Args:
            data: Input time series [C, T]
            label: Original class label
            target_class: Target class for counterfactual (optional, auto-selected if None)
            **kwargs: Additional arguments (save_dir, dataset, etc.)
            
        Returns:
            mask: Learned saliency mask [C, T]
            ori_perturbated: Counterfactual time series [C, T]
            target_prob: Probability of target class
            actual_iterations: Number of iterations completed
        """
        # Convert to tensor and move to device
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32, device=self.device)
        else:
            data = data.to(self.device)
        dim_nums, ts_length = data.shape[0], data.shape[1]
        
        # Get target class (counterfactual class)
        if target_class is not None:
            cf_label = target_class
        else:
            cf_label = self.cf_label_fun(data)
        
        # Retrieve native guide from background data
        print(f"Retrieving native guide for target class {cf_label}...")
        dist, guide_indices = self.native_guide_retrieval(
            data.cpu().numpy(),
            cf_label,
            distance='euclidean',
            n_neighbors=1
        )
        
        # Get the native guide (nearest neighbor from target class)
        NUN = torch.tensor(
            self.background_data[guide_indices[0]],
            dtype=torch.float32,
            device=self.device
        )
        
        # Initialize learnable mask
        mask_init = np.random.uniform(size=(dim_nums, ts_length), low=0, high=1)
        mask = Variable(torch.from_numpy(mask_init).to(self.device), requires_grad=True)
        
        # Optimizer
        optimizer = torch.optim.Adam([mask], lr=self.learning_rate)

        if self.enable_lr_decay:
            scheduler = ExponentialLR(optimizer, gamma=self.lr_decay)

        
        # Get original prediction
        original_output = self.softmax_fn(
            self.predict_fn(data.reshape(1, dim_nums, ts_length).float())
        )
        top_prediction_class = torch.argmax(original_output).item()
        
        print(f"{self.args.algo if self.args else 'MCELS'}: Optimizing... ")
        metrics = defaultdict(lambda: [])
        
        # Early stopping parameters
        max_iterations_without_improvement = 100
        imp_threshold = 0.001
        best_loss = float('inf')
        counter = 0
        
        # Training loop
        i = 0
        
        while i <= self.max_itr:
            # Generate perturbated input
            Rt = NUN  # NUN doesn't have gradients, no need to clone/detach
            perturbated_input = data.mul(1 - mask) + Rt.mul(mask)
            
            # Get prediction
            pred_outputs = self.softmax_fn(
                self.predict_fn(perturbated_input.reshape(1, dim_nums, ts_length).float())
            )
            
            # Loss components
            # 1. Classification loss: maximize target class probability
            loss_max = 1 - pred_outputs[0][cf_label]
            
            # 2. Budget: encourage sparsity
            loss_budget = torch.mean(torch.abs(mask)) * float(self.enable_budget)
            
            # 3. TV norm: encourage temporal smoothness
            loss_tv = tv_norm(mask, self.tv_beta) * float(self.enable_tvnorm)
            
            # Combined loss
            loss = (
                self.l_budget_coeff * loss_budget +
                self.l_tv_norm_coeff * loss_tv +
                self.l_max_coeff * loss_max
            )
            
            # Early stopping check (before backward pass)
            if best_loss - loss.item() < imp_threshold:
                counter += 1
            else:
                counter = 0
                best_loss = loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Track metrics
            metrics['L_Maximize'].append(float(loss_max.item()))
            metrics['L_Budget'].append(float(loss_budget.item()))
            metrics['L_TV_Norm'].append(float(loss_tv.item()))
            metrics['L_Total'].append(float(loss.item()))
            metrics['CF_Prob'].append(float(pred_outputs[0][cf_label].item()))
            
            optimizer.step()
            if self.enable_lr_decay:
                scheduler.step()
            
            # Clamp mask AFTER optimizer step
            mask.data.clamp_(0, 1)
            
            # Log progress
            if i % 100 == 0:
                mask_mean = mask.data.mean().item()
                grad_norm = mask.grad.norm().item() if mask.grad is not None else 0.0
                print(f"Iteration {i}/{self.max_itr}: "
                      f"Total Loss={loss.item():.4f}, "
                      f"Loss_max={loss_max.item():.4f}, "
                      f"Loss_TV={loss_tv.item():.4f}, "
                      f"Loss_budget={loss_budget.item():.4f}, "
                      f"Target Prob={pred_outputs[0][cf_label].item():.4f}, "
                      f"Mask_mean={mask_mean:.4f}, "
                      f"Grad_norm={grad_norm:.6f}")
            
            # Check early stopping
            if counter >= max_iterations_without_improvement:
                print(f"Early stopping at iteration {i}")
                break
            
            i += 1
        
        # Store actual iterations reached
        actual_iterations = i
        
        # Convert mask to numpy
        mask = mask.cpu().detach().numpy()
        
        # Apply threshold to mask
        threshold = 0.5
        converted_mask = np.where(mask >= threshold, mask, 0)
        
        # Generate final perturbated input with converted mask
        Rt = NUN  # NUN doesn't have gradients, no need to clone/detach
        converted_mask_tensor = torch.from_numpy(converted_mask).float().to(self.device)
        ori_perturbated = data.mul(1 - converted_mask_tensor) + Rt.mul(converted_mask_tensor)
        
        # Get final target probability
        pred_outputs = self.softmax_fn(
            self.predict_fn(ori_perturbated.reshape(1, dim_nums, ts_length).float())
        )
        target_prob = float(pred_outputs[0][cf_label].item())
        
        # Flatten converted mask for visualization
        flatten_mask = converted_mask.flatten()
        
        # Convert perturbated input to numpy (it's already a torch tensor here)
        perturbated_numpy = ori_perturbated.cpu().detach().numpy()
        
        # Save visualization if requested
        if kwargs.get('save_dir'):
            save_timeseries_mul(
                mask=flatten_mask,
                raw_mask=None,
                time_series=data.cpu().detach().numpy(),
                perturbated_output=perturbated_numpy,
                save_dir=kwargs['save_dir'],
                enable_wandb=self.enable_wandb,
                algo=self.args.algo if self.args else 'mcels',
                dataset=kwargs.get('dataset', ''),
                category=top_prediction_class
            )
        
        return converted_mask, perturbated_numpy, target_prob, actual_iterations
    
    def explain(
        self,
        X: np.ndarray,
        y: int,
        save_dir: Optional[str] = None,
        dataset: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        High-level API for generating explanations
        
        Args:
            X: Input time series [C, T] or [T, C]
            y: True label
            save_dir: Directory to save results
            dataset: Dataset name
            
        Returns:
            Dictionary containing:
                - mask: Saliency mask
                - counterfactual: Counterfactual time series
                - confidence: Target class probability
                - original_pred: Original prediction
                - cf_pred: Counterfactual prediction
        """
        # Ensure correct shape [C, T]
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        
        # Get original prediction
        with torch.no_grad():
            orig_output = self.softmax_fn(
                self.predict_fn(X.reshape(1, *X.shape).float())
            )
            orig_pred = torch.argmax(orig_output).item()
        
        # Generate explanation
        mask, counterfactual, cf_confidence, actual_iterations = self.generate_saliency(
            X,
            y,
            save_dir=save_dir,
            dataset=dataset
        )
        
        # Get counterfactual prediction
        with torch.no_grad():
            cf_tensor = torch.from_numpy(counterfactual).float().to(self.device)
            cf_output = self.softmax_fn(
                self.predict_fn(cf_tensor.reshape(1, *cf_tensor.shape))
            )
            cf_pred = torch.argmax(cf_output).item()
        
        return {
            'mask': mask,  # Already numpy from generate_saliency
            'counterfactual': counterfactual,  # Already numpy from generate_saliency
            'confidence': cf_confidence,
            'original_pred': orig_pred,
            'cf_pred': cf_pred,
            'original_label': y
        }
    
    def generate_counterfactual(
        self,
        data: np.ndarray,
        label: int,
        target_class: int
    ) -> Dict[str, Any]:
        """
        Generate counterfactual with target class specification
        
        Args:
            data: Input time series [C, T]
            label: Original class label
            target_class: Target class for counterfactual
            
        Returns:
            Dictionary containing:
                - saliency_mask: Learned saliency mask [C, T]
                - counterfactual: Counterfactual time series [C, T]
                - confidence: Target class probability
                - info: Additional information (iterations, etc.)
        """
        # Generate saliency explanation
        mask, counterfactual, target_prob, actual_iterations = self.generate_saliency(
            data,
            label,
            target_class=target_class
        )
        
        return {
            'saliency_mask': mask,  # Already numpy from generate_saliency
            'counterfactual': counterfactual,  # Already numpy from generate_saliency
            'confidence': target_prob,
            'info': {
                'iterations': actual_iterations,  # Actual iterations completed
                'target_class': target_class,
                'original_label': label
            }
        }
