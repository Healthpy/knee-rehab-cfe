"""
Shapley-Guided Adaptive Multi-Objective Counterfactual Explainer

This module combines:
1. DeepSHAP-based channel importance ranking
2. Adaptive multi-objective optimization
3. Group-wise counterfactual generation

Compatible with the M-CELS evaluation pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import shap
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict
from pathlib import Path
import json
from tslearn.neighbors import KNeighborsTimeSeries

logger = logging.getLogger(__name__)


def tv_norm(signal, tv_beta=3):
    """Calculate TV norm with power function for temporal smoothness"""
    signal = signal.flatten()
    signal_grad = torch.mean(torch.abs(signal[:-1] - signal[1:]).pow(tv_beta))
    return signal_grad


class ShapleyAdaptiveExplainer:
    """
    Shapley-Guided Adaptive Multi-Objective Counterfactual Explainer.
    
    Generates counterfactual explanations by:
    1. Ranking channel groups using DeepSHAP
    2. Selecting top influential groups
    3. Optimizing with adaptive multi-objective loss
    4. Balancing target confidence, sparsity, and smoothness
    """
    
    def __init__(
        self,
        background_data: np.ndarray,
        background_label: np.ndarray,
        predict_fn,
        enable_wandb: bool = False,
        args=None,
        use_cuda: bool = True
    ):
        """
        Initialize Shapley-Guided Adaptive Explainer.
        
        Args:
            background_data: Training data [N, C, T]
            background_label: Training labels [N]
            predict_fn: Model prediction function
            enable_wandb: Whether to use wandb logging
            args: Configuration arguments
            use_cuda: Whether to use CUDA
        """
        self.background_data = torch.FloatTensor(background_data) if isinstance(background_data, np.ndarray) else background_data
        self.background_label = background_label
        self.predict_fn = predict_fn
        self.enable_wandb = enable_wandb
        self.args = args
        
        # Device setup
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.background_data = self.background_data.to(self.device)
        
        # Hyperparameters
        self.max_itr = getattr(args, 'max_itr', 5000)
        self.learning_rate = getattr(args, 'learning_rate', 0.01)
        self.enable_lr_decay = getattr(args, 'enable_lr_decay', True)
        self.lr_decay = getattr(args, 'lr_decay', 0.9991)
        
        # Loss coefficients (adaptive)
        self.l_target_coeff = getattr(args, 'l_max_coeff', 1.0)
        self.l_sparsity_coeff = getattr(args, 'l_budget_coeff', 0.5)
        self.l_smoothness_coeff = getattr(args, 'l_tv_norm_coeff', 0.3)
        self.l_group_sparse_coeff = getattr(args, 'l_group_sparse_coeff', 0.4)
        
        # Ablation toggles
        self.enable_adaptive_weights = getattr(args, 'enable_adaptive_weights', True)
        self.enable_group_sparsity_loss = getattr(args, 'enable_group_sparsity_loss', True)
        
        # Quality thresholds
        self.min_confidence = getattr(args, 'min_target_probability', 0.7)
        self.target_threshold = getattr(args, 'target_threshold', 0.8)

        # Post-optimization refinement (improves sparsity/smoothness)
        self.refine_thresholds = getattr(
            args, 'refine_thresholds', [0.85, 0.75, 0.65, 0.55, 0.5]
        )
        self.refine_temporal_kernel = getattr(args, 'refine_temporal_kernel', 9)
        self.refine_require_valid = getattr(args, 'refine_require_valid', True)
        self.refine_cf_blends = getattr(args, 'refine_cf_blends', [0.0, 0.3, 0.6])
        
        # Shapley configuration
        self.use_shapley = getattr(args, 'use_shapley_ranking', True)
        self.max_groups_ratio = getattr(args, 'max_groups_ratio', 0.5)
        self.group_level = getattr(args, 'group_level', 'modality')
        
        # Define sensor groups
        self.sensor_groups = self._create_sensor_groups(self.group_level)
        
        # SHAP cache: keyed by original_class → channel_importance array
        self._shap_cache: Dict[int, np.ndarray] = {}
        self._shap_explainer = None  # lazily initialised
        
        # Prepare class-stratified background data (up to 10 samples per class)
        self._stratified_bg = self._prepare_stratified_background(max_per_class=10)
        
        # Adaptive weight manager
        self.adaptive_weights = {
            'target': self.l_target_coeff,
            'sparsity': self.l_sparsity_coeff,
            'smoothness': self.l_smoothness_coeff,
            'group_sparsity': self.l_group_sparse_coeff
        }
        self.weight_history = []
        
        logger.info(f"ShapleyAdaptiveExplainer initialized on {self.device}")
        logger.info(f"  - Use SHAP ranking: {self.use_shapley}")
        logger.info(f"  - Group level: {self.group_level} ({len(self.sensor_groups)} groups)")
        logger.info(f"  - Max groups ratio: {self.max_groups_ratio}")
        logger.info(f"  - Stratified background: {self._stratified_bg.shape[0]} samples")
    
    def _prepare_stratified_background(
        self, max_per_class: int = 10
    ) -> torch.Tensor:
        """Build a class-stratified background set for SHAP.
        
        Ensures every class is represented, giving SHAP a balanced
        reference distribution rather than a random subset.
        """
        bg_np = self.background_data.cpu().numpy() if isinstance(
            self.background_data, torch.Tensor
        ) else self.background_data
        labels = np.array(self.background_label)
        
        selected_indices = []
        for cls in np.unique(labels):
            cls_idx = np.where(labels == cls)[0]
            n_pick = min(len(cls_idx), max_per_class)
            chosen = np.random.choice(cls_idx, size=n_pick, replace=False)
            selected_indices.extend(chosen.tolist())
        
        np.random.shuffle(selected_indices)
        return torch.tensor(
            bg_np[selected_indices], dtype=torch.float32, device=self.device
        )
    
    def _create_sensor_groups(self, group_level: str) -> Dict[str, List[int]]:
        """Create sensor or modality groups for 48-channel IMU data"""
        if group_level == 'sensor':
            # 8 sensor-level groups (6 channels each)
            return {
                'R_RF': list(range(0, 6)),
                'R_HAM': list(range(6, 12)),
                'R_TA': list(range(12, 18)),
                'R_GAS': list(range(18, 24)),
                'L_RF': list(range(24, 30)),
                'L_HAM': list(range(30, 36)),
                'L_TA': list(range(36, 42)),
                'L_GAS': list(range(42, 48)),
            }
        elif group_level == 'modality':
            # 16 modality-level groups (3 channels each)
            groups = {}
            sensor_names = ['R_RF', 'R_HAM', 'R_TA', 'R_GAS', 'L_RF', 'L_HAM', 'L_TA', 'L_GAS']
            
            for i, sensor in enumerate(sensor_names):
                base_idx = i * 6
                groups[f'{sensor}_acc'] = list(range(base_idx, base_idx + 3))
                groups[f'{sensor}_gyr'] = list(range(base_idx + 3, base_idx + 6))
            
            return groups
        else:
            raise ValueError(f"Invalid group_level: {group_level}")
    
    def _compute_shapley_importance(
        self,
        x: torch.Tensor,
        original_class: int
    ) -> Tuple[List[int], np.ndarray]:
        """
        Compute channel importance for the *original* predicted class.
        
        We rank by the original class because we want to know which
        channels drive the current (incorrect) prediction — those are
        the channels that should be modified to flip the prediction.
        
        Results are cached per original_class so repeated calls with
        different samples of the same class reuse the SHAP explainer.
        
        Args:
            x: Input sample [C, T]
            original_class: The model's current predicted class
            
        Returns:
            Tuple of (ranked_channel_indices, importance_scores)
        """
        # ── Return cached result if available ──────────────────────────
        if original_class in self._shap_cache:
            channel_importance = self._shap_cache[original_class]
            ranked_indices = np.argsort(channel_importance)[::-1].tolist()
            logger.info(f"SHAP cache hit for class {original_class}, "
                       f"top 5 channels: {ranked_indices[:5]}")
            return ranked_indices, channel_importance
        
        try:
            x_np = x.cpu().numpy()
            x_tensor = torch.tensor(
                x_np, dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            x_tensor.requires_grad = True
            
            # ── Lazy-init the GradientExplainer (reused across calls) ──
            if self._shap_explainer is None:
                class _ModelWrapper(nn.Module):
                    def __init__(self, predict_fn):
                        super().__init__()
                        self.predict_fn = predict_fn
                    def forward(self, x):
                        return self.predict_fn(x)
                
                wrapper = _ModelWrapper(self.predict_fn)
                wrapper.eval()
                self._shap_explainer = shap.GradientExplainer(
                    wrapper, self._stratified_bg
                )
                logger.info("Initialized GradientExplainer with "
                           f"{self._stratified_bg.shape[0]} stratified bg samples")
            
            # ── Compute SHAP values ────────────────────────────────────
            shap_values = self._shap_explainer.shap_values(x_tensor)
            
            # Extract values for the *original* class
            if isinstance(shap_values, list):
                shap_values = shap_values[original_class]
            
            if isinstance(shap_values, torch.Tensor):
                shap_values = shap_values.cpu().numpy()
            
            shap_values = shap_values.squeeze(0)  # [C, T]
            
            # Aggregate temporal SHAP values per channel
            channel_importance = np.mean(np.abs(shap_values), axis=-1)
            
            # Cache for reuse
            self._shap_cache[original_class] = channel_importance
            
            ranked_indices = np.argsort(channel_importance)[::-1].tolist()
            logger.info(f"SHAP computed for original class {original_class}, "
                       f"top 5 channels: {ranked_indices[:5]}")
            
            return ranked_indices, channel_importance
            
        except Exception as e:
            logger.warning(f"SHAP failed: {e}. Falling back to gradient importance.")
            
            # Fallback: gradient w.r.t. original class logit
            x_tensor = x.unsqueeze(0).to(self.device).requires_grad_(True)
            output = self.predict_fn(x_tensor)
            output[0, original_class].backward()
            
            gradients = x_tensor.grad.squeeze(0).cpu().numpy()  # [C, T]
            channel_importance = np.mean(np.abs(gradients), axis=-1)
            
            self._shap_cache[original_class] = channel_importance
            
            ranked_indices = np.argsort(channel_importance)[::-1].tolist()
            logger.info(f"Gradient importance for class {original_class}, "
                       f"top 5 channels: {ranked_indices[:5]}")
            
            return ranked_indices, channel_importance
    
    def _aggregate_group_importance(
        self,
        channel_importance: np.ndarray
    ) -> Dict[str, float]:
        """Aggregate channel importance to group level using max.
        
        We use max rather than mean so that a single highly-important
        channel is enough to elevate the entire group.  This avoids
        dilution when other channels in the group are near-zero.
        """
        group_importance = {}
        
        for group_name, channel_indices in self.sensor_groups.items():
            group_scores = [channel_importance[ch] for ch in channel_indices]
            group_importance[group_name] = float(np.max(group_scores))
        
        return group_importance
    
    def _select_influential_groups(
        self,
        group_importance: Dict[str, float]
    ) -> List[str]:
        """Select top influential groups based on importance scores"""
        # Sort groups by importance
        sorted_groups = sorted(group_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Select top ~50% of groups
        total_groups = len(sorted_groups)
        num_select = max(1, int(total_groups * self.max_groups_ratio))
        
        selected_groups = [name for name, _ in sorted_groups[:num_select]]
        
        logger.info(f"Selected {len(selected_groups)}/{total_groups} groups: {selected_groups}")
        
        return selected_groups
    
    def _find_nearest_neighbor(
        self,
        x: torch.Tensor,
        target_class: int
    ) -> torch.Tensor:
        """Find nearest neighbor from target class using tslearn KNN (same as M-CELS)."""
        x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
        dim_nums, ts_length = x_np.shape

        # Get indices of target class samples
        df = pd.DataFrame(self.background_label, columns=['label'])
        target_indices = list(df[df['label'] == target_class].index.values)

        if len(target_indices) == 0:
            logger.warning(f"No samples found for target class {target_class}, using random sample")
            return self.background_data[0]

        # Get background data as numpy for tslearn
        bg_np = self.background_data.cpu().numpy() if isinstance(self.background_data, torch.Tensor) else self.background_data
        target_data = bg_np[target_indices]

        # k-NN search with tslearn (matching M-CELS)
        knn = KNeighborsTimeSeries(n_neighbors=1, metric='euclidean')
        knn.fit(target_data)
        dist, ind = knn.kneighbors(
            x_np.reshape(1, dim_nums, ts_length),
            return_distance=True
        )

        # Map back to original index and return as tensor
        original_idx = df[df['label'] == target_class].index[ind[0][0]]
        nn_np = bg_np[original_idx]
        return torch.tensor(nn_np, dtype=torch.float32, device=self.device)
    
    def _create_group_masks(
        self,
        selected_groups: List[str],
        channels: int,
        timesteps: int
    ) -> Dict[str, torch.Tensor]:
        """Create binary masks for selected groups"""
        group_masks = {}
        
        for group_name in selected_groups:
            mask = torch.zeros(1, channels, timesteps, device=self.device)
            channel_indices = self.sensor_groups[group_name]
            mask[0, channel_indices, :] = 1.0
            group_masks[group_name] = mask
        
        return group_masks
    
    def _adaptive_optimization(
        self,
        x: torch.Tensor,
        target_class: int,
        selected_groups: List[str],
        group_masks: Dict[str, torch.Tensor],
        nearest_neighbor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Perform adaptive multi-objective optimization with learnable group gates.
        
        Returns:
            Tuple of (best_mask, best_counterfactual, actual_iterations)
        """
        channels, timesteps = x.shape
        
        # Initialize mask for selected channels only
        selected_channels = []
        for group_name in selected_groups:
            selected_channels.extend(self.sensor_groups[group_name])
        selected_channels = sorted(list(set(selected_channels)))
        
        # Create fixed binary gate to enforce group selection
        group_gate = torch.zeros(channels, timesteps, device=self.device)
        group_gate[selected_channels, :] = 1.0
        
        logger.info(f"Group gate enabled: {len(selected_channels)}/{channels} channels from {len(selected_groups)} groups")
        # Initialize learnable mask
        mask_init = np.random.uniform(size=(channels, timesteps), low=0, high=1)
        mask = torch.tensor(mask_init, dtype=torch.float32, device=self.device, requires_grad=True)
        
        # Optimizer
        optimizer = optim.Adam([mask], lr=self.learning_rate)
        
        if self.enable_lr_decay:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay)
        
        # Tracking
        best_mask = mask.clone().detach()
        best_counterfactual = x.clone()
        best_confidence = 0.0
        
        # Early stopping
        max_iterations_without_improvement = 100
        imp_threshold = 0.001
        best_loss = float('inf')
        counter = 0
        
        i = 0
        while i <= self.max_itr:
            # Apply group gate to restrict modifications to selected groups only
            # Mask is optimized directly in [0, 1] space (same as M-CELS)
            gated_mask = mask * group_gate
            
            cf = x * (1 - gated_mask) + nearest_neighbor * gated_mask
            
            # Get prediction
            cf_expanded = cf.unsqueeze(0)
            logits = self.predict_fn(cf_expanded)
            probs = torch.softmax(logits, dim=1)
            
            # Compute loss components
            # 1. Target prediction loss
            target_prob = probs[0, target_class]
            loss_target = 1.0 - target_prob
            
            # 2. Sparsity loss (L1 on gated mask)
            loss_sparsity = torch.mean(torch.abs(gated_mask))
            
            # 3. Smoothness loss (TV norm on gated mask)
            loss_smoothness = tv_norm(gated_mask, tv_beta=3)
            
            # 4. Group sparsity loss
            if self.enable_group_sparsity_loss:
                loss_group_sparsity = 0.0
                for group_name, group_mask in group_masks.items():
                    group_activation = torch.sum(gated_mask * group_mask[0]) / torch.sum(group_mask[0])
                    loss_group_sparsity += group_activation
                loss_group_sparsity /= len(group_masks)
            else:
                loss_group_sparsity = torch.tensor(0.0, device=self.device)
            
            # Combined loss with adaptive weights
            loss = (
                self.adaptive_weights['target'] * loss_target +
                self.adaptive_weights['sparsity'] * loss_sparsity +
                self.adaptive_weights['smoothness'] * loss_smoothness +
                self.adaptive_weights['group_sparsity'] * loss_group_sparsity
            )
            
            # Update adaptive weights (only if enabled)
            if self.enable_adaptive_weights:
                if target_prob.item() >= self.target_threshold:
                    self.adaptive_weights['target'] *= 0.95
                    self.adaptive_weights['sparsity'] *= 1.05
                    self.adaptive_weights['smoothness'] *= 1.05
                else:
                    self.adaptive_weights['target'] *= 1.05
                    self.adaptive_weights['sparsity'] *= 0.95
                
                # Clamp weights
                for key in self.adaptive_weights:
                    self.adaptive_weights[key] = np.clip(self.adaptive_weights[key], 0.1, 2.0)
            
            # Early stopping check
            if best_loss - loss.item() < imp_threshold:
                counter += 1
            else:
                counter = 0
                best_loss = loss.item()
            
            # Update best
            if target_prob.item() > best_confidence:
                best_confidence = target_prob.item()
                best_mask = gated_mask.clone().detach()
                best_counterfactual = cf.clone().detach()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if self.enable_lr_decay:
                scheduler.step()
            
            # Clamp mask to [0, 1] (same as M-CELS)
            mask.data.clamp_(0, 1)
            
            # Logging
            if i % 100 == 0:
                logger.debug(f"Iter {i}/{self.max_itr}: "
                           f"Loss={loss.item():.4f}, "
                           f"Target_prob={target_prob.item():.4f}, "
                           f"Best_conf={best_confidence:.4f}")
            
            # Check early stopping
            if counter >= max_iterations_without_improvement:
                logger.info(f"Early stopping at iteration {i}")
                break
            
            if best_confidence >= self.min_confidence and i >= 200:
                logger.info(f"Target confidence {best_confidence:.2%} reached at iteration {i}")
                break
            
            i += 1
        
        actual_iterations = i
        
        # Log which groups/channels were actually modified
        modified_channels = (best_mask.mean(dim=1) > 0.01).nonzero(as_tuple=True)[0].cpu().numpy()
        modified_groups = []
        for group_name, channel_indices in self.sensor_groups.items():
            if any(ch in modified_channels for ch in channel_indices):
                modified_groups.append(group_name)
        
        logger.info(f"Modified {len(modified_channels)}/{channels} channels from {len(modified_groups)} groups: {modified_groups}")
        logger.info(f"Selected groups were: {selected_groups}")
        
        return best_mask, best_counterfactual, actual_iterations

    def _refine_counterfactual(
        self,
        x: torch.Tensor,
        nearest_neighbor: torch.Tensor,
        best_mask: torch.Tensor,
        best_counterfactual: torch.Tensor,
        target_class: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Refine CF by smoothing+thresholding mask; pick sparsest valid candidate."""
        with torch.no_grad():
            logits = self.predict_fn(best_counterfactual.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)
            base_prob = probs[0, target_class].item()
            base_pred = torch.argmax(probs, dim=1).item()

        chosen_mask = best_mask.clone().detach()
        chosen_cf = best_counterfactual.clone().detach()
        chosen_prob = base_prob
        chosen_density = float(chosen_mask.mean().item())
        chosen_tg = float(
            torch.mean(torch.abs(chosen_cf[:, 1:] - chosen_cf[:, :-1])).item()
        )

        kernel = int(self.refine_temporal_kernel)
        if kernel % 2 == 0:
            kernel += 1
        kernel = max(1, kernel)

        smoothed = chosen_mask
        if kernel > 1:
            smoothed = F.avg_pool1d(
                best_mask.unsqueeze(0), kernel_size=kernel,
                stride=1, padding=kernel // 2
            ).squeeze(0)

        for thr in self.refine_thresholds:
            candidate_mask = (smoothed >= float(thr)).float()
            base_candidate_cf = x * (1.0 - candidate_mask) + nearest_neighbor * candidate_mask

            for blend in self.refine_cf_blends:
                blend = float(blend)
                candidate_cf = base_candidate_cf
                if kernel > 1 and blend > 0.0:
                    smooth_cf = F.avg_pool1d(
                        base_candidate_cf.unsqueeze(0), kernel_size=kernel,
                        stride=1, padding=kernel // 2
                    ).squeeze(0)
                    candidate_cf = (1.0 - blend) * base_candidate_cf + blend * smooth_cf
                    candidate_cf = x * (1.0 - candidate_mask) + candidate_cf * candidate_mask

                with torch.no_grad():
                    logits = self.predict_fn(candidate_cf.unsqueeze(0))
                    probs = torch.softmax(logits, dim=1)
                    prob = probs[0, target_class].item()
                    pred = torch.argmax(probs, dim=1).item()

                valid = (pred == target_class)
                if self.refine_require_valid and not valid:
                    continue
                if prob < self.min_confidence:
                    continue

                density = float(candidate_mask.mean().item())
                temporal_grad = float(
                    torch.mean(torch.abs(candidate_cf[:, 1:] - candidate_cf[:, :-1])).item()
                )
                if (
                    density < chosen_density - 1e-8
                    or (abs(density - chosen_density) <= 1e-8 and temporal_grad < chosen_tg)
                ):
                    chosen_mask = candidate_mask
                    chosen_cf = candidate_cf
                    chosen_prob = prob
                    chosen_density = density
                    chosen_tg = temporal_grad

        return chosen_mask, chosen_cf, chosen_prob
    
    def generate_saliency(
        self,
        data: Union[torch.Tensor, np.ndarray],
        label: int,
        target_class: Optional[int] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """
        Generate counterfactual explanation using Shapley-guided adaptive optimization.
        
        Args:
            data: Input time series [C, T]
            label: Original class label
            target_class: Target class for counterfactual
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (mask, counterfactual, target_prob, actual_iterations)
        """
        # Convert to tensor
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32, device=self.device)
        else:
            data = data.to(self.device)
        
        channels, timesteps = data.shape
        
        # Get target class if not provided
        if target_class is None:
            with torch.no_grad():
                logits = self.predict_fn(data.unsqueeze(0))
                probs = torch.softmax(logits, dim=1)
                target_class = torch.argsort(probs, descending=True)[0, 1].item()
        
        # Reset adaptive weights for each new sample (important for ablation consistency)
        self.adaptive_weights = {
            'target': self.l_target_coeff,
            'sparsity': self.l_sparsity_coeff,
            'smoothness': self.l_smoothness_coeff,
            'group_sparsity': self.l_group_sparse_coeff
        }
        self.weight_history = []
        
        logger.info(f"Generating Shapley-Adaptive explanation: class {label} → {target_class}")
        
        # Step 1: Compute channel importance using SHAP (if enabled)
        # We rank by the *original* class to find which channels drive the
        # current (incorrect) prediction — those need to be modified.
        if self.use_shapley:
            ranked_channels, channel_importance = self._compute_shapley_importance(data, label)
            
            # Aggregate to group level
            group_importance = self._aggregate_group_importance(channel_importance)
            
            # Select influential groups
            selected_groups = self._select_influential_groups(group_importance)
        else:
            # Use all groups
            selected_groups = list(self.sensor_groups.keys())
            logger.info(f"SHAP disabled - using all {len(selected_groups)} groups")
        
        # Step 2: Find nearest neighbor from target class
        nearest_neighbor = self._find_nearest_neighbor(data, target_class)
        
        # Step 3: Create group masks
        group_masks = self._create_group_masks(selected_groups, channels, timesteps)
        
        # Step 4: Adaptive multi-objective optimization
        best_mask, best_counterfactual, actual_iterations = self._adaptive_optimization(
            data, target_class, selected_groups, group_masks, nearest_neighbor
        )
        
        # Step 5: Refine for stronger sparsity/smoothness while preserving validity
        refined_mask, refined_cf, target_prob = self._refine_counterfactual(
            data, nearest_neighbor, best_mask, best_counterfactual, target_class
        )

        # Convert to numpy
        mask_np = refined_mask.cpu().numpy()
        cf_np = refined_cf.cpu().numpy()
        
        logger.info(f"Optimization completed: {actual_iterations} iterations, "
                   f"confidence={target_prob:.2%}")
        
        return mask_np, cf_np, target_prob, actual_iterations
    
    def generate_counterfactual(
        self,
        data: Union[torch.Tensor, np.ndarray],
        label: int,
        target_class: int
    ) -> Dict[str, Any]:
        """
        Generate counterfactual with target class specification.
        Compatible with evaluation pipeline.
        
        Args:
            data: Input time series [C, T]
            label: Original class label
            target_class: Target class for counterfactual
            
        Returns:
            Dictionary with saliency_mask, counterfactual, confidence, info
        """
        # Generate explanation
        mask, counterfactual, target_prob, actual_iterations = self.generate_saliency(
            data,
            label,
            target_class=target_class
        )
        
        return {
            'saliency_mask': mask,
            'counterfactual': counterfactual,
            'confidence': target_prob,
            'info': {
                'iterations': actual_iterations,
                'target_class': target_class,
                'original_label': label,
                'method': 'Shapley-Adaptive',
                'use_shapley': self.use_shapley,
                'group_level': self.group_level
            }
        }
