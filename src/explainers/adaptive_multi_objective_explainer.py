"""
Adaptive Multi-Objective Channel Group Explainer.

This module implements an advanced explainer that:
1. Identifies the most influential channel groups using Shapley values
2. Selects approximately half of the groups for optimization
3. Uses adaptive multi-objective loss with dynamic weighting
4. Optimizes for target prediction, sparsity, and smoothness
5. Maintains counterfactual quality above threshold

Author: E.C. Chukwu
Date: November 5, 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

from .shapley_ranking import ShapleyChannelRanker, create_default_sensor_groups

logger = logging.getLogger(__name__)


class AdaptiveMultiObjectiveExplainer:
    """
    Adaptive Multi-Objective Channel Group Explainer.
    
    This explainer combines channel importance ranking with adaptive multi-objective
    optimization to generate high-quality counterfactual explanations using
    approximately half of the most influential channel groups.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        background_data: Optional[torch.Tensor] = None,
        device: str = 'auto',
        args=None
    ):
        """
        Initialize the adaptive multi-objective explainer.
        
        Args:
            model: Trained PyTorch model
            background_data: Background dataset for Shapley computation
            device: Computing device ('auto', 'cuda', 'cpu')
            args: Configuration arguments
        """
        self.model = model
        self.background_data = background_data
        self.args = args
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Move model to device
        self.model = self.model.to(self.device)
        if background_data is not None:
            if isinstance(background_data, np.ndarray):
                self.background_data = torch.tensor(background_data, dtype=torch.float32).to(self.device)
            else:
                self.background_data = background_data.to(self.device)
        
        # Set up channel groups
        group_level = getattr(args, 'group_level', 'sensor')
        self.sensor_groups = create_default_sensor_groups(group_level)
        
        # Initialize Shapley ranker
        self.shapley_ranker = ShapleyChannelRanker(
            model=self.model,
            background_data=self.background_data,
            max_evals=getattr(args, 'shapley_max_evals', 100),
            random_seed=getattr(args, 'seed_value', 42) if getattr(args, 'enable_seed', True) else None
        )
        
        # Initialize adaptive loss function
        self.loss_fn = AdaptiveMultiObjectiveLoss(
            lambda_target=getattr(args, 'l_target_coeff', 1.0),
            lambda_sparsity=getattr(args, 'l_sparsity_coeff', 0.5),
            lambda_smoothness=getattr(args, 'l_smoothness_coeff', 0.3),
            lambda_group_sparse=getattr(args, 'l_group_sparse_coeff', 0.4),
            sensor_groups=self.sensor_groups,
            device=self.device
        ).to(self.device)
        
        # Adaptive weighting parameters
        self.adaptive_weights = AdaptiveWeightManager(
            initial_weights={
                'target': getattr(args, 'l_target_coeff', 1.0),
                'sparsity': getattr(args, 'l_sparsity_coeff', 0.5),
                'smoothness': getattr(args, 'l_smoothness_coeff', 0.3),
                'group_sparsity': getattr(args, 'l_group_sparse_coeff', 0.4),
                'group_gates': getattr(args, 'l_group_gates_coeff', 0.2)
            },
            adaptation_rate=getattr(args, 'weight_adaptation_rate', 0.1),
            target_threshold=getattr(args, 'target_threshold', 0.8)
        )
        
        # Quality thresholds
        self.min_confidence = getattr(args, 'min_target_probability', 0.7)
        self.max_groups_ratio = getattr(args, 'max_groups_ratio', 0.5)  # Use 50% of groups
        
        # Results storage
        self.results = {}
        
        logger.info(f"Adaptive Multi-Objective Explainer initialized on {self.device}")
    
    def generate_saliency(
        self,
        data: Union[torch.Tensor, np.ndarray] = None,
        label: Optional[int] = None,
        target_class: Optional[int] = None,
        save_dir: Optional[str] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, float, List[int], Dict[str, Any]]:
        """
        Generate adaptive multi-objective counterfactual explanation.
        
        Args:
            data: Input sample data
            label: Original class label
            target_class: Target class for counterfactual
            save_dir: Directory to save results
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (mask, cf_input, target_prob, selected_channels, metrics)
        """
        # Handle input format
        if data is not None:
            if isinstance(data, np.ndarray):
                x = torch.tensor(data, dtype=torch.float32)
            else:
                x = data.clone()
        else:
            x = kwargs.get('x')
            if x is None:
                raise ValueError("No input data provided")
        
        # Ensure proper input format
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # Add batch dimension
        
        x = x.to(self.device)
        batch_size, channels, timesteps = x.shape

        # if target_class is None:
        #     target_class = 1 - label if label is not None else 0

        logger.info(f"Generating adaptive multi-objective explanation for input shape {x.shape}, target class {target_class}")
        
        # Step 1: Rank channels using Shapley values
        ranked_channels, importance_scores = self.shapley_ranker.rank_channels_shapley(x[0], target_class)
        
        # Step 2: Aggregate group importance and select top groups
        group_importance = self._aggregate_group_importance(importance_scores)
        selected_groups = self._select_influential_groups(group_importance)
        
        # Step 3: Create group masks for optimization
        group_masks = self._create_group_masks(selected_groups, channels)
        
        
        # Step 4: Adaptive multi-objective optimization
        best_mask, best_counterfactual, optimization_trace = self._adaptive_optimization(
            x, target_class, group_masks, selected_groups
        )
        
        # Step 5: Evaluate final counterfactual
        final_evaluation = self._evaluate_counterfactual(x, best_counterfactual, target_class)
        
        # Step 6: Extract results
        selected_channels = self._extract_selected_channels(best_mask, selected_groups)
        
        # Prepare return values
        mask_np = best_mask[0].cpu().numpy()
        cf_input = best_counterfactual[0].cpu().numpy()
        target_prob = final_evaluation['confidence']
        
        # Create binary mask for evaluation
        binary_mask = (mask_np >= 0.5).astype(float)
        
        # Minimal metrics dictionary - comprehensive metrics handled by EvaluationMetrics class
        metrics = {
            'predicted_class': final_evaluation['predicted_class'],
            'original_class': final_evaluation['original_class'],
            'success': final_evaluation['valid'],
            'num_selected_channels': len(selected_channels),
            'num_groups': len(selected_groups),
            'method': 'Adaptive Multi-Objective'
        }
        
        # Store detailed results
        self.results = {
            'original_input': x[0].cpu().numpy(),
            'counterfactual': cf_input,
            'mask': mask_np,
            'importance_scores': importance_scores,
            'selected_channels': selected_channels,
            'selected_groups': selected_groups,
            'group_importance': group_importance,
            'optimization_trace': optimization_trace,
            'evaluation': final_evaluation,
            'target_class': target_class,
            'input_shape': x.shape,
            'adaptive_weights_history': self.adaptive_weights.get_history(),
            'method': 'Adaptive Multi-Objective',
            'args': vars(self.args) if hasattr(self.args, '__dict__') else str(self.args)
        }
        
        logger.info(f"Adaptive optimization completed. Success: {metrics['success']}, "
                   f"Used {len(selected_channels)} channels from {len(selected_groups)} groups")
        
        return binary_mask.flatten(), cf_input, target_prob, selected_channels, metrics
    
    def _aggregate_group_importance(self, importance_scores: np.ndarray) -> Dict[str, float]:
        """Aggregate channel importance scores to group level."""
        group_importance = {}
        
        for group_name, channel_indices in self.sensor_groups.items():
            # Use mean importance as group importance
            group_score = np.mean([importance_scores[ch] for ch in channel_indices])
            group_importance[group_name] = group_score
        
        return group_importance
    
    def _select_influential_groups(self, group_importance: Dict[str, float]) -> List[str]:
        """Select approximately half of the most influential groups."""
        # Sort groups by importance
        sorted_groups = sorted(group_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Select top ~50% of groups (with minimum 1 group)
        total_groups = len(sorted_groups)
        num_select = max(1, int(total_groups * self.max_groups_ratio))
        
        selected_groups = [name for name, _ in sorted_groups[:num_select]]
        
        logger.info(f"Selected {len(selected_groups)}/{total_groups} most influential groups: {selected_groups}")
        
        return selected_groups
    
    def _create_group_masks(self, selected_groups: List[str], channels: int) -> Dict[str, torch.Tensor]:
        """Create binary masks for selected groups."""
        group_masks = {}
        # print(f"Channels: {channels}")
        for group_name in selected_groups:
            if group_name in self.sensor_groups:
                mask = torch.zeros(channels, dtype=torch.float32, device=self.device)
                group_channels = self.sensor_groups[group_name]
                mask[group_channels] = 1.0
                group_masks[group_name] = mask
        
        return group_masks
    
    def _find_nearest_unlike_neighbor(self, x: torch.Tensor, target_class: int) -> torch.Tensor:
        """
        Find nearest unlike neighbor from background data for the target class.
        
        Args:
            x: Input sample (channels, timesteps)
            target_class: Target class for counterfactual
            
        Returns:
            Nearest sample from target class in background data
        """
        if self.background_data is None:
            # If no background data, return a zero tensor as fallback
            logger.warning("No background data available, using zero substitution")
            return torch.zeros_like(x)
        
        # Get model predictions for background data to find samples of target class
        self.model.eval()
        with torch.no_grad():
            # Handle background data format
            if len(self.background_data.shape) == 2:
                # (samples, features) -> (samples, channels, timesteps)
                bg_data = self.background_data.view(self.background_data.shape[0], x.shape[0], -1)
            else:
                bg_data = self.background_data
            
            # Get predictions for background samples
            bg_outputs = self.model(bg_data)
            bg_predictions = torch.argmax(bg_outputs, dim=1)
            
            # Find samples that belong to target class
            target_class_mask = bg_predictions == target_class
            
            if not target_class_mask.any():
                # No samples of target class found, use random background sample
                logger.warning(f"No samples of target class {target_class} found in background data")
                return bg_data[0]
            
            # Filter background data to target class samples
            target_class_samples = bg_data[target_class_mask]
            
            # Compute L2 distances to find nearest neighbor
            x_expanded = x.unsqueeze(0).expand(target_class_samples.shape[0], -1, -1)
            distances = torch.norm(target_class_samples - x_expanded, dim=(1, 2))
            
            # Return nearest sample
            nearest_idx = torch.argmin(distances)
            return target_class_samples[nearest_idx]
    
    def _adaptive_optimization(
        self,
        x: torch.Tensor,
        target_class: int,
        group_masks: Dict[str, torch.Tensor],
        selected_groups: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Perform adaptive multi-objective optimization with dynamic group removal."""
        batch_size, channels, timesteps = x.shape
        
        # Find nearest unlike neighbor for counterfactual substitution
        nearest_unlike_neighbor = self._find_nearest_unlike_neighbor(x[0], target_class)
        
        # Initialize restricted learnable channel mask (only selected group channels)
        mask = self._create_restricted_channel_mask(selected_groups, batch_size, timesteps)
        
        # Initialize learnable group gates (one per selected group)
        group_gates = torch.ones(len(selected_groups), device=self.device, requires_grad=True)
        group_gates = torch.nn.Parameter(group_gates)
        
        # Initialize optimizer with adaptive learning rate for both mask and group gates
        optimizer = optim.AdamW([mask, group_gates], lr=getattr(self.args, 'lr', 0.01), weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100)
        # print(f"   Optimizer initialized with params: {optimizer.param_groups}")
        # Optimization tracking
        trace = {
            'loss_history': [],
            'component_losses': [],
            'confidence_history': [],
            'sparsity_history': [],
            'weight_history': [],
            'group_usage': [],
            'active_groups': [],
            'group_gates_history': []
        }
        
        # Initialize best tracking with full-size tensors
        initial_full_mask = self._expand_restricted_mask_to_full(torch.sigmoid(mask), channels)
        best_mask = initial_full_mask.clone()
        best_counterfactual = x.clone()
        best_confidence = 0.0
        
        max_iterations = getattr(self.args, 'max_itr', 1000)
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            # Apply sigmoid to get mask in [0,1] range
            mask_sigmoid = torch.sigmoid(mask)
            
            # Expand restricted mask to full channel dimensions
            full_mask_sigmoid = self._expand_restricted_mask_to_full(mask_sigmoid, channels)
            
            # Apply sigmoid to group gates and create group-level mask
            group_gates_sigmoid = torch.sigmoid(group_gates)
            # print(f"Iteration {iteration}: Group gates sigmoid values: {group_gates_sigmoid}")
            
            # Create combined mask that incorporates group-level gating
            combined_mask = self._apply_group_gating(full_mask_sigmoid, group_gates_sigmoid, selected_groups, group_masks)
            
            # Generate counterfactual using nearest unlike neighbor
            x_cf = x * (1 - combined_mask) + nearest_unlike_neighbor.unsqueeze(0) * combined_mask
            
            # Forward pass
            output = self.model(x_cf)
            target_prob = torch.softmax(output, dim=1)[0, target_class]
            
            # Compute loss components (now uses combined mask)
            loss_components = self.loss_fn(x_cf, x, combined_mask, output, target_class, group_masks)
            
            # Add group gate regularization to encourage sparse group selection
            group_gate_loss = self._compute_group_gate_loss(group_gates_sigmoid)
            loss_components['group_gates'] = group_gate_loss
            
            # Get adaptive weights
            current_weights = self.adaptive_weights.update_weights(
                loss_components, target_prob.item(), iteration
            )
            
            # Weighted total loss
            total_loss = sum(current_weights[key] * loss for key, loss in loss_components.items())
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for both mask and group gates
            torch.nn.utils.clip_grad_norm_([mask, group_gates], max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Track optimization
            with torch.no_grad():
                current_confidence = target_prob.item()
                current_sparsity = 1.0 - torch.mean(combined_mask).item()
                
                trace['loss_history'].append(total_loss.item())
                trace['component_losses'].append({k: v.item() for k, v in loss_components.items()})
                trace['confidence_history'].append(current_confidence)
                trace['sparsity_history'].append(current_sparsity)
                trace['weight_history'].append(current_weights.copy())
                
                # Track group usage and active groups
                group_usage = self._compute_group_usage(combined_mask[0], selected_groups)
                active_groups = self._get_active_groups(group_gates_sigmoid, selected_groups)
                trace['group_usage'].append(group_usage)
                trace['active_groups'].append(active_groups)
                trace['group_gates_history'].append(group_gates_sigmoid.cpu().numpy().tolist())
                
                # Update best if improved
                if current_confidence > best_confidence:
                    best_confidence = current_confidence
                    best_mask = combined_mask.clone()  # Store the full expanded mask
                    best_counterfactual = x_cf.clone()
            
            # Early stopping conditions
            if self._should_stop_early(iteration, best_confidence, total_loss.item(), trace):
                logger.info(f"Early stopping at iteration {iteration}, best confidence: {best_confidence:.3f}")
                break
        
        trace['iterations'] = iteration + 1
        trace['final_confidence'] = best_confidence

        return best_mask, best_counterfactual, trace
    
    def _apply_group_gating(
        self, 
        mask: torch.Tensor, 
        group_gates: torch.Tensor, 
        selected_groups: List[str], 
        group_masks: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply group-level gating to the channel mask.
        
        Args:
            mask: Channel-level mask (batch_size, channels, timesteps) - full size
            group_gates: Group-level gates (num_groups,) - corresponds to selected_groups
            selected_groups: List of selected group names
            group_masks: Dictionary of group masks for all groups
            
        Returns:
            Combined mask with group gating applied
        """
        combined_mask = mask.clone()
        
        # Apply group gating only to selected groups
        for i, group_name in enumerate(selected_groups):
            if group_name in group_masks:
                group_mask = group_masks[group_name]  # (channels,) - full size
                group_gate = group_gates[i]  # scalar - corresponds to this selected group

                # Apply group gate to all channels in this group
                # group_mask is 1 for channels in group, 0 for others
                # Multiply by group_gate to allow dynamic group removal
                group_effect = group_mask.unsqueeze(0).unsqueeze(-1) * group_gate  # (1, channels, 1)
                # print(f"Applying group gate for {group_name}: gate value {group_gate.item():.4f}, group effect {group_effect}")
                # Zero out channels in this group if gate is low
                combined_mask = combined_mask * (1 - group_mask.unsqueeze(0).unsqueeze(-1) + group_effect)

        return combined_mask

    def _compute_group_gate_loss(self, group_gates: torch.Tensor) -> torch.Tensor:
        """
        Compute regularization loss for group gates to encourage sparsity.
        
        Args:
            group_gates: Group gate values (num_groups,)
            
        Returns:
            Group gate regularization loss
        """
        # L1 regularization to encourage sparse group selection
        # print(group_gates)
        l1_loss = torch.mean(group_gates)
        # print(f"Group gate L1 loss: {l1_loss.item():.4f}")
        
        # Optional: Add entropy regularization to encourage binary decisions
        entropy_loss = torch.mean(group_gates * torch.log(group_gates + 1e-8) + 
                                 (1 - group_gates) * torch.log(1 - group_gates + 1e-8))
        
        # print(f"Group gate entropy loss: {entropy_loss.item():.4f}")
        
        return l1_loss - 0.1 * entropy_loss  # Negative entropy encourages binary decisions
    
    def _get_active_groups(self, group_gates: torch.Tensor, selected_groups: List[str]) -> List[str]:
        """
        Get list of currently active groups based on gate values.
        
        Args:
            group_gates: Group gate values (num_groups,)
            selected_groups: List of selected group names
            
        Returns:
            List of active group names (gate >= 0.5)
        """
        threshold = 0.5
        active_groups = []
        
        for i, group_name in enumerate(selected_groups):
            if group_gates[i].item() >= threshold:
                active_groups.append(group_name)
        
        return active_groups
    
    def _compute_group_usage(self, mask: torch.Tensor, selected_groups: List[str]) -> Dict[str, float]:
        """Compute usage statistics for each group."""
        group_usage = {}
        
        for group_name in selected_groups:
            if group_name in self.sensor_groups:
                group_channels = self.sensor_groups[group_name]
                group_activation = torch.mean(mask[group_channels]).item()
                group_usage[group_name] = group_activation
        
        return group_usage
    
    def _should_stop_early(
        self, 
        iteration: int, 
        best_validity: float, 
        current_loss: float,
        trace: Dict[str, Any]
    ) -> bool:
        """Determine if optimization should stop early."""
        min_iterations = 100
        
        if iteration < min_iterations:
            return False
        
        # Stop if target confidence achieved
        if best_validity >= self.min_confidence:
            return True
        
        # Stop if loss has converged
        if iteration >= 200:
            recent_losses = trace['loss_history'][-20:]
            if len(recent_losses) >= 20:
                loss_variance = np.var(recent_losses)
                if loss_variance < 1e-6:
                    return True
        
        return False
    
    def _extract_selected_channels(self, mask: torch.Tensor, selected_groups: List[str]) -> List[int]:
        """Extract channels that are significantly activated in the mask."""
        threshold = 0.5
        activated_mask = mask[0] >= threshold  # Remove batch dimension
        
        selected_channels = []
        for group_name in selected_groups:
            if group_name in self.sensor_groups:
                group_channels = self.sensor_groups[group_name]
                for ch in group_channels:
                    if activated_mask[ch].any():  # If any timestep is activated
                        selected_channels.append(ch)
        
        return sorted(list(set(selected_channels)))

    def _evaluate_counterfactual(
        self,
        x_orig: torch.Tensor,
        x_cf: torch.Tensor,
        target_class: int
    ) -> Dict[str, float]:
        """Basic counterfactual evaluation for optimization purposes."""
        self.model.eval()
        
        with torch.no_grad():
            # Original prediction
            orig_output = self.model(x_orig)
            orig_class = torch.argmax(orig_output, dim=1).item()
            
            # Counterfactual prediction
            cf_output = self.model(x_cf)
            cf_probs = torch.softmax(cf_output, dim=1)
            cf_class = torch.argmax(cf_output, dim=1).item()
            cf_confidence = cf_probs[0, target_class].item()
            
            # Basic validity check for optimization
            valid = cf_class == target_class and cf_confidence >= self.min_confidence
            
        return {
            'valid': valid,
            'predicted_class': cf_class,
            'target_class': target_class,
            'original_class': orig_class,
            'confidence': cf_confidence
        }
            
    def generate_cf_explanation_summary(
        self,
        data: Optional[np.ndarray] = None,
        original_label: Optional[int] = None,
        target_class: Optional[int] = None,
        subject_id: Optional[str] = None,
        injured_foot: Optional[str] = None,
        selected_channels: Optional[List[int]] = None,
        counterfactual_data: Optional[np.ndarray] = None,
        **kwargs
    ) -> str:
        """
        Generate human-readable explanation summary for adaptive multi-objective approach.
        
        Args:
            data: Original input sample data
            original_label: Original class label
            target_class: Target class for counterfactual
            subject_id: Subject identifier
            injured_foot: Injury side information
            selected_channels: List of selected channels
            counterfactual_data: Generated counterfactual data
            **kwargs: Additional parameters
            
        Returns:
            Formatted explanation string
        """
        # Generate counterfactual if not already computed
        if not hasattr(self, 'results') or not self.results:
            if data is not None and target_class is not None:
                self.generate_saliency(data=data, target_class=target_class)
        
        if not hasattr(self, 'results') or not self.results:
            return "No explanation results available."
        
        results = self.results
        
        # Build summary
        success = results.get('evaluation', {}).get('valid', False)
        summary_lines = [
            "=== Adaptive Multi-Objective Counterfactual Explanation ===",
            f"Method: Adaptive Multi-Objective Channel Group Optimization",
            f"Target Class: {results.get('target_class', 'Unknown')}",
            f"Success: {'✓' if success else '✗'}",
            "",
            "Group Selection Strategy:",
            f"  - Total available groups: {len(self.sensor_groups)}",
            f"  - Selected groups: {len(results.get('selected_groups', []))}",
            f"  - Selection ratio: {len(results.get('selected_groups', [])) / len(self.sensor_groups):.1%}",
            f"  - Strategy: Top ~{self.max_groups_ratio:.0%} most influential groups",
            "",
            "Multi-Objective Optimization:",
            f"  - Target prediction maximization",
            f"  - Channel sparsity minimization", 
            f"  - Temporal smoothness enforcement",
            f"  - Group sparsity control",
            f"  - Dynamic group removal via learnable gates",
            ""
        ]
        
        if success:
            evaluation = results.get('evaluation', {})
            summary_lines.extend([
                "Performance Metrics:",
                f"  - Target probability: {evaluation.get('confidence', 0.0):.3f}",
                f"  - Validity: {evaluation.get('validity', 0.0):.3f}",
                f"  - L1 distance: {evaluation.get('l1_distance', 0.0):.3f}",
                f"  - L2 distance: {evaluation.get('l2_distance', 0.0):.3f}",
                ""
            ])
        
        # Selected groups information
        selected_groups = results.get('selected_groups', [])
        group_importance = results.get('group_importance', {})
        
        if selected_groups:
            summary_lines.extend([
                "Selected Channel Groups:",
            ])
            for i, group in enumerate(selected_groups, 1):
                importance = group_importance.get(group, 0.0)
                group_channels = self.sensor_groups.get(group, [])
                summary_lines.append(f"  {i}. {group}: importance={importance:.4f}, channels={len(group_channels)}")
            summary_lines.append("")
        
        # Dynamic group removal information
        trace = results.get('optimization_trace', {})
        if trace and 'active_groups' in trace:
            active_groups_history = trace['active_groups']
            if active_groups_history:
                initial_active = set(active_groups_history[0]) if active_groups_history[0] else set()
                final_active = set(active_groups_history[-1]) if active_groups_history[-1] else set()
                removed_groups = initial_active - final_active
                
                summary_lines.extend([
                    "Dynamic Group Removal:",
                    f"  - Initial active groups: {len(initial_active)}/{len(selected_groups)}",
                    f"  - Final active groups: {len(final_active)}/{len(selected_groups)}",
                    f"  - Removed during optimization: {sorted(list(removed_groups)) if removed_groups else 'None'}",
                    ""
                ])
        
        # Adaptive weight information
        weight_history = results.get('adaptive_weights_history', [])
        if weight_history:
            initial_weights = weight_history[0]['weights'] if weight_history else {}
            final_weights = weight_history[-1]['weights'] if weight_history else {}
            
            summary_lines.extend([
                "Adaptive Weight Evolution:",
            ])
            for component in ['target', 'sparsity', 'smoothness', 'group_sparsity', 'group_gates']:
                initial = initial_weights.get(component, 0.0)
                final = final_weights.get(component, 0.0)
                change = ((final - initial) / initial * 100) if initial > 0 else 0
                summary_lines.append(f"  - {component}: {initial:.3f} → {final:.3f} ({change:+.1f}%)")
            summary_lines.append("")
        
        # Optimization trace summary
        trace = results.get('optimization_trace', {})
        if trace:
            iterations = trace.get('iterations', 0)
            final_validity = trace.get('final_validity', 0.0)
            
            summary_lines.extend([
                "Optimization Summary:",
                f"  - Total iterations: {iterations}",
                f"  - Final validity: {final_validity:.3f}",
                f"  - Convergence: {'Early stopping' if iterations < 300 else 'Max iterations'}",
                ""
            ])
        
        summary_lines.extend([
            "Key Features:",
            "  ✓ Shapley-based group importance ranking",
            "  ✓ Selective group optimization (~50% of groups)",
            "  ✓ Multi-objective loss balancing",
            "  ✓ Adaptive weight management",
            "  ✓ Dynamic group removal during optimization",
            "  ✓ Learnable group gates for adaptive selection",
            "  ✓ Quality threshold enforcement"
        ])
        
        return "\n".join(summary_lines)
    
    def save_results(self, save_path: str) -> None:
        """Save adaptive multi-objective explainer results to file."""
        if not hasattr(self, 'results') or not self.results:
            logger.warning("No results to save")
            return
        
        # Prepare serializable results
        save_data = {}
        for key, value in self.results.items():
            if isinstance(value, (torch.Tensor, np.ndarray)):
                save_data[key] = value.tolist() if hasattr(value, 'tolist') else value.cpu().numpy().tolist()
            elif isinstance(value, dict):
                # Handle nested dictionaries (like optimization_trace)
                save_data[key] = self._serialize_dict(value)
            else:
                save_data[key] = value
        
        # Save to JSON
        import json
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        logger.info(f"Adaptive Multi-Objective results saved to {save_path}")
    
    def _serialize_dict(self, d: Dict) -> Dict:
        """Recursively serialize dictionary with tensors/arrays."""
        serialized = {}
        for key, value in d.items():
            if isinstance(value, (torch.Tensor, np.ndarray)):
                serialized[key] = value.tolist() if hasattr(value, 'tolist') else value.cpu().numpy().tolist()
            elif isinstance(value, dict):
                serialized[key] = self._serialize_dict(value)
            elif isinstance(value, list):
                serialized[key] = [self._serialize_dict(item) if isinstance(item, dict) else item for item in value]
            else:
                serialized[key] = value
        return serialized

    def _create_restricted_channel_mask(
        self, 
        selected_groups: List[str], 
        batch_size: int, 
        timesteps: int
    ) -> torch.Tensor:
        """Create channel mask that only covers selected group channels."""
        
        # Identify channels that belong to selected groups
        selected_channels = []
        for group_name in selected_groups:
            if group_name in self.sensor_groups:
                selected_channels.extend(self.sensor_groups[group_name])
        
        selected_channels = sorted(list(set(selected_channels)))
        num_selected_channels = len(selected_channels)
        
        # Create restricted mask
        restricted_mask = torch.rand(
            batch_size, num_selected_channels, timesteps, 
            device=self.device, requires_grad=True
        )
        
        # Create mapping from restricted mask to full mask
        self.selected_channel_indices = selected_channels
        
        return torch.nn.Parameter(restricted_mask)

    def _expand_restricted_mask_to_full(
        self, 
        restricted_mask: torch.Tensor, 
        full_channels: int
    ) -> torch.Tensor:
        """Expand restricted mask to full channel dimensionality."""
        batch_size, _, timesteps = restricted_mask.shape
        full_mask = torch.zeros(batch_size, full_channels, timesteps, device=self.device)
        
        # Map restricted channels back to full channel indices
        for i, channel_idx in enumerate(self.selected_channel_indices):
            full_mask[:, channel_idx, :] = restricted_mask[:, i, :]
        
        return full_mask

class AdaptiveMultiObjectiveLoss(nn.Module):
    """Multi-objective loss function with adaptive weighting."""
    
    def __init__(
        self,
        lambda_target: float = 1.0,
        lambda_sparsity: float = 0.5,
        lambda_smoothness: float = 0.3,
        lambda_group_sparse: float = 0.4,
        sensor_groups: Dict[str, List[int]] = None,
        device: str = 'cpu'
    ):
        super().__init__()
        self.lambda_target = lambda_target
        self.lambda_sparsity = lambda_sparsity
        self.lambda_smoothness = lambda_smoothness
        self.lambda_group_sparse = lambda_group_sparse
        self.sensor_groups = sensor_groups or {}
        self.device = device
    
    def forward(
        self,
        x_cf: torch.Tensor,
        x_orig: torch.Tensor,
        mask: torch.Tensor,
        model_output: torch.Tensor,
        target_class: int,
        group_masks: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-objective loss components."""
        
        # 1. Target prediction loss (maximize target class probability)
        target_loss = self._compute_target_loss(model_output, target_class)
        
        # 2. Sparsity loss (minimize total selected channels)
        sparsity_loss = self._compute_sparsity_loss(mask)
        
        # 3. Smoothness loss (ensure temporal coherence)
        smoothness_loss = self._compute_smoothness_loss(mask)
        
        # 4. Group sparsity loss (minimize number of active groups)
        group_sparsity_loss = self._compute_group_sparsity_loss(mask, group_masks)
        
        return {
            'target': target_loss,
            'sparsity': sparsity_loss,
            'smoothness': smoothness_loss,
            'group_sparsity': group_sparsity_loss
        }
    
    def _compute_target_loss(self, model_output: torch.Tensor, target_class: int) -> torch.Tensor:
        """Compute loss to maximize target class probability."""
        probs = torch.softmax(model_output, dim=1)
        target_prob = probs[0, target_class]
        return 1.0 - target_prob  # Minimize (1 - target_prob) = maximize target_prob
    
    def _compute_sparsity_loss(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute sparsity loss to minimize total selected channels."""
        return torch.mean(mask)  # L1 norm of mask
    
    def _compute_smoothness_loss(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute smoothness loss for temporal coherence."""
        if mask.shape[-1] <= 1:
            return torch.tensor(0.0, device=mask.device)
        
        # Temporal differences
        temporal_diff = mask[:, :, 1:] - mask[:, :, :-1]
        # print(f"Temporal differences: {temporal_diff.shape}")
        temporal_smoothness = torch.mean(temporal_diff ** 2)
        
        return temporal_smoothness
    
    def _compute_group_sparsity_loss(
        self, 
        mask: torch.Tensor, 
        group_masks: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute group sparsity loss to minimize number of active groups."""
        total_group_activation = 0.0
        
        for group_name, group_mask in group_masks.items():
            # Compute group activation as max activation within group
            group_activation = torch.max(mask[0] * group_mask.unsqueeze(-1))
            total_group_activation += group_activation
        
        totalgact = total_group_activation / len(group_masks)
        # print(f"Total group activation (average): {totalgact:.3f}")

        return totalgact  # Average group activation as loss

class AdaptiveWeightManager:
    """Manages adaptive weighting of loss components during optimization."""
    
    def __init__(
        self,
        initial_weights: Dict[str, float],
        adaptation_rate: float = 0.1,
        target_threshold: float = 0.8
    ):
        self.weights = initial_weights.copy()
        self.initial_weights = initial_weights.copy()
        self.adaptation_rate = adaptation_rate
        self.target_threshold = target_threshold
        self.history = []
    
    def update_weights(
        self,
        loss_components: Dict[str, torch.Tensor],
        target_prob: float,
        iteration: int
    ) -> Dict[str, float]:
        """Update weights based on current performance."""
        
        # Adaptive strategy based on target achievement
        if target_prob >= self.target_threshold:
            # Target achieved, focus more on sparsity and group removal
            self.weights['target'] *= (1.0 - self.adaptation_rate)
            self.weights['sparsity'] *= (1.0 + self.adaptation_rate)
            self.weights['group_sparsity'] *= (1.0 + self.adaptation_rate)
            self.weights['group_gates'] *= (1.0 + self.adaptation_rate)
        else:
            # Target not achieved, focus more on target
            self.weights['target'] *= (1.0 + self.adaptation_rate)
            self.weights['sparsity'] *= (1.0 - self.adaptation_rate * 0.5)
            self.weights['group_sparsity'] *= (1.0 - self.adaptation_rate * 0.5)
            self.weights['group_gates'] *= (1.0 - self.adaptation_rate * 0.3)
        
        # Clamp weights to reasonable ranges
        for key in self.weights:
            self.weights[key] = np.clip(
                self.weights[key], 
                self.initial_weights[key] * 0.1, 
                self.initial_weights[key] * 3.0
            )
        
        # Store history
        self.history.append({
            'iteration': iteration,
            'weights': self.weights.copy(),
            'target_prob': target_prob
        })
        
        return self.weights.copy()
    
    def get_history(self) -> List[Dict]:
        """Get weight adaptation history."""
        return self.history.copy()