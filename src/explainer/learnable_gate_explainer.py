"""
Learnable Gate Counterfactual Explainer

Extends the Shapley-Adaptive approach with learnable per-group gate parameters
that are jointly optimized alongside the perturbation mask. This allows the
optimizer to *prune* initially-selected groups during optimization:

    SHAP selects K candidate groups  →  learnable gates reduce to K' ≤ K active groups

Each group receives a scalar gate logit passed through a hard-sigmoid.
A gate sparsity loss (L1 + binarization) drives unused gates toward 0,
effectively removing their channels from the counterfactual search space.

Compatible with the evaluation pipeline (evaluation_utils.py).
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
from tslearn.neighbors import KNeighborsTimeSeries

logger = logging.getLogger(__name__)


def tv_norm(signal, tv_beta=3):
    """Total-variation norm for temporal smoothness."""
    signal = signal.flatten()
    return torch.mean(torch.abs(signal[:-1] - signal[1:]).pow(tv_beta))


class LearnableGateExplainer:
    """
    Counterfactual explainer with learnable per-group gates.

    Pipeline:
        1. (Optional) SHAP ranks channel groups and pre-selects the top-K.
        2. Each selected group gets a learnable gate parameter  g_k ∈ ℝ.
        3. During optimization the effective mask is:
               gated_mask = mask_sigmoid ⊙ expand(σ(g_k))
           where σ is sigmoid and expand broadcasts each scalar gate to
           all (channels, timesteps) belonging to group k.
        4. A gate-sparsity loss encourages gates toward 0 or 1, and an L1
           penalty on gate activations drives unnecessary groups to 0.
        5. After convergence, groups whose gate < prune_threshold are
           reported as pruned — their channels are fully excluded from the
           final counterfactual delta.
    """

    def __init__(
        self,
        background_data: np.ndarray,
        background_label: np.ndarray,
        predict_fn,
        enable_wandb: bool = False,
        args=None,
        use_cuda: bool = True,
    ):
        self.background_data = (
            torch.FloatTensor(background_data)
            if isinstance(background_data, np.ndarray)
            else background_data
        )
        self.background_label = background_label
        self.predict_fn = predict_fn
        self.enable_wandb = enable_wandb
        self.args = args

        # Device
        self.device = torch.device(
            'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
        )
        self.background_data = self.background_data.to(self.device)

        # Hyperparameters – optimisation
        self.max_itr = getattr(args, 'max_itr', 5000)
        self.learning_rate = getattr(args, 'learning_rate', 0.01)
        self.gate_lr_multiplier = getattr(args, 'gate_lr_multiplier', 2.0)
        self.enable_lr_decay = getattr(args, 'enable_lr_decay', True)
        self.lr_decay = getattr(args, 'lr_decay', 0.9991)

        # Loss coefficients
        self.l_target_coeff = getattr(args, 'l_max_coeff', 1.0)
        self.l_sparsity_coeff = getattr(args, 'l_budget_coeff', 0.5)
        self.l_smoothness_coeff = getattr(args, 'l_tv_norm_coeff', 0.3)
        self.l_group_sparse_coeff = getattr(args, 'l_group_sparse_coeff', 0.4)
        self.l_gate_coeff = getattr(args, 'l_gate_coeff', 0.6)

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

        # Gate pruning
        #   adaptive_prune=True  → threshold = mean - 1·std of gate activations
        #                          (clamped to [0.15, 0.6] for safety)
        #   adaptive_prune=False → fixed threshold from gate_prune_threshold
        self.adaptive_prune = getattr(args, 'adaptive_prune', True)
        self.prune_threshold_fixed = getattr(args, 'gate_prune_threshold', 0.3)

        # Gate warm-up: gates frozen open for this many iterations,
        # giving the mask time to find a valid counterfactual before pruning begins.
        # Gate loss linearly ramps from 0 → full over the warm-up window.
        self.gate_warmup_itr = getattr(args, 'gate_warmup_itr', 300)

        # Shapley / group configuration
        self.use_shapley = getattr(args, 'use_shapley_ranking', True)
        self.max_groups_ratio = getattr(args, 'max_groups_ratio', 0.6)
        self.group_level = getattr(args, 'group_level', 'sensor')
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
            'group_sparsity': self.l_group_sparse_coeff,
            'gate': self.l_gate_coeff,
        }

        logger.info(f"LearnableGateExplainer initialized on {self.device}")
        logger.info(f"  SHAP ranking : {self.use_shapley}")
        logger.info(f"  Group level  : {self.group_level} ({len(self.sensor_groups)} groups)")
        logger.info(f"  Gate LR mult : {self.gate_lr_multiplier}x")
        if self.adaptive_prune:
            logger.info(f"  Prune mode   : adaptive (mean − 1·std, clamped [0.15, 0.6])")
        else:
            logger.info(f"  Prune thresh : {self.prune_threshold_fixed} (fixed)")
        logger.info(f"  Gate warm-up : {self.gate_warmup_itr} iterations")
        logger.info(f"  Stratified bg: {self._stratified_bg.shape[0]} samples")

    def _prepare_stratified_background(
        self, max_per_class: int = 10
    ) -> torch.Tensor:
        """Build a class-stratified background set for SHAP."""
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

    # ------------------------------------------------------------------
    # Group helpers (identical to ShapleyAdaptiveExplainer)
    # ------------------------------------------------------------------

    def _create_sensor_groups(self, group_level: str) -> Dict[str, List[int]]:
        if group_level == 'sensor':
            return {
                'R_RF': list(range(0, 6)),   'R_HAM': list(range(6, 12)),
                'R_TA': list(range(12, 18)), 'R_GAS': list(range(18, 24)),
                'L_RF': list(range(24, 30)), 'L_HAM': list(range(30, 36)),
                'L_TA': list(range(36, 42)), 'L_GAS': list(range(42, 48)),
            }
        elif group_level == 'modality':
            groups = {}
            names = ['R_RF', 'R_HAM', 'R_TA', 'R_GAS',
                     'L_RF', 'L_HAM', 'L_TA', 'L_GAS']
            for i, s in enumerate(names):
                b = i * 6
                groups[f'{s}_acc'] = list(range(b, b + 3))
                groups[f'{s}_gyr'] = list(range(b + 3, b + 6))
            return groups
        raise ValueError(f"Invalid group_level: {group_level}")

    # ------------------------------------------------------------------
    # SHAP importance
    # ------------------------------------------------------------------

    def _compute_shapley_importance(
        self, x: torch.Tensor, original_class: int
    ) -> Tuple[List[int], np.ndarray]:
        """Channel importance for the *original* class (cached, stratified bg)."""
        # Return cached result if available
        if original_class in self._shap_cache:
            importance = self._shap_cache[original_class]
            ranked = np.argsort(importance)[::-1].tolist()
            logger.info(f"SHAP cache hit for class {original_class}, "
                       f"top-5: {ranked[:5]}")
            return ranked, importance

        try:
            x_tensor = (
                torch.tensor(x.cpu().numpy(), dtype=torch.float32)
                .unsqueeze(0).to(self.device)
            )
            x_tensor.requires_grad = True

            # Lazy-init the GradientExplainer (reused across calls)
            if self._shap_explainer is None:
                class _Wrap(nn.Module):
                    def __init__(self, fn):
                        super().__init__()
                        self.fn = fn
                    def forward(self, x):
                        return self.fn(x)

                wrapper = _Wrap(self.predict_fn)
                wrapper.eval()
                self._shap_explainer = shap.GradientExplainer(
                    wrapper, self._stratified_bg
                )
                logger.info(f"Initialized GradientExplainer with "
                           f"{self._stratified_bg.shape[0]} stratified bg samples")

            sv = self._shap_explainer.shap_values(x_tensor)

            # Extract values for the *original* class
            if isinstance(sv, list):
                sv = sv[original_class]
            if isinstance(sv, torch.Tensor):
                sv = sv.cpu().numpy()
            sv = sv.squeeze(0)

            importance = np.mean(np.abs(sv), axis=-1)
            self._shap_cache[original_class] = importance

            ranked = np.argsort(importance)[::-1].tolist()
            logger.info(f"SHAP computed for original class {original_class}, "
                       f"top-5: {ranked[:5]}")
            return ranked, importance

        except Exception as e:
            logger.warning(f"SHAP failed ({e}), using gradient fallback")
            xt = x.unsqueeze(0).to(self.device).requires_grad_(True)
            out = self.predict_fn(xt)
            out[0, original_class].backward()
            grads = xt.grad.squeeze(0).cpu().numpy()
            importance = np.mean(np.abs(grads), axis=-1)
            self._shap_cache[original_class] = importance
            ranked = np.argsort(importance)[::-1].tolist()
            return ranked, importance

    def _aggregate_group_importance(self, ch_imp: np.ndarray) -> Dict[str, float]:
        """Aggregate to group level using max (one important channel elevates group)."""
        return {
            name: float(np.max([ch_imp[c] for c in chs]))
            for name, chs in self.sensor_groups.items()
        }

    def _select_influential_groups(self, g_imp: Dict[str, float]) -> List[str]:
        sorted_g = sorted(g_imp.items(), key=lambda x: x[1], reverse=True)
        n = max(1, int(len(sorted_g) * self.max_groups_ratio))
        selected = [name for name, _ in sorted_g[:n]]
        logger.info(f"Pre-selected {len(selected)}/{len(sorted_g)} groups: {selected}")
        return selected

    # ------------------------------------------------------------------
    # Nearest neighbour
    # ------------------------------------------------------------------

    def _find_nearest_neighbor(self, x: torch.Tensor, target_class: int) -> torch.Tensor:
        """Find nearest neighbor from target class using tslearn KNN (same as M-CELS)."""
        x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
        dim_nums, ts_length = x_np.shape

        df = pd.DataFrame(self.background_label, columns=['label'])
        target_indices = list(df[df['label'] == target_class].index.values)

        if len(target_indices) == 0:
            return self.background_data[0]

        bg_np = self.background_data.cpu().numpy() if isinstance(self.background_data, torch.Tensor) else self.background_data
        target_data = bg_np[target_indices]

        knn = KNeighborsTimeSeries(n_neighbors=1, metric='euclidean')
        knn.fit(target_data)
        dist, ind = knn.kneighbors(
            x_np.reshape(1, dim_nums, ts_length),
            return_distance=True
        )

        original_idx = df[df['label'] == target_class].index[ind[0][0]]
        nn_np = bg_np[original_idx]
        return torch.tensor(nn_np, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # Core: learnable-gate optimization
    # ------------------------------------------------------------------

    def _build_gate_tensor(
        self,
        gate_logits: torch.Tensor,
        selected_groups: List[str],
        channels: int,
        timesteps: int,
    ) -> torch.Tensor:
        """Expand per-group scalar gate activations to [C, T] tensor."""
        gate_map = torch.zeros(channels, timesteps, device=self.device)
        gate_activations = torch.sigmoid(gate_logits)

        for idx, name in enumerate(selected_groups):
            ch_indices = self.sensor_groups[name]
            gate_map[ch_indices, :] = gate_activations[idx]

        return gate_map

    def _gate_sparsity_loss(self, gate_logits: torch.Tensor) -> torch.Tensor:
        """Combined L1 + binarisation loss on gate activations.

        L1 drives unused gates toward 0.
        Binarisation  min(σ(g), 1-σ(g))  pushes gates to be decisive (0 or 1).
        """
        g = torch.sigmoid(gate_logits)
        l1 = torch.mean(g)
        binarisation = torch.mean(torch.min(g, 1.0 - g))
        return l1 + binarisation

    def _adaptive_optimization(
        self,
        x: torch.Tensor,
        target_class: int,
        selected_groups: List[str],
        nearest_neighbor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, Dict[str, float]]:
        """
        Joint optimization of perturbation mask + group gate logits.

        Returns:
            (best_mask, best_cf, actual_iterations, final_gate_values)
        """
        channels, timesteps = x.shape
        num_gates = len(selected_groups)

        # Learnable parameters --------------------------------------------------
        # Mask logits
        mask_logits = torch.tensor(
            np.random.uniform(0, 1, (channels, timesteps)),
            dtype=torch.float32, device=self.device, requires_grad=True,
        )
        # Gate logits – initialised at +2 so σ(g)≈0.88 (start open, let optimizer close)
        gate_logits = nn.Parameter(
            torch.full((num_gates,), 2.0, device=self.device)
        )

        # Optimizer with different LR for gates
        optimizer = optim.AdamW([
            {'params': [mask_logits], 'lr': self.learning_rate},
            {'params': [gate_logits], 'lr': self.learning_rate * self.gate_lr_multiplier},
        ])

        scheduler = None
        if self.enable_lr_decay:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay)

        # Create group masks for group-sparsity loss
        group_masks = {}
        for name in selected_groups:
            m = torch.zeros(channels, timesteps, device=self.device)
            m[self.sensor_groups[name], :] = 1.0
            group_masks[name] = m

        # Tracking
        best_mask = torch.zeros_like(mask_logits)
        best_cf = x.clone()
        best_confidence = 0.0
        best_loss = float('inf')
        patience_counter = 0
        max_patience = 200          # More patience — gates need time to settle
        imp_threshold = 0.001

        warmup = self.gate_warmup_itr
        gate_ramp_window = 200      # Gates ramp 0→1 over this many iters after warm-up
        # Min iterations before confidence-based early stop:
        # must be past warm-up + full ramp + some buffer for gates to prune
        min_itr_for_stop = warmup + gate_ramp_window + 100

        i = 0
        while i <= self.max_itr:
            # ----------------------------------------------------------
            # Phase logic:  i < warmup  → gates frozen open (mask-only)
            #               i >= warmup → gates unfrozen, loss ramps in
            # ----------------------------------------------------------
            in_warmup = (i < warmup)

            # Gate ramp factor: 0 during warm-up, linearly 0→1 over
            # gate_ramp_window iterations after warm-up ends, then stays 1.
            if in_warmup:
                gate_ramp = 0.0
            else:
                ramp_progress = (i - warmup) / max(gate_ramp_window, 1)
                gate_ramp = min(ramp_progress, 1.0)

            # Build gated mask
            mask_sigmoid = torch.sigmoid(mask_logits)

            if in_warmup:
                # During warm-up: all selected groups fully open (gate=1)
                gate_tensor = torch.zeros(channels, timesteps, device=self.device)
                for name in selected_groups:
                    gate_tensor[self.sensor_groups[name], :] = 1.0
            else:
                gate_tensor = self._build_gate_tensor(
                    gate_logits, selected_groups, channels, timesteps
                )
            gated_mask = mask_sigmoid * gate_tensor

            # Counterfactual
            cf = x * (1.0 - gated_mask) + nearest_neighbor * gated_mask

            # Prediction
            logits = self.predict_fn(cf.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)
            target_prob = probs[0, target_class]

            # ---- Losses ----
            loss_target = 1.0 - target_prob
            loss_sparsity = torch.mean(torch.abs(gated_mask))
            loss_smooth = tv_norm(gated_mask, tv_beta=3)

            # # Group-sparsity (mean activation per group)
            # loss_group = torch.tensor(0.0, device=self.device)
            # for name, gm in group_masks.items():
            #     loss_group = loss_group + torch.sum(gated_mask * gm) / torch.sum(gm)
            # loss_group = loss_group / len(group_masks)

            # Gate-sparsity — scaled by ramp factor AND by target confidence.
            # Gates should only close when the counterfactual already works.
            # Use a hard threshold: no gate pressure until confidence > 50%
            confidence_gate = max(0.0, (target_prob.item() - 0.5) / (self.target_threshold - 0.5)) if target_prob.item() > 0.5 else 0.0
            confidence_gate = min(confidence_gate, 1.0)
            loss_gate = self._gate_sparsity_loss(gate_logits) * gate_ramp * confidence_gate

            loss = (
                self.adaptive_weights['target'] * loss_target
                + self.adaptive_weights['sparsity'] * loss_sparsity
                + self.adaptive_weights['smoothness'] * loss_smooth
                # + self.adaptive_weights['group_sparsity'] * loss_group
                + self.adaptive_weights['gate'] * loss_gate
            )

            # Adaptive weight updates (only after warm-up)
            if not in_warmup:
                if target_prob.item() >= self.target_threshold:
                    self.adaptive_weights['target'] *= 0.95
                    self.adaptive_weights['sparsity'] *= 1.05
                    self.adaptive_weights['smoothness'] *= 1.05
                    self.adaptive_weights['gate'] *= 1.02   # Gentler gate ramp-up
                else:
                    self.adaptive_weights['target'] *= 1.05
                    self.adaptive_weights['sparsity'] *= 0.95
                    self.adaptive_weights['gate'] *= 0.98  # Gently reduce gate pressure when struggling

            for k in self.adaptive_weights:
                self.adaptive_weights[k] = float(np.clip(self.adaptive_weights[k], 0.1, 2.0))

            # Early-stopping bookkeeping (only after warm-up AND gate ramp)
            ramp_done = (i >= warmup + gate_ramp_window)
            if ramp_done:
                if best_loss - loss.item() < imp_threshold:
                    patience_counter += 1
                else:
                    patience_counter = 0
                    best_loss = loss.item()

            if target_prob.item() > best_confidence:
                best_confidence = target_prob.item()
                best_mask = gated_mask.clone().detach()
                best_cf = cf.clone().detach()

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # During warm-up, zero out gate gradients to keep them frozen
            if in_warmup:
                gate_logits.grad = None

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            mask_logits.data.clamp_(-5, 5)
            # Clamp gate logits: floor at -1.0 (sigmoid ≈ 0.27) so gates stay
            # recoverable during optimization. Hard prune happens post-loop.
            gate_logits.data.clamp_(-1.0, 5.0)

            # Logging
            if i % 200 == 0:
                gate_vals = torch.sigmoid(gate_logits).detach().cpu().numpy()
                active = (gate_vals >= self.prune_threshold_fixed).sum()
                phase = "WARM-UP" if in_warmup else f"GATE (ramp={gate_ramp:.2f})"
                logger.debug(
                    f"Iter {i} [{phase}]: loss={loss.item():.4f}  "
                    f"p(target)={target_prob.item():.4f}  "
                    f"gates_active={active}/{num_gates}  "
                    f"gate_vals={np.round(gate_vals, 2).tolist()}"
                )

            # Stopping conditions — never during warm-up or ramp
            if ramp_done:
                if patience_counter >= max_patience:
                    logger.info(f"Early stopping at iteration {i}")
                    break
                if best_confidence >= self.min_confidence and i >= min_itr_for_stop:
                    logger.info(f"Confidence {best_confidence:.2%} reached at iter {i}")
                    break

            i += 1

        actual_iterations = i

        # ---------- Post-optimization analysis ----------
        final_gate_activations = torch.sigmoid(gate_logits).detach().cpu().numpy()

        # Determine pruning threshold
        if self.adaptive_prune:
            g_mean = float(np.mean(final_gate_activations))
            g_std  = float(np.std(final_gate_activations))
            prune_thresh = float(np.clip(g_mean - g_std, 0.15, 0.6))
            print(f"  Adaptive prune: mean={g_mean:.3f}, std={g_std:.3f} "
                  f"→ threshold={prune_thresh:.3f}")
        else:
            prune_thresh = self.prune_threshold_fixed
            print(f"  Fixed prune: threshold={prune_thresh:.3f}")

        gate_values = {}
        active_groups = []
        pruned_groups = []

        for idx, name in enumerate(selected_groups):
            gv = float(final_gate_activations[idx])
            gate_values[name] = gv
            if gv >= prune_thresh:
                active_groups.append(name)
            else:
                pruned_groups.append(name)

        # Safety: if ALL groups would be pruned, keep the top-1 by gate value
        if len(active_groups) == 0 and len(selected_groups) > 0:
            best_name = max(gate_values, key=gate_values.get)
            active_groups.append(best_name)
            pruned_groups.remove(best_name)
            print(f"  ⚠ All groups below threshold — kept top group: {best_name}")

        # Zero-out pruned groups in the best mask
        for name in pruned_groups:
            ch = self.sensor_groups[name]
            best_mask[ch, :] = 0.0
            # Also zero the counterfactual delta for pruned channels
            best_cf[ch, :] = x[ch, :]

        # Collect active channels
        active_channels = []
        for name in active_groups:
            active_channels.extend(self.sensor_groups[name])
        active_channels = sorted(set(active_channels))

        print(f"  Gate results: {len(active_groups)}/{len(selected_groups)} groups active "
              f"(pruned {len(pruned_groups)})")
        for name, gv in gate_values.items():
            status = "✓ ACTIVE" if gv >= prune_thresh else "✗ PRUNED"
            print(f"    {name:12s}  gate={gv:.3f}  {status}")
        print(f"  Active channels: {len(active_channels)}/{channels}")

        logger.info(f"Active groups: {active_groups}")
        logger.info(f"Pruned groups: {pruned_groups}")

        return best_mask, best_cf, actual_iterations, gate_values

    def _refine_counterfactual(
        self,
        x: torch.Tensor,
        nearest_neighbor: torch.Tensor,
        best_mask: torch.Tensor,
        best_cf: torch.Tensor,
        target_class: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Refine CF by smoothing+thresholding mask; pick sparsest valid candidate."""
        with torch.no_grad():
            logits = self.predict_fn(best_cf.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)
            base_prob = probs[0, target_class].item()
            base_pred = torch.argmax(probs, dim=1).item()

        chosen_mask = best_mask.clone().detach()
        chosen_cf = best_cf.clone().detach()
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

    # ------------------------------------------------------------------
    # Public API (compatible with evaluation_utils)
    # ------------------------------------------------------------------

    def generate_saliency(
        self,
        data: Union[torch.Tensor, np.ndarray],
        label: int,
        target_class: Optional[int] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """Generate counterfactual explanation.

        Returns:
            (binary_mask [C,T], counterfactual [C,T], target_prob, iterations)
        """
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32, device=self.device)
        else:
            data = data.to(self.device)

        channels, timesteps = data.shape

        if target_class is None:
            with torch.no_grad():
                logits = self.predict_fn(data.unsqueeze(0))
                probs = torch.softmax(logits, dim=1)
                target_class = torch.argsort(probs, descending=True)[0, 1].item()

        # Step 1: SHAP-based group pre-selection (use *original* class)
        if self.use_shapley:
            _, ch_importance = self._compute_shapley_importance(data, label)
            g_importance = self._aggregate_group_importance(ch_importance)
            selected_groups = self._select_influential_groups(g_importance)
        else:
            selected_groups = list(self.sensor_groups.keys())

        # Step 2: Nearest neighbour from target class
        nn = self._find_nearest_neighbor(data, target_class)

        # Step 3: Joint mask + gate optimization
        best_mask, best_cf, iters, gate_values = self._adaptive_optimization(
            data, target_class, selected_groups, nn
        )

        # Refine for stronger sparsity/smoothness while preserving validity
        refined_mask, refined_cf, target_prob = self._refine_counterfactual(
            data, nn, best_mask, best_cf, target_class
        )
        mask_np = refined_mask.cpu().numpy()
        cf_np = refined_cf.cpu().numpy()

        logger.info(f"Done: {iters} iters, confidence={target_prob:.2%}")
        return mask_np, cf_np, target_prob, iters

    def generate_counterfactual(
        self,
        data: Union[torch.Tensor, np.ndarray],
        label: int,
        target_class: int,
    ) -> Dict[str, Any]:
        """Generate counterfactual — compatible with evaluation pipeline.

        Returns dict with keys: saliency_mask, counterfactual, confidence, info
        """
        if isinstance(data, np.ndarray):
            data_t = torch.tensor(data, dtype=torch.float32, device=self.device)
        else:
            data_t = data.to(self.device)

        channels, timesteps = data_t.shape

        if target_class is None:
            with torch.no_grad():
                logits = self.predict_fn(data_t.unsqueeze(0))
                probs = torch.softmax(logits, dim=1)
                target_class = torch.argsort(probs, descending=True)[0, 1].item()

        # Step 1: SHAP group selection (use *original* class)
        if self.use_shapley:
            _, ch_importance = self._compute_shapley_importance(data_t, label)
            g_importance = self._aggregate_group_importance(ch_importance)
            selected_groups = self._select_influential_groups(g_importance)
        else:
            selected_groups = list(self.sensor_groups.keys())

        # Step 2: Nearest neighbour
        nn = self._find_nearest_neighbor(data_t, target_class)

        # Step 3: Joint optimisation with learnable gates
        best_mask, best_cf, iters, gate_values = self._adaptive_optimization(
            data_t, target_class, selected_groups, nn
        )

        refined_mask, refined_cf, target_prob = self._refine_counterfactual(
            data_t, nn, best_mask, best_cf, target_class
        )
        mask_np = refined_mask.cpu().numpy()
        cf_np = refined_cf.cpu().numpy()

        # Determine which groups survived pruning (same logic as _adaptive_optimization)
        if self.adaptive_prune:
            g_vals = np.array(list(gate_values.values()))
            prune_thresh = float(np.clip(g_vals.mean() - g_vals.std(), 0.15, 0.6))
        else:
            prune_thresh = self.prune_threshold_fixed
        active_groups = [g for g, v in gate_values.items() if v >= prune_thresh]
        pruned_groups = [g for g, v in gate_values.items() if v < prune_thresh]

        return {
            'saliency_mask': mask_np,
            'counterfactual': cf_np,
            'confidence': target_prob,
            'info': {
                'iterations': iters,
                'target_class': target_class,
                'original_label': label,
                'method': 'Learnable-Gate',
                'use_shapley': self.use_shapley,
                'group_level': self.group_level,
                'gate_values': gate_values,
                'active_groups': active_groups,
                'pruned_groups': pruned_groups,
                'groups_selected': len(selected_groups),
                'groups_active': len(active_groups),
                'groups_pruned': len(pruned_groups),
            },
        }
