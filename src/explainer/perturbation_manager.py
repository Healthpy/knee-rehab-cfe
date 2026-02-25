"""
Perturbation Manager for Tracking Explanation Generation

Manages and tracks perturbations during counterfactual generation,
including distance metrics and confidence scores.
"""

import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from typing import Dict, List, Any


class PerturbationManager:
    """
    Manages perturbations during counterfactual generation
    
    Tracks original signal, perturbations, distances, and confidences
    throughout the optimization process.
    """
    
    def __init__(
        self,
        original_signal: np.ndarray,
        algo: str,
        prediction_prob: float,
        original_label: int,
        sample_id: int
    ):
        """
        Initialize perturbation manager
        
        Args:
            original_signal: Original time series [T*C]
            algo: Algorithm name (e.g., 'mcels', 'cf')
            prediction_prob: Original prediction probability
            original_label: True label
            sample_id: Sample identifier
        """
        self.original_signal = original_signal.flatten()
        self.sample_id = sample_id
        self.algo = algo
        self.prediction_prob = prediction_prob
        self.original_label = original_label
        self.perturbations = [original_signal.flatten()]
        self.rows = []
        self.column_names = []
    
    def add_perturbation(
        self,
        perturbation: np.ndarray,
        step: int,
        confidence: float,
        saliency: np.ndarray,
        **kwargs
    ):
        """
        Add a perturbation step with metrics
        
        Args:
            perturbation: Perturbed time series [T*C]
            step: Iteration step
            confidence: Model confidence on perturbation
            saliency: Saliency/mask used [T*C]
            **kwargs: Additional metrics to track
        """
        # Weighted distance function
        weighted_euc_dist = lambda x, y: euclidean(x, y, w=saliency)
        
        # Initialize column names on first call
        if not self.column_names:
            self.column_names = [
                *[f'f{str(i)}' for i in range(len(self.original_signal))],
                *[f's{str(i)}' for i in range(len(self.original_signal))],
                "type", "sample_id", "algo", "itr", "label", "prob",
                "abs_euc", "abs_dtw", "rel_euc", "rel_dtw",
                "w_rel_euc", "w_rel_dtw", "w_abs_euc", "w_abs_dtw",
                *list(kwargs.keys())
            ]
            # Add original signal row
            self.rows.append([
                *self.original_signal.tolist(),
                *[f"{1}" for _ in self.original_signal.tolist()],
                "o", self.sample_id, self.algo, step,
                self.original_label, confidence,
                "0", "0", "0", "0", "0", "0", "0", "0",
                *["0" for _ in kwargs.keys()]
            ])
        
        step += 1
        perturbation = perturbation.flatten()
        self.perturbations.append(perturbation)
        
        # Build row with all metrics
        row = [*perturbation.tolist()]
        row = [*row, *saliency.tolist()]
        row = [*row, "p", self.sample_id, self.algo, step, self.original_label, confidence]
        
        # Compute distances
        abs_euc = euclidean(self.original_signal, perturbation)
        abs_dtw = fastdtw(self.original_signal, perturbation, dist=euclidean)[0]
        
        if len(self.perturbations) >= 2:
            rel_euc = euclidean(self.perturbations[-2], perturbation)
            rel_dtw = fastdtw(self.perturbations[-2], perturbation, dist=euclidean)[0]
        else:
            rel_euc = 0
            rel_dtw = 0
        
        # Weighted distances
        w_abs_euc = weighted_euc_dist(self.original_signal, perturbation)
        w_abs_dtw = fastdtw(self.original_signal, perturbation, dist=weighted_euc_dist)[0]
        
        if len(self.perturbations) >= 2:
            w_rel_euc = weighted_euc_dist(self.perturbations[-2], perturbation)
            w_rel_dtw = fastdtw(self.perturbations[-2], perturbation, dist=weighted_euc_dist)[0]
        else:
            w_rel_euc = 0
            w_rel_dtw = 0
        
        row.extend([abs_euc, abs_dtw, rel_euc, rel_dtw, w_rel_euc, w_rel_dtw, w_abs_euc, w_abs_dtw])
        row.extend([kwargs[k] for k in kwargs.keys()])
        
        self.rows.append(row)
    
    def update_perturbation(
        self,
        perturbations: List[np.ndarray],
        confidences: List[float]
    ):
        """
        Update multiple perturbations at once
        
        Args:
            perturbations: List of perturbed time series
            confidences: Corresponding confidence scores
        """
        for e, (perturbation, c) in enumerate(zip(perturbations, confidences)):
            perturbation_flat = perturbation.flatten()
            
            # Compute absolute distances
            abs_euc = euclidean(self.original_signal, perturbation_flat)
            abs_dtw = fastdtw(self.original_signal, perturbation_flat, dist=euclidean)[0]
            
            # Compute relative distances
            if len(self.perturbations) >= 1:
                rel_euc = euclidean(self.perturbations[-1], perturbation_flat)
                rel_dtw = fastdtw(self.perturbations[-1], perturbation_flat, dist=euclidean)[0]
            else:
                rel_euc = 0
                rel_dtw = 0
            
            self.perturbations.append(perturbation_flat)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics of perturbation process
        
        Returns:
            Dictionary with summary metrics
        """
        if len(self.rows) < 2:
            return {}
        
        # Extract distances from rows
        distances = {
            'abs_euc': [row[len(self.original_signal)*2 + 6] for row in self.rows[1:]],
            'abs_dtw': [row[len(self.original_signal)*2 + 7] for row in self.rows[1:]],
        }
        
        return {
            'num_iterations': len(self.perturbations) - 1,
            'final_abs_euc': distances['abs_euc'][-1] if distances['abs_euc'] else 0,
            'final_abs_dtw': distances['abs_dtw'][-1] if distances['abs_dtw'] else 0,
            'mean_abs_euc': np.mean(distances['abs_euc']) if distances['abs_euc'] else 0,
            'mean_abs_dtw': np.mean(distances['abs_dtw']) if distances['abs_dtw'] else 0,
        }
