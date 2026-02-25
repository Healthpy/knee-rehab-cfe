"""
Utility Functions for Explainability

Helper functions for normalization, distance metrics, and data processing.
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from collections import defaultdict
from typing import Optional, Tuple, Dict


def rounder(arr: np.ndarray, decimals: int = 2) -> list:
    """
    Round array values to specified decimals
    
    Args:
        arr: Input array
        decimals: Number of decimal places
        
    Returns:
        List of rounded values
    """
    return [round(float(a), decimals) for a in arr]


def normalize(saliency: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Normalize saliency values to [0, 1]
    
    Args:
        saliency: Saliency array
        epsilon: Small constant for numerical stability
        
    Returns:
        Normalized saliency
    """
    abs_saliency = np.abs(saliency)
    max_val = np.max(abs_saliency)
    return (abs_saliency + epsilon) / (max_val + epsilon)


def confidence_score(predictions: np.ndarray, labels: np.ndarray) -> Tuple[Dict, float]:
    """
    Compute confidence scores per class
    
    Args:
        predictions: Model predictions [N, C]
        labels: True labels [N]
        
    Returns:
        scores: Dictionary of per-class confidence scores
        mean_score: Mean confidence across all classes
    """
    correct_indices = defaultdict(lambda: [])
    
    for e, (p, l) in enumerate(zip(predictions, labels)):
        if np.argmax(p) == l:
            correct_indices[l].append(e)
    
    scores = {}
    for k, v in correct_indices.items():
        if len(v) > 0:
            scores[k] = np.max(predictions[v], axis=1).mean()
    
    mean_score = np.mean(list(scores.values())) if scores else 0.0
    return scores, mean_score


def accuracy(true_val: torch.Tensor, pred_val: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute binary accuracy
    
    Args:
        true_val: True labels
        pred_val: Predicted probabilities
        threshold: Decision threshold
        
    Returns:
        Accuracy score
    """
    out = (pred_val > threshold).float()
    return accuracy_score(true_val.cpu().numpy(), out.cpu().numpy())


def accuracy_softmax(true_val: torch.Tensor, pred_val: torch.Tensor) -> float:
    """
    Compute accuracy for multi-class classification
    
    Args:
        true_val: True labels [N]
        pred_val: Predicted probabilities [N, C]
        
    Returns:
        Accuracy score
    """
    pred_labels = torch.argmax(pred_val, dim=1)
    return accuracy_score(true_val.cpu().numpy(), pred_labels.cpu().numpy())


def generate_gaussian_noise(data: np.ndarray, snrdb: float = 20.0) -> np.ndarray:
    """
    Add Gaussian noise to signal with specified SNR
    
    Args:
        data: Input signal
        snrdb: Signal-to-Noise Ratio in dB
        
    Returns:
        Noisy signal
    """
    signal_power = np.mean(data ** 2)
    snr_linear = 10 ** (snrdb / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
    return data + noise


def softmax(X: np.ndarray, theta: float = 1.0, axis: Optional[int] = None) -> np.ndarray:
    """
    Compute softmax function
    
    Args:
        X: Input array
        theta: Temperature parameter
        axis: Axis along which to compute softmax
        
    Returns:
        Softmax probabilities
    """
    y = np.atleast_2d(X)
    
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    
    y = y * float(theta)
    y = y - np.expand_dims(np.max(y, axis=axis), axis)
    y = np.exp(y)
    
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
    p = y / ax_sum
    
    if len(X.shape) == 1:
        p = p.flatten()
    
    return p


def save_timeseries_mul(
    mask: np.ndarray,
    raw_mask: Optional[np.ndarray],
    time_series: np.ndarray,
    perturbated_output: Optional[np.ndarray] = None,
    save_dir: Optional[str] = None,
    enable_wandb: bool = False,
    algo: str = "mcels",
    dataset: str = "",
    category: int = 0
):
    """
    Save time series saliency visualization
    
    Args:
        mask: Normalized saliency mask
        raw_mask: Raw saliency values
        time_series: Original time series
        perturbated_output: Perturbed time series
        save_dir: Directory to save results
        enable_wandb: Whether to log to Weights & Biases
        algo: Algorithm name
        dataset: Dataset name
        category: Class label
    """
    # This is a placeholder - implement visualization saving as needed
    # Can integrate with matplotlib or other visualization libraries
    pass


def find_unique_candidates(data: np.ndarray, labels: np.ndarray) -> Dict[str, list]:
    """
    Find most/least similar pairs between classes
    
    Args:
        data: Input data [N, ...]
        labels: Binary labels [N]
        
    Returns:
        Dictionary with 'max', 'min', 'avg' candidate pairs
    """
    zi = np.argwhere(labels == 0).flatten()
    oi = np.argwhere(labels == 1).flatten()
    
    euclid_dist = {}
    for zc in zi:
        for oc in oi:
            euclid_dist[f"{zc},{oc}"] = np.linalg.norm(data[zc] - data[oc])
    
    t1 = [int(i) for i in max(euclid_dist, key=lambda key: euclid_dist[key]).split(',')]
    t2 = [int(i) for i in min(euclid_dist, key=lambda key: euclid_dist[key]).split(',')]
    
    t3 = [
        np.mean(np.take(data, np.argwhere(labels == 0).flatten(), axis=0), axis=0),
        np.mean(np.take(data, np.argwhere(labels == 1).flatten(), axis=0), axis=0)
    ]
    
    return {"max": t1, "min": t2, "avg": t3}


def dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Dynamic Time Warping distance
    
    Args:
        x: First time series
        y: Second time series
        
    Returns:
        DTW distance
    """
    try:
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean
        distance, path = fastdtw(x, y, dist=euclidean)
        return distance
    except ImportError:
        # Fallback to simple Euclidean if fastdtw not available
        return np.linalg.norm(x - y)
