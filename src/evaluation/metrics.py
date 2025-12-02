"""
Evaluation metrics for counterfactual explanations..
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import accuracy_score
import logging
from sklearn.neighbors import LocalOutlierFactor
from ..core.config import SensorSpecifications
import torch

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Comprehensive evaluation metrics for counterfactual explanations."""
    
    def __init__(
        self, 
        model, 
        X_train: Optional[np.ndarray] = None,
        enable_noise_stability: bool = True,
        noise_levels: List[float] = None,
        noise_trials: int = 10
    ):
        """
        Initialize evaluation metrics.
        
        Args:
            model: Trained model for predictions
            X_train: Training data for nearest neighbor calculations
            enable_noise_stability: Whether to enable noise stability testing
            noise_levels: List of noise standard deviations to test
            noise_trials: Number of trials per noise level
        """
        self.model = model
        self.X_train = X_train
        
        # Noise stability configuration
        self.enable_noise_stability = enable_noise_stability
        self.noise_levels = noise_levels if noise_levels is not None else [0.1, 0.55, 1.0]
        self.noise_trials = noise_trials
        
    def evaluate_counterfactual(
        self,
        original: np.ndarray,
        counterfactual: np.ndarray,
        target_class: int,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single counterfactual explanation.
        
        Args:
            original: Original instance
            counterfactual: Generated counterfactual
            target_class: Target prediction class
            mask: Boolean mask of modified features
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        try:
            # 1. Validity - does counterfactual achieve target class?
            cf_prediction = self.model.predict([counterfactual])
            if hasattr(cf_prediction, '__len__') and len(cf_prediction) > 0:
                cf_pred_class = cf_prediction[0]
            else:
                cf_pred_class = cf_prediction
            metrics['validity'] = float(cf_pred_class == target_class)
            
            # 1b. Target confidence - model confidence on target class  
            try:
                # Get prediction probabilities if available
                if hasattr(self.model, 'predict_proba'):
                    cf_proba = self.model.predict_proba([counterfactual])
                    if len(cf_proba) > 0 and len(cf_proba[0]) > target_class:
                        metrics['target_confidence'] = float(cf_proba[0][target_class])
                    else:
                        metrics['target_confidence'] = 0.0
                elif hasattr(cf_prediction, '__len__') and len(cf_prediction) > target_class:
                    # If prediction returns probabilities directly
                    metrics['target_confidence'] = float(cf_prediction[target_class])
                else:
                    # Binary confidence: 1.0 if correct class, 0.0 otherwise
                    metrics['target_confidence'] = float(cf_pred_class == target_class)
            except:
                metrics['target_confidence'] = float(cf_pred_class == target_class)
            
            # 2. Sparsity - how many features were changed?
            if mask is not None:
                metrics['sparsity'] = 1.0 - (sum(mask) / len(mask))
                metrics['modified_features'] = int(np.sum(mask))
            else:
                # Calculate from difference
                diff_mask = ~np.isclose(original, counterfactual, rtol=1e-5)
                metrics['sparsity'] = 1.0 - (np.sum(diff_mask) / len(diff_mask))
                metrics['modified_features'] = int(np.sum(diff_mask))
            
            # 2b. Channel-level sparsity  
            try:
                channels = SensorSpecifications.TOTAL_CHANNELS
                if original.ndim == 1:
                    seq_len = original.shape[0] // channels
                    if seq_len > 0:
                        # Reshape to (channels, timesteps)
                        orig_reshaped = original.reshape(channels, seq_len)
                        cf_reshaped = counterfactual.reshape(channels, seq_len)
                        
                        # Check which channels have any changes
                        channel_changed = np.any(~np.isclose(orig_reshaped, cf_reshaped, rtol=1e-5), axis=1)
                        metrics['channel_sparsity'] = 1.0 - (np.sum(channel_changed) / channels)
                        
                        # Temporal smoothness - measure changes between consecutive timesteps
                        diff_over_time = np.abs(cf_reshaped[:, 1:] - cf_reshaped[:, :-1])
                        metrics['temporal_smoothness'] = float(np.mean(diff_over_time))
                    else:
                        metrics['channel_sparsity'] = metrics['sparsity']
                        metrics['temporal_smoothness'] = 0.0
                else:
                    metrics['channel_sparsity'] = metrics['sparsity'] 
                    metrics['temporal_smoothness'] = 0.0
            except:
                metrics['channel_sparsity'] = metrics['sparsity']
                metrics['temporal_smoothness'] = 0.0
            
            # 3. Distance - how far is the counterfactual?
            metrics['l1_distance'] = float(np.sum(np.abs(original - counterfactual)))
            metrics['l2_distance'] = float(np.linalg.norm(original - counterfactual))

            # 4. Channel-wise changes (mean and max)
            try:
                channels = SensorSpecifications.TOTAL_CHANNELS
                if original.ndim == 1:
                    seq_len = original.shape[0] // channels
                    if seq_len > 0:
                        reshaped_original = original.reshape(channels, seq_len)
                        reshaped_counterfactual = counterfactual.reshape(channels, seq_len)
                        channel_diffs = np.abs(reshaped_original - reshaped_counterfactual)
                        metrics['mean_channel_change'] = float(np.mean(channel_diffs))
                        metrics['max_channel_change'] = float(np.max(channel_diffs))
                    else:
                        metrics['mean_channel_change'] = 0.0
                        metrics['max_channel_change'] = 0.0
                else:
                    metrics['mean_channel_change'] = float(np.mean(np.abs(original - counterfactual)))
                    metrics['max_channel_change'] = float(np.max(np.abs(original - counterfactual)))
            except:
                metrics['mean_channel_change'] = 0.0
                metrics['max_channel_change'] = 0.0

            # 6. Plausibility - distance to nearest training instance
            # Local Outlier Factor (LOF) for plausibility (local density-based outlier detection)
            if self.X_train is not None:
                try:
                    # Ensure both X_train and counterfactual are 2D for LOF
                    X_train_2d = self.X_train.reshape(self.X_train.shape[0], -1) if self.X_train.ndim > 2 else self.X_train
                    counterfactual_2d = counterfactual.reshape(1, -1) if counterfactual.ndim > 1 else counterfactual.reshape(1, -1)
                    
                    # Combine training data and counterfactual for LOF analysis
                    combined_data = np.vstack([X_train_2d, counterfactual_2d])
                    
                    # Initialize LOF with adaptive parameters
                    n_samples = len(combined_data)
                    n_neighbors = min(20, max(5, n_samples // 10))  # Adaptive neighbors based on data size
                    
                    lof = LocalOutlierFactor(
                        n_neighbors=n_neighbors,
                        algorithm='auto',
                        contamination=0.1,
                        novelty=False  # Use fit_predict for outlier detection
                    )
                    
                    # Fit and predict outlier scores
                    outlier_labels = lof.fit_predict(combined_data)
                    lof_scores = lof.negative_outlier_factor_
                    print(f'LOF Scores: {lof_scores}')
                    
                    # Get LOF score for the counterfactual (last element)
                    cf_lof_score = lof_scores[-1]
                    
                    # Transform LOF score to plausibility score [0, 1]
                    # LOF scores are negative, with -1.0 being "normal" and more negative being outliers
                    # Transform so that scores closer to -1.0 get higher plausibility
                    plausibility = max(0.0, min(1.0, 2.0 + cf_lof_score))  # Maps [-inf, -1] to [0, 1]
                    metrics['plausibility_score'] = float(plausibility)
                    
                except Exception as e:
                    logging.warning(f"LOF plausibility computation failed: {e}")
                    metrics['plausibility_score'] = 0.5  # Default neutral score

            # 7. Noise stability - test robustness to noise (only for valid counterfactuals)
            if self.enable_noise_stability and metrics.get('validity', 0.0) > 0.0:
                try:
                    stability_metrics = self.compute_noise_stability(counterfactual, target_class)
                    metrics.update(stability_metrics)
                except Exception as e:
                    logger.warning(f"Failed to compute noise stability: {str(e)}")
                    metrics.update({
                        'noise_stability_overall': -1.0,
                        'noise_stability_error': str(e),
                        'noise_stable_tests': 0,
                        'noise_total_tests': 0
                    })
            else:
                if not self.enable_noise_stability:
                    reason = "disabled"
                elif metrics.get('validity', 0.0) == 0.0:
                    reason = "invalid_counterfactual"
                else:
                    reason = "unknown"
                
                metrics.update({
                    'noise_stability_overall': -1.0,
                    'noise_stability_disabled': reason,
                    'noise_stable_tests': 0,
                    'noise_total_tests': 0
                })

            
        except Exception as e:
            logger.warning(f"Error computing some metrics: {e}")
            # Ensure we have basic metrics even if some fail
            metrics.setdefault('validity', 0.0)
            metrics.setdefault('sparsity', 0.0)
            metrics.setdefault('l2_distance', float('inf'))
            # Ensure noise stability metrics are present
            metrics.setdefault('noise_stability_overall', -1.0)
            metrics.setdefault('noise_stable_tests', 0)
            metrics.setdefault('noise_total_tests', 0)
            
        return metrics
    
    def compute_noise_stability(
        self,
        counterfactual: np.ndarray,
        target_class: int
    ) -> Dict[str, float]:
        """
        Compute noise stability for a counterfactual explanation.
        
        Tests robustness by adding controlled Gaussian noise and checking
        if the model prediction remains consistent with the target class.
        
        Args:
            counterfactual: Counterfactual sample to test
            target_class: Expected target class for stability testing
            
        Returns:
            Dictionary containing stability metrics
        """
        if not self.enable_noise_stability:
            return {
                'noise_stability_overall': -1.0, 
                'noise_stability_disabled': True,
                'noise_stable_tests': 0,
                'noise_total_tests': 0
            }
        
        logger.debug(f"Computing noise stability for counterfactual with target class {target_class}")
        
        # Verify that the counterfactual actually achieves the target class before testing stability
        initial_prediction = self.model.predict([counterfactual])
        if hasattr(initial_prediction, '__len__') and len(initial_prediction) > 0:
            initial_pred_class = initial_prediction[0]
        else:
            initial_pred_class = initial_prediction
            
        if initial_pred_class != target_class:
            logger.warning(f"Counterfactual does not achieve target class {target_class}, got {initial_pred_class}. Skipping stability test.")
            return {
                'noise_stability_overall': -1.0,
                'noise_stability_disabled': "invalid_counterfactual",
                'noise_stable_tests': 0,
                'noise_total_tests': 0
            }
        
        # Ensure counterfactual is numpy array
        if isinstance(counterfactual, torch.Tensor):
            counterfactual = counterfactual.cpu().numpy()
        
        # Compute standard deviation for noise scaling
        cf_std = np.std(counterfactual)
        
        total_stable_tests = 0
        total_tests = 0
        stability_by_level = {}
        
        try:
            # Test each noise level
            for noise_level in self.noise_levels:
                stable_count = 0
                noise_std = noise_level * cf_std
                print(f'Testing noise level {noise_level} with std {noise_std}')
                
                # Run multiple trials for this noise level
                for trial in range(self.noise_trials):
                    try:
                        # Generate Gaussian noise: ε ~ N(0, σ²I)
                        noise = np.random.normal(0, noise_std, counterfactual.shape)
                        cf_noisy = counterfactual + noise
                        
                        # Get model prediction for noisy counterfactual
                        noisy_pred = self.model.predict([cf_noisy])
                        
                        # Handle different prediction formats
                        if hasattr(noisy_pred, '__len__') and len(noisy_pred) > 0:
                            noisy_class = noisy_pred[0]
                        else:
                            noisy_class = noisy_pred
                        
                        # Check if prediction remains stable (same as target)
                        if noisy_class == target_class:
                            stable_count += 1
                            total_stable_tests += 1
                        
                        total_tests += 1
                        
                    except Exception as e:
                        logger.warning(f"Error during noise stability trial {trial} "
                                     f"at noise level {noise_level}: {str(e)}")
                        total_tests += 1  # Count as failed test
                
                # Calculate stability score for this noise level
                stability_score = stable_count / self.noise_trials if self.noise_trials > 0 else 0.0
                stability_by_level[f'noise_level_{noise_level}'] = stability_score
                
                logger.debug(f"Noise level {noise_level}: {stable_count}/{self.noise_trials} "
                           f"stable predictions ({stability_score:.3f})")
        
        except Exception as e:
            logger.warning(f"Error computing noise stability: {str(e)}")
            return {
                'noise_stability_overall': -1.0,
                'noise_stability_error': str(e),
                'noise_stable_tests': 0,
                'noise_total_tests': 0
            }
        
        # Calculate overall stability score
        overall_stability = total_stable_tests / total_tests if total_tests > 0 else 0.0
        
        # Compile results
        stability_results = {
            'noise_stability_overall': overall_stability,
            'noise_stable_tests': total_stable_tests,
            'noise_total_tests': total_tests,
            'cf_std': cf_std,
            **stability_by_level
        }
        
        # Add individual noise level scores for backward compatibility
        for noise_level in self.noise_levels:
            key = f'noise_stability_{noise_level}'
            level_key = f'noise_level_{noise_level}'
            stability_results[key] = stability_results.get(level_key, 0.0)
        
        # Add specific noise level categories for CSV export
        noise_level_mapping = {
            0.1: 'low',
            0.55: 'medium',
            1.0: 'high'
        }
        
        for noise_level, category in noise_level_mapping.items():
            level_key = f'noise_level_{noise_level}'
            category_key = f'noise_stability_{category}'
            if level_key in stability_results:
                stability_results[category_key] = stability_results[level_key]
        
        logger.debug(f"Noise stability computed: {overall_stability:.3f} "
                   f"({total_stable_tests}/{total_tests} stable)")
        
        return stability_results
    
    def evaluate_batch(
        self,
        results: List[Dict],
        include_failed: bool = True
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Evaluate a batch of counterfactual results.
        
        Args:
            results: List of result dictionaries from explainer
            include_failed: Whether to include failed explanations in stats
            
        Returns:
            Dictionary of aggregated evaluation metrics
        """
        if not results:
            return {'error': 'No results to evaluate'}
            
        # Filter successful results
        successful_results = [r for r in results if r.get('success', False)]
        failed_count = len(results) - len(successful_results)
        
        if not successful_results:
            return {
                'success_rate': 0.0,
                'failed_count': failed_count,
                'total_count': len(results),
                'error': 'No successful counterfactuals generated'
            }
        
        # Compute individual metrics
        all_metrics = []
        for result in successful_results:
            if all(key in result for key in ['original', 'counterfactual', 'target_class']):
                metrics = self.evaluate_counterfactual(
                    result['original'],
                    result['counterfactual'],
                    result['target_class'],
                    result.get('mask')
                )
                all_metrics.append(metrics)
        
        if not all_metrics:
            return {
                'success_rate': 0.0,
                'failed_count': failed_count,
                'total_count': len(results),
                'error': 'Could not compute metrics for any results'
            }
        
        # Aggregate metrics
        aggregated = {
            'success_rate': len(successful_results) / len(results),
            'failed_count': failed_count,
            'total_count': len(results),
            'successful_count': len(successful_results)
        }
        
        # Compute statistics for each metric
        metric_names = all_metrics[0].keys()
        for metric_name in metric_names:
            values = [m[metric_name] for m in all_metrics if metric_name in m]
            if values:
                aggregated[f'{metric_name}_mean'] = float(np.mean(values))
                aggregated[f'{metric_name}_std'] = float(np.std(values))
                aggregated[f'{metric_name}_median'] = float(np.median(values))
                aggregated[f'{metric_name}_min'] = float(np.min(values))
                aggregated[f'{metric_name}_max'] = float(np.max(values))
        
        # Add timing information if available
        generation_times = [r.get('generation_time', 0) for r in successful_results]
        if generation_times:
            aggregated['generation_time_mean'] = float(np.mean(generation_times))
            aggregated['generation_time_std'] = float(np.std(generation_times))
            
        return aggregated
    
    def compare_methods(
        self,
        method_results: Dict[str, List[Dict]],
        metrics_to_compare: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple counterfactual generation methods.
        
        Args:
            method_results: Dictionary mapping method names to result lists
            metrics_to_compare: Specific metrics to compare (default: all)
            
        Returns:
            Dictionary of comparison statistics
        """
        if metrics_to_compare is None:
            metrics_to_compare = [
                'success_rate', 'validity_mean', 'sparsity_mean',
                'l2_distance_mean', 'generation_time_mean'
            ]
        
        comparison = {}
        method_evaluations = {}
        
        # Evaluate each method
        for method_name, results in method_results.items():
            evaluation = self.evaluate_batch(results)
            method_evaluations[method_name] = evaluation
            
        # Create comparison table
        for metric in metrics_to_compare:
            comparison[metric] = {}
            values = []
            
            for method_name, evaluation in method_evaluations.items():
                if metric in evaluation and not np.isnan(evaluation[metric]):
                    comparison[metric][method_name] = evaluation[metric]
                    values.append(evaluation[metric])
                else:
                    comparison[metric][method_name] = None
            
            # Add ranking information
            if values:
                # For most metrics, lower is better (except success_rate, validity, sparsity)
                reverse = metric in ['success_rate', 'validity_mean', 'sparsity_mean']
                sorted_methods = sorted(
                    [(name, val) for name, val in comparison[metric].items() if val is not None],
                    key=lambda x: x[1],
                    reverse=reverse
                )
                
                # Add rank information
                comparison[f'{metric}_ranking'] = {
                    name: rank + 1 for rank, (name, _) in enumerate(sorted_methods)
                }
        
        return comparison


# Convenience functions
def evaluate_single_result(
    result: Dict,
    model,
    X_train: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Evaluate a single counterfactual result.
    
    Args:
        result: Result dictionary from explainer
        model: Trained model
        X_train: Training data for plausibility
        
    Returns:
        Evaluation metrics dictionary
    """
    evaluator = EvaluationMetrics(model, X_train)
    
    if not result.get('success', False):
        return {'validity': 0.0, 'error': 'Counterfactual generation failed'}
    
    return evaluator.evaluate_counterfactual(
        result['original'],
        result['counterfactual'],
        result['target_class'],
        result.get('mask')
    )


def evaluate_explanation_quality(
    results: List[Dict],
    model,
    X_train: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Quick evaluation of explanation quality for a batch of results.
    
    Args:
        results: List of result dictionaries
        model: Trained model
        X_train: Training data for plausibility
        
    Returns:
        Aggregated quality metrics
    """
    evaluator = EvaluationMetrics(model, X_train)
    return evaluator.evaluate_batch(results)


def compute_target_confidence(cf_data, model, target_class=1):
    """
    Compute the confidence score for the target class.
    
    Args:
        cf_data: Counterfactual data
        model: Trained model
        target_class: Target class (default 1)
        
    Returns:
        Target confidence score
    """
    try:
        import torch
        
        if torch.is_tensor(cf_data):
            cf_tensor = cf_data
        else:
            cf_tensor = torch.FloatTensor(cf_data)
            
        if cf_tensor.dim() == 1:
            cf_tensor = cf_tensor.unsqueeze(0)
            
        with torch.no_grad():
            model.eval()
            predictions = model(cf_tensor)
            
            if hasattr(predictions, 'softmax') or predictions.dim() > 1:
                probs = torch.softmax(predictions, dim=1)
                target_conf = probs[0, target_class].item()
            else:
                # Binary classification
                target_conf = torch.sigmoid(predictions[0]).item()
                if target_class == 0:
                    target_conf = 1.0 - target_conf
                    
        return target_conf
    except Exception as e:
        print(f"Error computing target confidence: {e}")
        return 0.0


def compute_channel_sparsity(original_data, cf_data, threshold=1e-6):
    """
    Compute channel-level sparsity (proportion of channels that changed).
    
    Args:
        original_data: Original input (2D shape: channels x timesteps)
        cf_data: Counterfactual data (2D shape: channels x timesteps)
        threshold: Minimum change threshold
        
    Returns:
        Channel sparsity ratio (proportion of unchanged channels)
    """
    try:
        if hasattr(original_data, 'detach'):
            orig = original_data.detach().cpu().numpy()
        else:
            orig = np.array(original_data)
            
        if hasattr(cf_data, 'detach'):
            cf = cf_data.detach().cpu().numpy()
        else:
            cf = np.array(cf_data)
        
        # Ensure 2D shape (channels, timesteps)
        if orig.ndim == 1:
            channels = SensorSpecifications.TOTAL_CHANNELS
            seq_len = orig.shape[0] // channels
            orig = orig.reshape(channels, seq_len)
            cf = cf.reshape(channels, seq_len)
        
        # Check if each channel has any changes across timesteps
        channel_changed = np.any(np.abs(orig - cf) > threshold, axis=1)
        changed_channels = np.sum(channel_changed)
        total_channels = orig.shape[0]
        
        # Return sparsity as proportion of unchanged channels
        sparsity = 1.0 - (changed_channels / total_channels) if total_channels > 0 else 0.0
        
        return float(sparsity)
    except Exception as e:
        logger.warning(f"Error computing channel sparsity: {e}")
        return 0.0


def compute_temporal_smoothness(cf_data, window_size=3):
    """
    Compute temporal smoothness of the counterfactual.
    
    Args:
        cf_data: Counterfactual time series data
        window_size: Window size for smoothness computation
        
    Returns:
        Temporal smoothness score (lower = smoother)
    """
    try:
        if hasattr(cf_data, 'detach'):
            data = cf_data.detach().cpu().numpy()
        else:
            data = np.array(cf_data)
            
        # Ensure we have a 2D array (time, features)
        if data.ndim == 1:
            return 0.0  # Single point, perfectly smooth
            
        if data.ndim > 2:
            data = data.reshape(-1, data.shape[-1])
            
        # Compute second derivatives as smoothness measure
        if data.shape[0] < 3:
            return 0.0
            
        # Second derivative approximation
        second_deriv = np.abs(data[2:] - 2*data[1:-1] + data[:-2])
        smoothness = np.mean(second_deriv)
        
        return float(smoothness)
    except Exception as e:
        print(f"Error computing temporal smoothness: {e}")
        return 0.0