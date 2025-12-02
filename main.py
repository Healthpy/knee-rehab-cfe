import torch
import torch.nn as nn
import numpy as np
import os
import json
import csv
import tempfile
import matplotlib.pyplot as plt
from datetime import datetime
import time
import logging

# Project imports
from src.explainers import AdaptiveIMUCFExplainer, AdaptiveMultiObjectiveExplainer
from src.core.utils import CustomJsonEncoder, setup_logging
from src.core.config import get_injury_side, SensorSpecifications
from src.experiments.default_args import parse_arguments
from src.experiments.utils import set_global_seed
from src.data.data_loader import load_movement_data
from src.explainers.mcels_engine import MCELSExplainer
from src.models.fcn_pytorch_model import FCN
from src.evaluation.metrics import EvaluationMetrics, evaluate_single_result

# Constants
TARGET_CLASS = SensorSpecifications.TARGET_CLASS


def extract_target_probability(prob_array, target_class):
    """Extract target class probability from prediction array."""
    if hasattr(prob_array, 'shape') and len(prob_array.shape) > 1:
        return float(prob_array[0, target_class])
    elif hasattr(prob_array, '__len__') and len(prob_array) > target_class:
        return float(prob_array[target_class])
    else:
        return float(prob_array)


def get_model_prediction(model, signal_tensor):
    """Get model prediction and probabilities for a signal."""
    with torch.no_grad():
        # Handle both flattened and unflattened inputs
        if len(signal_tensor.shape) == 1:
            # If flattened, reshape to (channels, timesteps)
            # Assuming 48 channels and timesteps = total_length / 48
            total_length = signal_tensor.shape[0]
            if total_length % 48 == 0:
                timesteps = total_length // 48
                signal_tensor = signal_tensor.reshape(48, timesteps)
            else:
                raise ValueError(f"Cannot reshape flattened tensor of length {total_length} to (48, timesteps)")
        
        # Add batch dimension: (channels, timesteps) -> (1, channels, timesteps)
        if len(signal_tensor.shape) == 2:
            signal_tensor = signal_tensor.unsqueeze(0)
        elif len(signal_tensor.shape) == 3 and signal_tensor.shape[0] != 1:
            # If already has batch dimension but wrong size, fix it
            signal_tensor = signal_tensor.unsqueeze(0)
        
        softmax_fn = nn.Softmax(dim=-1)
        prediction = softmax_fn(model(signal_tensor))
        label = np.argmax(prediction.cpu().numpy())
        return prediction, label


class ModelWrapper:
    """Wrapper to make PyTorch model compatible with evaluation metrics."""
    
    def __init__(self, pytorch_model, device):
        self.model = pytorch_model
        self.device = device
        
    def predict(self, X):
        """Predict class labels for input data."""
        if isinstance(X, list):
            X = np.array(X)
        
        # Handle single sample vs batch
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Convert to tensor and reshape for FCN
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        # Reshape to match FCN input expectations (batch_size, channels, sequence_length)
        if len(X_tensor.shape) == 2:
            # Assume flattened input, reshape to (batch, channels, time_steps)
            batch_size = X_tensor.shape[0]
            total_features = X_tensor.shape[1]
            # Assuming 48 channels and sequence length = total_features / 48
            channels = SensorSpecifications.TOTAL_CHANNELS
            seq_len = total_features // channels
            X_tensor = X_tensor.reshape(batch_size, channels, seq_len)
        
        with torch.no_grad():
            predictions = self.model.predict(X_tensor)
            return predictions.cpu().numpy()


def setup_environment(args):
    """Setup environment, load data and model."""
    logger = logging.getLogger(__name__)
    logger.info("Configuration loaded")
    
    if args.enable_seed:
        set_global_seed(args.seed_value)
    
    # Load data
    logger.info(f"Loading {args.movement_type} movement data...")
    movement_data = load_movement_data(args.movement_type, data_dir="src/data/norm_movement_data")

    # Load model
    model_path = os.path.join('models', f'{args.movement_type}_best_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FCN(SensorSpecifications.TOTAL_CHANNELS, len(np.unique(movement_data['y_train'])))
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()
    
    logger.info(f"Model loaded from {model_path} on device: {device}")
    
    # Create model wrapper for evaluation compatibility
    model_wrapper = ModelWrapper(model, device)
    
    # Initialize evaluation metrics
    evaluator = EvaluationMetrics(model_wrapper, movement_data['X_train'])
    
    return movement_data, model, device, model_wrapper, evaluator


def initialize_explainer(args, model, bg_data, bg_labels):
    """Initialize the appropriate explainer based on algorithm."""
    logger = logging.getLogger(__name__)
    
    explainer_map = {
        'cf_adaptive': AdaptiveIMUCFExplainer,
        'mcels': MCELSExplainer,
        'adaptive_multi': AdaptiveMultiObjectiveExplainer,
    }
    
    if args.algo not in explainer_map:
        raise ValueError(f"Unknown algorithm: {args.algo}")
    
    logger.info(f"Initializing {explainer_map[args.algo].__name__}...")
    

    if args.algo == 'adaptive_multi':
        return AdaptiveMultiObjectiveExplainer(
            model=model,
            background_data=bg_data,
            device='auto',
            args=args
        )
    else:
        common_params = {
            'background_data': bg_data,
            'background_label': bg_labels,
            'predict_fn': model,
            'enable_wandb': False,
            'use_cuda': torch.cuda.is_available()
        }
        
        # Add args parameter for all explainers
        common_params['args'] = args
        
        return explainer_map[args.algo](**common_params)


def get_sample_indices(args, test_data_filtered):
    """Determine which samples to process based on run mode."""
    if args.run_mode == 'single':
        if args.single_sample_id >= len(test_data_filtered):
            raise ValueError(f"Sample ID {args.single_sample_id} exceeds available samples ({len(test_data_filtered)})")
        return [args.single_sample_id]
    else:
        return list(range(min(args.num_samples, len(test_data_filtered))))


def process_sample(args, explainer, model, sample_data, save_dir, evaluator=None):
    """Process a single sample and generate counterfactual explanation."""
    
    original_signal, original_label, subject_id, emg_data = sample_data
    
    # Convert to tensor and get prediction
    original_tensor = torch.tensor(original_signal, dtype=torch.float32)
    ori_pred, category = get_model_prediction(model, original_tensor)
    
    injury_side = get_injury_side(subject_id)
    
    # Set background data if needed
    if args.background_data == "none":
        explainer.background_data = original_tensor
        explainer.background_label = original_label
    
    start_time = time.time()
    # Generate explanation
    result = explainer.generate_saliency(
        data=original_signal,
        label=original_label,
        save_dir=save_dir,
        use_sensor_selection=args.sensor_selection,
        max_retry_attempts=5,
        target_class=TARGET_CLASS,
        importance_method=args.importance_method,
        subject_id=subject_id,
        injury_side=injury_side,
        movement_type=args.movement_type
    )
    
    # Unpack result based on algorithm
    explainer_metrics = {}
    if args.algo == 'mcels':
        if len(result) == 4:
            mask, cf_input, target_prob, mcels_metrics = result
            # MCELS uses all channels (no channel selection)
            selected_channels = list(range(SensorSpecifications.TOTAL_CHANNELS))
            if isinstance(mcels_metrics, dict):
                explainer_metrics = mcels_metrics
        elif len(result) == 5:
            mask, cf_input, target_prob, selected_channels, mcels_metrics = result
            if isinstance(mcels_metrics, dict):
                explainer_metrics = mcels_metrics
        else:
            raise ValueError(f"Unexpected result format from MCELS explainer: {len(result)} elements")
    else:
        if len(result) >= 5:
            mask, cf_input, target_prob, selected_channels, adaptive_metrics = result
            if isinstance(adaptive_metrics, dict):
                explainer_metrics = adaptive_metrics
        
        # Generate summary for counterfactual explainers
        explainer.generate_cf_explanation_summary(
            data=original_signal,
            original_label=original_label,
            target_class=TARGET_CLASS,
            subject_id=subject_id,
            injured_foot=injury_side,
            use_sensor_selection=args.sensor_selection,
            selected_channels=selected_channels,
            selected_sensors=None,
            counterfactual_data=cf_input,
            original_prob=extract_target_probability(ori_pred.cpu().numpy(), TARGET_CLASS),
            cf_prob=float(target_prob) if not isinstance(target_prob, (int, float)) else target_prob,
            importance_method=args.importance_method,
            save_csv=True,
            save_dir=save_dir
        )

    if cf_input is None:
        return None
    generation_time = time.time() - start_time
    # Get counterfactual prediction
    cf_tensor = torch.tensor(cf_input, dtype=torch.float32)
    cf_pred, cf_label = get_model_prediction(model, cf_tensor)
    print(f"Counterfactual prediction: {cf_label}, Probability: {cf_pred[0][cf_label].item():.3f}")
        
    # Prepare base result dictionary
    result_dict = {
        'original_signal': original_tensor.flatten(),
        'original_prob': ori_pred.cpu().numpy(),
        'original_label': original_label,  # Use ground truth label, not model prediction
        'cf_signal': cf_input.flatten(),
        'cf_prob': target_prob,
        'cf_label': cf_label,
        'mask': mask,
        'subject_id': subject_id,
        'injury_side': injury_side,
        'emg_data': emg_data,
        'selected_channels': selected_channels,
        'generation_time': generation_time,
        'success': True,
        'explainer_metrics': explainer_metrics  # Add explainer-specific metrics
    }
    
    # Add evaluation metrics if evaluator is provided
    if evaluator is not None:
        try:
            eval_metrics = evaluator.evaluate_counterfactual(
                original=original_tensor.flatten().cpu().numpy(),
                counterfactual=cf_input.flatten(),
                target_class=TARGET_CLASS,
                mask=mask
            )
            
            # Add convergence information from explainer if available
            if 'iterations' in explainer_metrics:
                eval_metrics['convergence_iterations'] = explainer_metrics['iterations']
            elif 'total_iterations' in explainer_metrics:
                eval_metrics['convergence_iterations'] = explainer_metrics['total_iterations']
            else:
                eval_metrics['convergence_iterations'] = -1  # Unknown
            
            result_dict['evaluation_metrics'] = eval_metrics
            
            # Save individual evaluation metrics
            with open(os.path.join(save_dir, 'evaluation_metrics.json'), 'w') as f:
                json.dump(eval_metrics, f, indent=2, cls=CustomJsonEncoder)
            
            print(f"Evaluation - Validity: {eval_metrics.get('validity', 0):.3f}, "
                  f"Sparsity: {eval_metrics.get('sparsity', 0):.3f}, "
                  f"L2 Distance: {eval_metrics.get('l2_distance', 0):.3f}")
                  
        except Exception as e:
            print(f"Warning: Could not compute evaluation metrics: {e}")
            result_dict['evaluation_metrics'] = {}
    
    return result_dict


def save_results(results, save_path):
    """Save all results to numpy files."""
    if not results:
        return
    
    # Save numpy arrays
    data_arrays = {
        'saliencycf.npy': [r['cf_signal'] for r in results],
        'saliency_cf_prob.npy': [r['cf_prob'] for r in results],
        'map_cf.npy': [r['mask'] for r in results],
        'original.npy': [r['original_signal'] for r in results],
        'ori_prob.npy': [r['original_prob'] for r in results],
        'cf_labels.npy': [r['cf_label'] for r in results],
        'categorys.npy': [r['original_label'] for r in results],
        'subjects_tested.npy': [r['subject_id'] for r in results],
        'emg2test.npy': [r['emg_data'] for r in results],
        'generation_times.npy': [r.get('generation_time', 0) for r in results]
    }
    
    for filename, data in data_arrays.items():
        np.save(os.path.join(save_path, filename), np.array(data))
    
    # Save evaluation metrics if available
    evaluation_metrics = [r.get('evaluation_metrics', {}) for r in results]
    if any(evaluation_metrics):
        with open(os.path.join(save_path, 'evaluation_metrics.json'), 'w') as f:
            json.dump(evaluation_metrics, f, indent=2, cls=CustomJsonEncoder)


def create_visualizations(results, save_path, movement_type, importance_method, max_plots=10):
    """Create visualization plots for results."""
    num_plots = min(max_plots, len(results))
    
    for i in range(num_plots):
        result = results[i]
        
        plt.figure(figsize=(16, 10))
        plt.suptitle(f'Counterfactual Analysis - Subject {result["subject_id"]} '
                    f'({result["injury_side"]} Foot Injury) | Movement: {movement_type.capitalize()}',
                    fontsize=16, fontweight='bold')
        
        # Original signal
        plt.subplot(3, 1, 1)
        plt.plot(result['original_signal'], label='Original', linewidth=1.5, color='blue')
        plt.title(f'Original Signal - Label: {result["original_label"]}')
        plt.ylabel('Signal Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Counterfactual signal
        plt.subplot(3, 1, 2)
        plt.plot(result['cf_signal'], label='Counterfactual', color='green', linewidth=1.5)
        plt.title(f'Counterfactual Signal - Label: {result["cf_label"]}')
        plt.ylabel('Signal Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Saliency mask
        plt.subplot(3, 1, 3)
        plt.plot(np.array(result['mask']).flatten(), label='Saliency Mask', color='orange', linewidth=1.5)
        plt.title('Saliency Mask')
        plt.xlabel('Time Steps')
        plt.ylabel('Mask Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 
                   f'{movement_type}_{importance_method}_subject_{result["subject_id"]}_'
                   f'{result["injury_side"].lower()}_foot.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()


def create_consolidated_csv(results, save_path, movement_type, importance_method, algorithm_name, group_level=None, args=None):
    """Create consolidated CSV summary of all results."""
    if not results:
        return []
    
    # Create simple, consistent filename
    # Format: {algo}_{group_level}_{movement_type}.csv for Adaptive-MO
    # Format: {algo}_{movement_type}_{timestamp}.csv for others
    
    if algorithm_name == 'adaptive_multi' and group_level:
        filename = f"{algorithm_name}_{group_level}_{movement_type}.csv"
    elif algorithm_name == 'mcels':
        filename = f"{algorithm_name}_{movement_type}.csv"
    else:
        # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{algorithm_name}_{movement_type}.csv"
    
    filepath = os.path.join(save_path, filename)
    
    consolidated_data = []
    for i, result in enumerate(results):
        # Extract probabilities using helper function
        original_target_prob = extract_target_probability(result['original_prob'], TARGET_CLASS)
        cf_target_prob = float(result['cf_prob']) if not isinstance(result['cf_prob'], (int, float)) else result['cf_prob']
        
        # Base row data
        row_data = {
            # Subject and experimental information
            'Subject_ID': result['subject_id'],
            'Injured_Foot': result['injury_side'],
            'Movement_Type': movement_type.capitalize(),
            'Processing_Index': i,
            'Timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            
            # Classification results
            'Original_Label': result['original_label'],
            'CF_Target_Label': TARGET_CLASS,
            'CF_Predicted_Label': result['cf_label'],
            'Original_Target_Probability': f"{original_target_prob:.4f}",
            'CF_Target_Probability': f"{cf_target_prob:.4f}",
            'Probability_Change': f"{cf_target_prob - original_target_prob:.4f}",
            'CF_Success': result['cf_label'] == TARGET_CLASS,
            'Target_Achievement': result['cf_label'] == TARGET_CLASS,
            'Probability_Improvement': cf_target_prob > original_target_prob,
            
            # Channel and group information
            'Channels_Selected': str(result['selected_channels']),
            'Num_Channels_Selected': len(result.get('selected_channels', [])),
            'Channel_Selection_Ratio': f"{len(result.get('selected_channels', [])) / SensorSpecifications.TOTAL_CHANNELS:.4f}",
            
            # Performance metrics
            'Generation_Time': f"{result.get('generation_time', 0):.3f}",
            'Success_Rate': 1.0 if result['cf_label'] == TARGET_CLASS else 0.0,
        }
        
        # Add evaluation metrics if available
        eval_metrics = result.get('evaluation_metrics', {})
        if eval_metrics:
            # Core quality metrics
            metric_fields = {
                'Validity': 'validity',
                'Sparsity': 'sparsity',
                'Channel_Sparsity': 'channel_sparsity',
                'Modified_Features': 'modified_features',
                
                # Distance metrics
                'L1_Distance': 'l1_distance',
                'L2_Distance': 'l2_distance',
                'Mean_Channel_Change': 'mean_channel_change',
                'Max_Channel_Change': 'max_channel_change',
                
                # Plausibility metrics
                'Plausibility_LOF': 'plausibility_score',
                
                # Robustness metrics
                'Noise_Stability_Overall': 'noise_stability_overall',
                'Noise_Stability_0.1': 'noise_stability_0.1',
                'Noise_Stability_0.55': 'noise_stability_0.55',
                'Noise_Stability_1.0': 'noise_stability_1.0',
                
                # Temporal metrics
                'Temporal_Smoothness': 'temporal_smoothness',
                
                # Convergence metrics
                'Convergence_Iterations': 'convergence_iterations'
            }
            
            for csv_field, metric_key in metric_fields.items():
                value = eval_metrics.get(metric_key, 0)
                if csv_field in ['Modified_Features', 'Convergence_Iterations']:
                    row_data[csv_field] = int(value) if value >= 0 else 0
                else:
                    row_data[csv_field] = f"{value:.4f}" if value >= 0 else "N/A"
        else:
            # If no evaluation metrics available, set default values
            default_metrics = {
                'Validity': "N/A",
                'Sparsity': "N/A",
                'Channel_Sparsity': "N/A",
                'Modified_Features': 0,
                'L1_Distance': "N/A",
                'L2_Distance': "N/A",
                'Mean_Channel_Change': "N/A",
                'Max_Channel_Change': "N/A",
                'Plausibility_LOF': "N/A",
                'Noise_Stability_Overall': "N/A",
                'Noise_Stability_0.1': "N/A",
                'Noise_Stability_0.55': "N/A",
                'Noise_Stability_1.0': "N/A",
                'Noise_Stability_Low': "N/A", 
                'Noise_Stability_Medium': "N/A",
                'Noise_Stability_High': "N/A",
                'Temporal_Smoothness': "N/A",
                'Convergence_Iterations': 0
            }
            row_data.update(default_metrics)

        if algorithm_name == 'mcels':
            # For MCELS, indicate that all channels are used (no grouping)
            all_channels = list(range(SensorSpecifications.TOTAL_CHANNELS))
            
            # Extract MCELS-specific metrics from evaluation_metrics if available
            mcels_metrics = result.get('evaluation_metrics', {})
            
            row_data.update({
                'Algorithm': 'M-CELS',
                'Group_Level': 'all_channels',
                'Selected_Groups': 'ALL_CHANNELS',
                'Num_Groups_Selected': 1,  # One group containing all channels
                'Group_Selection_Method': 'None (All Channels)',
                'Dynamic_Group_Removal': 'Disabled',
                'Adaptive_Weighting': 'Disabled',
                'Final_Group_Channels': str(all_channels),
                'Channel_Level_Optimization': 'True',
                'Guide_Retrieval': 'KNN-based',
                'Mask_Optimization': 'Gradient-based',
                'Total_Channels_Used': len(all_channels)
            })
        
        elif algorithm_name == 'adaptive_multi':
            # For Adaptive Multi-Objective Explainer, add relevant fields
            group_level_to_use = getattr(args, 'group_level', 'sensor') if args else (group_level or 'sensor')
            
            # Extract additional metrics from result if available
            additional_metrics = result.get('evaluation_metrics', {})
            
            row_data.update({
                'Algorithm': 'Adaptive-Multi-Objective',
                'Group_Level': group_level_to_use,
                'Selected_Groups': str(additional_metrics.get('selected_groups', 'N/A')),
                'Num_Groups_Selected': len(result.get('selected_channels', [])),
                'Group_Selection_Method': 'Shapley-Based',
                'Dynamic_Group_Removal': 'Enabled',
                'Adaptive_Weighting': 'Enabled',
                'Final_Group_Channels': str(result.get('selected_channels', [])),
                'Channel_Level_Optimization': 'True',
                'Guide_Retrieval': 'Shapley-based',
                'Mask_Optimization': 'Multi-objective',
                'Total_Channels_Used': len(result.get('selected_channels', []))
            })
            
            # Add additional group-specific information if available in metrics
            if 'removed_groups' in additional_metrics:
                row_data['Removed_Groups'] = str(additional_metrics['removed_groups'])
            if 'group_importance_scores' in additional_metrics:
                row_data['Group_Importance_Scores'] = str(additional_metrics['group_importance_scores'])
        
        else:
            # For any other algorithms, add default algorithm-specific columns
            row_data.update({
                'Algorithm': algorithm_name,
                'Group_Level': 'unknown',
                'Selected_Groups': 'N/A',
                'Num_Groups_Selected': len(result.get('selected_channels', [])),
                'Group_Selection_Method': 'Unknown',
                'Dynamic_Group_Removal': 'Unknown',
                'Adaptive_Weighting': 'Unknown',
                'Final_Group_Channels': str(result.get('selected_channels', [])),
                'Channel_Level_Optimization': 'Unknown',
                'Guide_Retrieval': 'Unknown',
                'Mask_Optimization': 'Unknown',
                'Total_Channels_Used': len(result.get('selected_channels', []))
            })

        consolidated_data.append(row_data)
    
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            if consolidated_data:
                fieldnames = consolidated_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(consolidated_data)
        # print(f"Consolidated summary saved to: {filepath}")
    except Exception as e:
        print(f"Error saving consolidated CSV: {e}")
    
    return consolidated_data


def print_summary_statistics(consolidated_data, movement_type, samples):
    """Print summary statistics of the analysis."""
    if not consolidated_data:
        print("No data to summarize.")
        return
    
    successful_cfs = sum(1 for data in consolidated_data if data['CF_Success'])
    total_processed = len(samples)
    left_injury_count = sum(1 for data in consolidated_data if data['Injured_Foot'] == 'Left')
    right_injury_count = sum(1 for data in consolidated_data if data['Injured_Foot'] == 'Right')
    
    print(f"\n=== Counterfactual Generation Summary ===")
    print(f"Movement Type: {movement_type.capitalize()}")
    print(f"Total Processed: {total_processed}")
    print(f"Successful Counterfactuals: {successful_cfs}/{total_processed} ({100 * successful_cfs / total_processed:.1f}%)")
    print(f"Left Foot Injuries: {left_injury_count}")
    print(f"Right Foot Injuries: {right_injury_count}")
    
    # Extract and print metrics
    metrics_to_extract = [
        ('Probability_Change', 'Average Probability Change', 4),
        ('Validity', 'Average Validity', 4),
        ('Sparsity', 'Average Sparsity', 4),
        ('L2_Distance', 'Average L2 Distance', 4),
        ('Noise_Stability', 'Average Noise Stability', 4),
        ('Generation_Time', 'Average Generation Time', 3)
    ]
    
    print(f"\n=== Performance Metrics ===")
    for metric_key, label, precision in metrics_to_extract:
        if metric_key == 'Noise_Stability':
            # Handle noise stability specially - exclude disabled cases (-1.0)
            values = [float(data[metric_key]) for data in consolidated_data 
                     if metric_key in data and float(data[metric_key]) >= 0.0]
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                num_valid = len(values)
                total_samples = len(consolidated_data)
                print(f"{label}: {mean_val:.{precision}f} ± {std_val:.{precision}f} "
                      f"({num_valid}/{total_samples} samples with stability enabled)")
            else:
                print(f"{label}: N/A (noise stability disabled or no valid measurements)")
        else:
            values = [float(data[metric_key]) for data in consolidated_data 
                  if metric_key in data and data[metric_key] != 'N/A' and data[metric_key] != '']
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                if metric_key == 'Generation_Time':
                    print(f"{label}: {mean_val:.{precision}f}s ± {std_val:.{precision}f}s")
                else:
                    print(f"{label}: {mean_val:.{precision}f} ± {std_val:.{precision}f}")


def main():
    """Main execution function."""
    # Setup arguments
    args = parse_arguments()
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join('logs', f'{args.algo}_{args.movement_type}')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{args.algo}_{timestamp}.log')
    
    logger = setup_logging(
        level=getattr(args, 'log_level', 'INFO'),
        log_file=log_file
    )
    
    logger.info("Starting XAI analysis")
    logger.info(f"Configuration: {json.dumps(args.__dict__, indent=2)}")
    
    # Log TARGET_CLASS information for verification
    logger.info(f"TARGET_CLASS constant: {TARGET_CLASS} (should be 0 for 'Healthy' class)")
    print(f"Target class for counterfactual generation: {TARGET_CLASS}")
    print(f"Note: We will process only samples that do NOT belong to class {TARGET_CLASS}")
    
    # Setup environment
    movement_data, model, device, model_wrapper, evaluator = setup_environment(args)
    
    # Extract data
    X_test, y_test = movement_data['X_test'], movement_data['y_test']
    subjects_test, X_test_emg = movement_data['subjects_test'], movement_data['X_test_emg']
    
    logger.info(f"Movement: {args.movement_type}, Classes: {len(np.unique(movement_data['y_train']))}, "
               f"Train samples: {len(movement_data['X_train'])}, Test samples: {len(X_test)}")
    
    # Initialize explainer and setup paths
    explainer = initialize_explainer(args, model, movement_data['X_train'], movement_data['y_train'])
    
    # Create simple, consistent directory structure
    # Format: results/{algo}_{movement_type}/

    if args.algo == 'adaptive_multi' and hasattr(args, 'group_level'):
        tag = f'{args.algo}_{args.group_level}_{args.movement_type}'
    else:
        tag = f'{args.algo}_{args.movement_type}'
    
    base_save_dir = os.path.join('results', tag)
    
    os.makedirs(base_save_dir, exist_ok=True)
    
    logger.info(f"Results will be saved to: {base_save_dir}")
    
    # Filter samples to ensure NO test samples have target class label (0)
    logger.info(f"Target class for counterfactual generation: {TARGET_CLASS}")
    logger.info(f"Original test samples: {len(y_test)}")
    logger.info(f"Test samples with target class {TARGET_CLASS}: {np.sum(y_test == TARGET_CLASS)}")
    
    # Create mask to exclude any samples that already have the target class
    non_target_mask = y_test != TARGET_CLASS
    filtered_data = (X_test[non_target_mask], y_test[non_target_mask], 
                    subjects_test[non_target_mask], X_test_emg[non_target_mask])
    
    # Verify filtering worked correctly
    assert len(filtered_data[1]) == np.sum(non_target_mask), "Filtering failed: sample count mismatch"
    assert np.sum(filtered_data[1] == TARGET_CLASS) == 0, f"CRITICAL ERROR: Filtered data still contains target class {TARGET_CLASS} samples!"
    
    logger.info(f"Filtered test samples (excluding target class): {len(filtered_data[1])}")
    logger.info(f"Available class labels in filtered data: {np.unique(filtered_data[1])}")
    
    # Ensure we have samples to process
    if len(filtered_data[1]) == 0:
        logger.error("No samples available for counterfactual generation after filtering!")
        print("ERROR: All test samples already belong to target class. No counterfactuals to generate.")
        return
    
    sample_indices = get_sample_indices(args, filtered_data[0])
    
    # Process samples
    results = []
    stats = {'attempted': 0, 'successful': 0, 'failed': 0}
    
    print(f"Processing {len(sample_indices)} samples (all guaranteed to be non-target class {TARGET_CLASS})...")
    
    for loop_idx, actual_idx in enumerate(sample_indices):
        stats['attempted'] += 1
        sample_data = (filtered_data[0][actual_idx], filtered_data[1][actual_idx], 
                      filtered_data[2][actual_idx], filtered_data[3][actual_idx])
        
        # Double-check that sample is not target class (should never happen due to filtering)
        assert sample_data[1] != TARGET_CLASS, f"INTERNAL ERROR: Sample {actual_idx} has target class {TARGET_CLASS} despite filtering!"
        
        print(f"Processing sample {loop_idx+1}/{len(sample_indices)} (index {actual_idx}) - "
              f"Original class: {sample_data[1]}, Target: {TARGET_CLASS}")
        
        # Create simple experiment directory
        experiment_name = f'sample_{actual_idx}'
        save_dir = os.path.join(base_save_dir, experiment_name)
        os.makedirs(save_dir, exist_ok=True)
        
        config = {**args.__dict__, 'save_dir': save_dir, 'tag': tag}
        with open(os.path.join(save_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2, cls=CustomJsonEncoder)
        
        # Process sample
        result = process_sample(args, explainer, model, sample_data, save_dir, evaluator)
        
        if result is None:
            stats['failed'] += 1
            print(f"Failed to generate counterfactual for sample {actual_idx}")
            continue
        
        stats['successful'] += 1
        results.append(result)
        print(f"Generated counterfactual for Subject {result['subject_id']} "
                f"({result['injury_side']} injury): probability {result['cf_prob']:.3f}")
        
    
    
    # Print processing statistics
    print(f"\n=== Processing Statistics ===")
    print(f"Total samples attempted: {stats['attempted']}")
    print(f"Successfully processed: {stats['successful']}")
    print(f"Processing failures: {stats['failed']}")
    print(f"Success rate: {100 * stats['successful'] / stats['attempted']:.1f}%" if stats['attempted'] > 0 else "N/A")
    
    # Verify no target class samples were processed
    if results:
        processed_labels = [r.get('original_label', -1) for r in results]
        target_class_count = sum(1 for label in processed_labels if label == TARGET_CLASS)
        print(f"Verification: {target_class_count} samples with target class {TARGET_CLASS} were processed (should be 0)")
        assert target_class_count == 0, f"VALIDATION ERROR: {target_class_count} target class samples were processed!"
    
    if not results:
        print("No results to save!")
        return
    
    # Save results and create outputs
    if results:
        # Save explainer-specific results
        explainer_save_path = os.path.join(base_save_dir, 'explainer_results.json')
        explainer.save_results(explainer_save_path)
    
    consolidated_data = create_consolidated_csv(results, base_save_dir, args.movement_type, args.importance_method, args.algo, getattr(args, 'group_level', None), args)
    
    print_summary_statistics(consolidated_data, args.movement_type, sample_indices)
    
    # Final verification: Ensure no target class samples were processed
    logger.info("="*50)
    logger.info("FINAL VERIFICATION")
    logger.info(f"Target class (should be excluded): {TARGET_CLASS}")
    logger.info(f"Total samples processed: {len(results)}")
    
    if results:
        original_labels = [r.get('original_label', -1) for r in results]
        unique_labels = np.unique(original_labels)
        target_class_processed = TARGET_CLASS in unique_labels
        
        logger.info(f"Original labels processed: {unique_labels}")
        logger.info(f"Target class {TARGET_CLASS} was processed: {target_class_processed}")
        
        if target_class_processed:
            logger.error(f"VALIDATION FAILED: Target class {TARGET_CLASS} samples were processed!")
            print(f"ERROR: Target class {TARGET_CLASS} samples were processed despite filtering!")
        else:
            logger.info("SUCCESS: No target class samples were processed")
            print(f"VERIFIED: No samples with target class {TARGET_CLASS} were processed")
    
    logger.info("="*50)
    logger.info(f"Results saved to: {base_save_dir}")
    logger.info("Analysis complete!")


if __name__ == '__main__':
    main()
