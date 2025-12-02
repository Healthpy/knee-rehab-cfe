"""
Default arguments and configuration for experiments.
"""

import argparse
from typing import Any


def parse_arguments() -> Any:
    """
    Parse command line arguments for counterfactual explanation experiments.
    
    Returns:
        Parsed arguments object
    """
    parser = argparse.ArgumentParser(description='Counterfactual Explanation Experiments')
    
    # Basic experiment settings
    parser.add_argument('--algo', type=str, default='mcels',
                       choices=['mcels', 'cf_adaptive', 'adaptive_multi'], 
                       help='Algorithm to use for counterfactual generation')
    parser.add_argument('--movement_type', type=str, default='squat',
                       choices=['squat', 'extension', 'gait'],
                       help='Type of movement data to analyze')
    parser.add_argument('--run_id', type=str, default='1',
                       help='Run identifier for experiment tracking')
    parser.add_argument('--run_mode', type=str, default='batch',
                       choices=['single', 'batch'],
                       help='Run mode: single sample or batch processing')
    parser.add_argument('--single_sample_id', type=int, default=0,
                       help='Sample ID when running in single mode')
    parser.add_argument('--num_samples', type=int, default=150,
                       help='Number of samples to process (ignored in single mode)')
    
    # Model and data settings
    parser.add_argument('--background_data', type=str, default='full',
                       choices=['full', 'subset', 'none'],
                       help='Background data strategy')
    # parser.add_argument('--importance_method', type=str, default='shap',
    #                    choices=['gradient', 'marginal', 'shap'],
    #                    help='Method for computing feature importance')
    parser.add_argument('--sensor_selection', action='store_true', default=True,
                       help='Enable sensor-based channel selection (logical sensors)')
    parser.add_argument('--no_sensor_selection', action='store_false', dest='sensor_selection',
                       help='Disable sensor selection (use global channel selection)')
    
    # Optimization parameters
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Learning rate for optimization')
    parser.add_argument('--max_itr', type=int, default=5000,
                       help='Maximum number of iterations')
    parser.add_argument('--enable_lr_decay', action='store_true', default=True,
                       help='Enable learning rate decay')
    parser.add_argument('--lr_decay', type=float, default=0.99,
                       help='Learning rate decay factor')
    
    # Loss function coefficients
    parser.add_argument('--l_budget_coeff', type=float, default=0.6,
                       help='Budget loss coefficient')
    parser.add_argument('--l_tv_norm_coeff', type=float, default=0.5,
                       help='Total variation norm loss coefficient')
    parser.add_argument('--l_max_coeff', type=float, default=0.7,
                       help='Maximization loss coefficient')
    parser.add_argument('--l_sparse_coeff', type=float, default=0.3,
                       help='Sparsity loss coefficient')
    parser.add_argument('--l_valid_coeff', type=float, default=0.7,
                       help='Validity loss coefficient')
    parser.add_argument('--l_group_coeff', type=float, default=0.5,
                       help='Group consistency loss coefficient')
    parser.add_argument('--use_mcels_loss', action='store_true', default=False,
                       help='Use MCELS loss instead of adaptive MO loss')
    
    # Mask initialization parameters
    parser.add_argument('--mask_threshold', type=float, default=0.5,
                       help='Threshold for binary mask conversion')
    
    # Counterfactual generation parameters
    parser.add_argument('--min_target_probability', type=float, default=0.7,
                       help='Minimum target probability for successful counterfactual')
    parser.add_argument('--guide_neighbors', type=int, default=1,
                       help='Number of guide neighbors for counterfactual generation')
    parser.add_argument('--guide_distance_metric', type=str, default='euclidean',
                       choices=['euclidean', 'cosine', 'manhattan'],
                       help='Distance metric for guide selection')
    parser.add_argument('--target_class', type=int, default=0,
                       help='Target class for counterfactual generation')
    # parser.add_argument('--conf_threshold', type=float, default=0.5,
    #                    help='Confidence threshold for classification')
    
    # Loss function toggles
    parser.add_argument('--enable_budget', action='store_true', default=True,
                       help='Enable budget constraint')
    parser.add_argument('--enable_tvnorm', action='store_true', default=True,
                       help='Enable total variation norm regularization')
    parser.add_argument('--enable_sparsity', action='store_true', default=True,
                       help='Enable sparsity regularization')
    
    # TV norm parameters
    parser.add_argument('--tv_beta', type=int, default=3,
                       help='TV norm beta parameter')
    
    # Random seed settings
    parser.add_argument('--enable_seed', action='store_true', default=True,
                       help='Enable global random seed')
    parser.add_argument('--seed_value', type=int, default=42,
                       help='Random seed value')
    parser.add_argument('--enable_seed_per_instance', action='store_true', default=False,
                       help='Enable different seed per instance')
    
    # Dataset information
    parser.add_argument('--dataset', type=str, default='movement',
                       help='Dataset name')
    
    # Adaptive Multi-Objective specific parameters
    parser.add_argument('--l_target_coeff', type=float, default=1.0,
                       help='Target loss coefficient for adaptive multi-objective')
    parser.add_argument('--l_sparsity_coeff', type=float, default=0.5,
                       help='Sparsity loss coefficient for adaptive multi-objective')
    parser.add_argument('--l_smoothness_coeff', type=float, default=0.3,
                       help='Smoothness loss coefficient for adaptive multi-objective')
    parser.add_argument('--l_group_sparse_coeff', type=float, default=0.4,
                       help='Group sparsity loss coefficient for adaptive multi-objective')
    parser.add_argument('--weight_adaptation_rate', type=float, default=0.1,
                       help='Rate of weight adaptation during optimization')
    parser.add_argument('--target_threshold', type=float, default=0.8,
                       help='Target threshold for weight adaptation')
    parser.add_argument('--max_groups_ratio', type=float, default=0.5,
                       help='Maximum ratio of groups to select (0.5 = half of available groups)')
    parser.add_argument('--group_level', type=str, default='modality',
                       choices=['sensor', 'modality'],
                       help='Grouping level: sensor (8 groups) or modality (16 groups)')
    parser.add_argument('--top_k_channels', type=int, default=24,
                       help='Number of top channels to select')
    parser.add_argument('--shapley_max_evals', type=int, default=50,
                       help='Maximum evaluations for Shapley value computation')
    parser.add_argument('--ranking_method', type=str, default='shapley',
                       choices=['shapley', 'gradient', 'random'],
                       help='Method for ranking channel importance')
    parser.add_argument('--max_groups', type=int, default=8,
                       help='Maximum number of groups to select')
    
    # Logging and output
    parser.add_argument('--enable_wandb', action='store_true', default=False,
                       help='Enable Weights & Biases logging')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    return args
