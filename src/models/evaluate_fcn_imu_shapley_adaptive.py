"""
Shapley-Adaptive Explainer Evaluation for FCN IMU-Only Model

Evaluates Shapley-Guided Adaptive Multi-Objective explanations on the FCN model.
Uses shared utilities from evaluation_utils.py.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.explainer.shapley_adaptive_explainer import ShapleyAdaptiveExplainer
from src.models.evaluation_utils import (
    load_model, load_imu_data, create_train_test_split,
    select_test_samples, create_predict_fn, IMUExplainerEvaluator,
    run_evaluation_loop, save_detailed_results,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Shapley-Adaptive Explainer Evaluation')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of test samples to evaluate (default: 10)')
    parser.add_argument('--group_level', type=str, default='modality',
                        choices=['sensor', 'modality'],
                        help="Group level: 'sensor' (8 groups) or 'modality' (16 groups) (default: modality)")
    parser.add_argument('--group_ratio', type=float, default=0.75,
                        help='Max groups ratio: fraction of groups to select (default: 0.5)')
    parser.add_argument('--no_shap', action='store_true',
                        help='Disable SHAP ranking (use all groups)')
    return parser.parse_args()


def main():
    print("=" * 70)
    print("FCN IMU-Only Model - Shapley-Adaptive Explainer Evaluation")
    print("=" * 70)

    # 1. Load model
    print("\n[1/7] Loading FCN IMU-only model...")
    model, device = load_model()

    # 2. Load data
    print("\n[2/7] Loading IMU data...")
    imu_processed, labels, subject_ids, imu_mean, imu_std = load_imu_data()

    # 3. Train/val/test split
    print("\n[3/7] Splitting data (trial-level 70/15/15)...")
    split = create_train_test_split(imu_processed, labels, subject_ids=subject_ids)

    # 4. Select test samples
    cli = parse_args()
    print(f"\n[4/7] Selecting test samples (n={cli.n_samples})...")
    selected = select_test_samples(
        model, device, split['imu_test'], split['y_test'], n_samples=cli.n_samples
    )

    # 5. Initialize Shapley-Adaptive explainer
    print("\n[5/7] Initializing Shapley-Adaptive explainer...")
    predict_fn = create_predict_fn(model, device)

    class Args:
        algo = 'Shapley-Adaptive'
        max_itr = 5000
        l_max_coeff = 1.0
        l_budget_coeff = 0.5
        l_tv_norm_coeff = 0.3
        l_group_sparse_coeff = 0.3
        enable_lr_decay = True
        lr_decay = 0.9991
        learning_rate = 0.01
        use_shapley_ranking = not cli.no_shap
        group_level = cli.group_level
        max_groups_ratio = cli.group_ratio
        min_target_probability = 0.7
        target_threshold = 0.8

    explainer = ShapleyAdaptiveExplainer(
        background_data=split['imu_train'],
        background_label=split['y_train'],
        predict_fn=predict_fn,
        enable_wandb=False,
        args=Args(),
        use_cuda=device.type == 'cuda'
    )
    print("\u2713 Shapley-Adaptive explainer initialized")
    print(f"  SHAP ranking: {'Enabled' if Args.use_shapley_ranking else 'Disabled'}")
    print(f"  Group level: {Args.group_level}")
    print(f"  Max groups ratio: {Args.max_groups_ratio}")

    evaluator = IMUExplainerEvaluator(model, device=device)

    # 6. Run evaluation loop
    print("\n[6/7] Generating Shapley-Adaptive explanations on test samples...")
    results = run_evaluation_loop(
        explainer, evaluator,
        split['imu_test'], split['y_test'], selected,
        imu_mean=imu_mean, imu_std=imu_std,
        vis_dir='results/evaluation/shapley_adaptive/visualizations',
        vis_count=3,
        change_threshold=0.0001,
        channel_threshold=0.01,
    )

    # 7. Save results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (Test Set Only)")
    print("=" * 70)

    save_detailed_results(
        results, selected,
        csv_path='results/evaluation/shapley_adaptive/fcn_imu_shapley_adaptive_evaluation.csv',
        subject_ids_test=split.get('subject_ids_test'),
        extra_summary={
            'Method': 'Shapley-Adaptive',
            'SHAP Ranking': 'Enabled' if Args.use_shapley_ranking else 'Disabled',
            'Group Level': Args.group_level,
        }
    )

    print("\n" + "=" * 70)
    print("\u2713 Shapley-Adaptive Evaluation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
