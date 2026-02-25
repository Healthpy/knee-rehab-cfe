"""
M-CELS Explainer Evaluation for FCN IMU-Only Model

Evaluates M-CELS explanations on the FCN model trained with IMU data only.
Uses shared utilities from evaluation_utils.py.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.explainer.mcels_explainer import MCELSExplainer
from src.models.evaluation_utils import (
    load_model, load_imu_data, create_train_test_split,
    select_test_samples, create_predict_fn, IMUExplainerEvaluator,
    run_evaluation_loop, save_detailed_results,
)


def parse_args():
    parser = argparse.ArgumentParser(description='M-CELS Explainer Evaluation')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of test samples to evaluate (default: 10)')
    return parser.parse_args()


def main():
    print("=" * 70)
    print("FCN IMU-Only Model - M-CELS Explainer Evaluation")
    print("=" * 70)

    # 1. Load model
    print("\n[1/5] Loading FCN IMU-only model...")
    model, device = load_model()

    # 2. Load data
    print("\n[2/5] Loading IMU data...")
    imu_processed, labels, subject_ids, imu_mean, imu_std = load_imu_data()

    # 3. Train/val/test split
    print("\n[3/5] Splitting data (trial-level 70/15/15)...")
    split = create_train_test_split(imu_processed, labels, subject_ids=subject_ids)

    # 4. Select test samples
    cli = parse_args()
    print(f"\n[4/5] Selecting test samples (n={cli.n_samples})...")
    selected = select_test_samples(
        model, device, split['imu_test'], split['y_test'], n_samples=cli.n_samples
    )

    # 5. Initialize M-CELS explainer
    print("\n[5/5] Initializing M-CELS explainer...")
    predict_fn = create_predict_fn(model, device)

    class Args:
        algo = 'M-CELS'
        max_itr = 5000
        l_tv_norm_coeff = 0.6
        l_budget_coeff = 0.5
        l_max_coeff = 0.7
        enable_lr_decay = True
        lr_decay = 0.9991
        enable_tvnorm = True
        enable_budget = True
        learning_rate = 0.01
        tv_beta = 3

    explainer = MCELSExplainer(
        background_data=split['imu_train'],
        background_label=split['y_train'],
        predict_fn=predict_fn,
        enable_wandb=False,
        args=Args(),
        use_cuda=device.type == 'cuda'
    )
    print("\u2713 M-CELS explainer initialized")

    evaluator = IMUExplainerEvaluator(model, device=device)

    # 6. Run evaluation loop
    print("\n[6/7] Generating M-CELS explanations on test samples...")
    results = run_evaluation_loop(
        explainer, evaluator,
        split['imu_test'], split['y_test'], selected,
        imu_mean=imu_mean, imu_std=imu_std,
        vis_dir='results/evaluation/mcels/visualizations',
        vis_count=3,
        change_threshold=0.00001,
        channel_threshold=0.01,
    )

    # 7. Save results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (Test Set Only)")
    print("=" * 70)

    save_detailed_results(
        results, selected,
        csv_path='results/evaluation/mcels/fcn_imu_mcels_evaluation.csv',
        subject_ids_test=split.get('subject_ids_test'),
    )

    print("\n" + "=" * 70)
    print("\u2713 M-CELS Evaluation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
