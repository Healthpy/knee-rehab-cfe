"""
M-CELS Explainer Evaluation for FCN IMU-Only (Subject-Disjoint Split)

Evaluates M-CELS counterfactual explanations using:
- Subject-disjoint FCN model (`best_fcn_imu_subject_split.pth`)
- Subject-disjoint data split (`create_subject_split`)
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.explainer.mcels_explainer import MCELSExplainer
from src.models.evaluation_utils import (
    load_model,
    load_imu_data,
    create_subject_split,
    select_test_samples,
    create_predict_fn,
    IMUExplainerEvaluator,
    run_evaluation_loop,
    save_detailed_results,
)


def parse_args():
    parser = argparse.ArgumentParser(description="M-CELS Subject-Split Evaluation")
    parser.add_argument("--n_samples", type=int, default=50,
                        help="Number of test samples to evaluate (default: 50)")
    return parser.parse_args()


def main():
    cli = parse_args()

    print("=" * 70)
    print("FCN IMU-Only - M-CELS (Subject-Disjoint Split)")
    print("=" * 70)

    print("\n[1/7] Loading subject-split FCN model...")
    model, device = load_model(model_path="models/best_fcn_imu_subject_split.pth")

    print("\n[2/7] Loading IMU data + subject normalization...")
    imu_processed, labels, subject_ids, imu_mean, imu_std = load_imu_data(
        norm_stats_path="models/fcn_imu_subject_normalization.npz"
    )

    print("\n[3/7] Creating subject-disjoint split...")
    split = create_subject_split(imu_processed, labels, subject_ids, seed=42)

    print(f"\n[4/7] Selecting test samples (n={cli.n_samples})...")
    selected = select_test_samples(
        model, device, split["imu_test"], split["y_test"], n_samples=cli.n_samples
    )

    print("\n[5/7] Initializing M-CELS explainer...")
    predict_fn = create_predict_fn(model, device)

    class Args:
        algo = "M-CELS"
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
        background_data=split["imu_train"],
        background_label=split["y_train"],
        predict_fn=predict_fn,
        enable_wandb=False,
        args=Args(),
        use_cuda=device.type == "cuda",
    )

    evaluator = IMUExplainerEvaluator(model, device=device)

    print("\n[6/7] Running evaluation loop...")
    results = run_evaluation_loop(
        explainer,
        evaluator,
        split["imu_test"],
        split["y_test"],
        selected,
        imu_mean=imu_mean,
        imu_std=imu_std,
        vis_dir="results/evaluation/mcels_subject_split/visualizations",
        vis_count=3,
        change_threshold=0.0001,
        channel_threshold=0.01,
    )

    print("\n[7/7] Saving results...")
    save_detailed_results(
        results,
        selected,
        csv_path="results/evaluation/mcels_subject_split/fcn_imu_mcels_subject_split_evaluation.csv",
        subject_ids_test=split.get("subject_ids_test"),
        extra_summary={
            "Method": "M-CELS",
            "Split": "Subject-disjoint",
            "Model": "best_fcn_imu_subject_split.pth",
        },
    )

    print("\n" + "=" * 70)
    print("✓ Subject-split M-CELS evaluation complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
