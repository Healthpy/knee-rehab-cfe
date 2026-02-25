"""
Learnable Gate Explainer Evaluation for FCN IMU-Only (Subject-Disjoint Split)

Evaluates Learnable-Gate counterfactual explanations using:
- Subject-disjoint FCN model (`best_fcn_imu_subject_split.pth`)
- Subject-disjoint data split (`create_subject_split`)
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.explainer.learnable_gate_explainer import LearnableGateExplainer
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
    parser = argparse.ArgumentParser(description="Learnable-Gate Subject-Split Evaluation")
    parser.add_argument("--n_samples", type=int, default=50,
                        help="Number of test samples to evaluate (default: 50)")
    parser.add_argument("--group_level", type=str, default="modality",
                        choices=["sensor", "modality"],
                        help="Grouping level for SHAP/gates (default: modality)")
    parser.add_argument("--group_ratio", type=float, default=0.6,
                        help="Fraction of groups to allow (default: 0.6)")
    parser.add_argument("--fixed_prune", action="store_true",
                        help="Use fixed gate prune threshold instead of adaptive")
    parser.add_argument("--no_shap", action="store_true",
                        help="Disable SHAP ranking (use all groups)")
    return parser.parse_args()


def main():
    cli = parse_args()

    print("=" * 70)
    print("FCN IMU-Only - Learnable Gate (Subject-Disjoint Split)")
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

    print("\n[5/7] Initializing Learnable Gate explainer...")
    predict_fn = create_predict_fn(model, device)

    class Args:
        algo = "Learnable-Gate"
        max_itr = 5000

        l_max_coeff = 1.0
        l_budget_coeff = 0.5
        l_tv_norm_coeff = 0.3
        l_group_sparse_coeff = 0.3
        l_gate_coeff = 0.4

        enable_lr_decay = True
        lr_decay = 0.9991
        learning_rate = 0.01
        gate_lr_multiplier = 1.0

        use_shapley_ranking = not cli.no_shap
        group_level = cli.group_level
        max_groups_ratio = cli.group_ratio

        gate_warmup_itr = 200
        gate_prune_threshold = 0.3
        adaptive_prune = not cli.fixed_prune

        min_target_probability = 0.7
        target_threshold = 0.8

    explainer = LearnableGateExplainer(
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
        vis_dir="results/evaluation/learnable_gate_subject_split/visualizations",
        vis_count=3,
        change_threshold=0.0001,
        channel_threshold=0.01,
    )

    print("\n[7/7] Saving results...")
    save_detailed_results(
        results,
        selected,
        csv_path="results/evaluation/learnable_gate_subject_split/fcn_imu_learnable_gate_subject_split_evaluation.csv",
        subject_ids_test=split.get("subject_ids_test"),
        extra_summary={
            "Method": "Learnable-Gate",
            "Split": "Subject-disjoint",
            "Model": "best_fcn_imu_subject_split.pth",
            "SHAP Ranking": "Enabled" if Args.use_shapley_ranking else "Disabled",
            "Group Level": Args.group_level,
            "Group Ratio": Args.max_groups_ratio,
            "Prune Mode": "adaptive" if Args.adaptive_prune else f"fixed ({Args.gate_prune_threshold})",
        },
    )

    print("\n" + "=" * 70)
    print("✓ Subject-split Learnable Gate evaluation complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
