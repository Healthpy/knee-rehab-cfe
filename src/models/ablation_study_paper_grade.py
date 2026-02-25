"""
Paper-Grade Fair Ablation Study (Fixed Test Set, Single Evaluation)

Purpose
-------
Produce publication-ready ablation evidence aligned with the paper hypothesis:
- Fair comparisons on fixed test set
- Performance evaluation across methods and group ratios
- Summary statistics for key metrics

Design
------
- Baseline: M-CELS
- SA variants with varying group ratios (SHAP-based preselection):
    * r = 0.3, 0.6, 0.9
- SA mechanism checks (fixed ratio):
    * no SHAP ranking (all groups)
    * no adaptive weights
    * no group sparsity loss
- LG variants (fixed ratio):
    * full (adaptive prune)
    * no SHAP ranking
    * fixed prune (no adaptive prune)

Outputs
-------
results/ablation/paper_grade/<split>/
  - ablation_paper_grade_results.csv (all methods and ratios)
  - ablation_paper_grade_summary.csv (aggregated by method)
  - ablation_paper_grade_sa_ratio_analysis.csv (SA ratio sweep)
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.explainer.mcels_explainer import MCELSExplainer
from src.explainer.shapley_adaptive_explainer import ShapleyAdaptiveExplainer
from src.explainer.learnable_gate_explainer import LearnableGateExplainer
from src.models.evaluation_utils import (
    load_model,
    load_imu_data,
    create_train_test_split,
    create_subject_split,
    select_test_samples,
    create_predict_fn,
    IMUExplainerEvaluator,
    run_evaluation_loop,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Paper-grade ablation on fixed test set")
    parser.add_argument("--split", type=str, default="subject", choices=["trial", "subject"],
                        help="Evaluation split protocol (default: subject)")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of test samples to evaluate (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for split/sample selection (default: 42)")
    parser.add_argument("--group_level", type=str, default="modality", choices=["sensor", "modality"],
                        help="Group level for SA/LG gating (default: modality)")
    parser.add_argument("--sa_ratios", type=float, nargs="*", default=[0.3, 0.6, 0.9],
                        help="Group ratios for SA ablation sweep (default: 0.3 0.6 0.9)")
    parser.add_argument("--lg_ratio", type=float, default=0.8,
                        help="Fixed group ratio for LG variants (default: 0.8)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    return parser.parse_args()


def _build_mcels(split, predict_fn, device):
    class Args:
        pass

    Args.algo = "M-CELS"
    Args.max_itr = 5000
    Args.l_tv_norm_coeff = 0.6
    Args.l_budget_coeff = 0.5
    Args.l_max_coeff = 0.7
    Args.enable_lr_decay = True
    Args.lr_decay = 0.9991
    Args.enable_tvnorm = True
    Args.enable_budget = True
    Args.learning_rate = 0.01
    Args.tv_beta = 3

    return MCELSExplainer(
        background_data=split["imu_train"],
        background_label=split["y_train"],
        predict_fn=predict_fn,
        enable_wandb=False,
        args=Args(),
        use_cuda=device.type == "cuda",
    )


def _build_sa(split, predict_fn, device, cfg, group_level, group_ratio):
    class Args:
        pass

    Args.algo = "Shapley-Adaptive"
    Args.max_itr = 5000
    Args.l_max_coeff = 1.0
    Args.l_budget_coeff = 0.55
    Args.l_tv_norm_coeff = 0.50
    Args.l_group_sparse_coeff = 0.15
    Args.enable_lr_decay = True
    Args.lr_decay = 0.9991
    Args.learning_rate = 0.01
    Args.use_shapley_ranking = cfg.get("use_shap", True)
    Args.group_level = group_level
    Args.max_groups_ratio = group_ratio
    Args.min_target_probability = 0.55
    Args.target_threshold = 0.72
    Args.enable_adaptive_weights = cfg.get("adaptive_weights", True)
    Args.enable_group_sparsity_loss = cfg.get("group_sparsity_loss", True)
    Args.refine_thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    Args.refine_temporal_kernel = 15
    Args.refine_require_valid = True
    Args.refine_cf_blends = [0.0, 0.3, 0.6]

    return ShapleyAdaptiveExplainer(
        background_data=split["imu_train"],
        background_label=split["y_train"],
        predict_fn=predict_fn,
        enable_wandb=False,
        args=Args(),
        use_cuda=device.type == "cuda",
    )


def _build_lg(split, predict_fn, device, cfg, group_level, group_ratio):
    class Args:
        pass

    Args.algo = "Learnable-Gate"
    Args.max_itr = 5000
    Args.l_max_coeff = 1.0
    Args.l_budget_coeff = 0.75
    Args.l_tv_norm_coeff = 0.65
    Args.l_group_sparse_coeff = 0.60
    Args.l_gate_coeff = 0.90
    Args.enable_lr_decay = True
    Args.lr_decay = 0.9991
    Args.learning_rate = 0.01
    Args.gate_lr_multiplier = 1.0
    Args.use_shapley_ranking = cfg.get("use_shap", True)
    Args.group_level = group_level
    Args.max_groups_ratio = group_ratio
    Args.gate_warmup_itr = 150
    Args.gate_prune_threshold = 0.3
    Args.adaptive_prune = cfg.get("adaptive_prune", True)
    Args.min_target_probability = 0.65
    Args.target_threshold = 0.75
    Args.refine_thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    Args.refine_temporal_kernel = 15
    Args.refine_require_valid = True
    Args.refine_cf_blends = [0.0, 0.3, 0.6]

    return LearnableGateExplainer(
        background_data=split["imu_train"],
        background_label=split["y_train"],
        predict_fn=predict_fn,
        enable_wandb=False,
        args=Args(),
        use_cuda=device.type == "cuda",
    )


def _extract_metrics(results):
    """Extract performance metrics from evaluation results (mean and std over samples)."""
    valid = np.array(results["validity"], dtype=float)
    conf = np.array(results["confidence"], dtype=float)
    l2 = np.array([p["imu_l2"] for p in results["proximity"]], dtype=float)
    ch = np.array([s["imu_channels_changed"] for s in results["sparsity"]], dtype=float)
    sg = np.array([g["imu_groups_changed"] for g in results["group_sparsity_sensor"]], dtype=float)
    mg = np.array([g["imu_groups_changed"] for g in results["group_sparsity_modality"]], dtype=float)
    tg = np.array([c["imu_temporal_grad"] for c in results["continuity"]], dtype=float)
    it = np.array(results["iterations"], dtype=float)
    tm = np.array(results["time_seconds"], dtype=float)
    
    n = len(valid)
    
    return {
        "n_samples": n,
        "success_rate_mean": float(valid.mean() * 100),
        "success_rate_std": float(valid.std(ddof=1) * 100) if n > 1 else 0.0,
        "confidence_mean": float(conf.mean() * 100),
        "confidence_std": float(conf.std(ddof=1) * 100) if n > 1 else 0.0,
        "l2_mean": float(l2.mean()),
        "l2_std": float(l2.std(ddof=1)) if n > 1 else 0.0,
        "channels_mean": float(ch.mean()),
        "channels_std": float(ch.std(ddof=1)) if n > 1 else 0.0,
        "sensor_groups_mean": float(sg.mean()),
        "sensor_groups_std": float(sg.std(ddof=1)) if n > 1 else 0.0,
        "modality_groups_mean": float(mg.mean()),
        "modality_groups_std": float(mg.std(ddof=1)) if n > 1 else 0.0,
        "temporal_grad_mean": float(tg.mean()),
        "temporal_grad_std": float(tg.std(ddof=1)) if n > 1 else 0.0,
        "iterations_mean": float(it.mean()),
        "iterations_std": float(it.std(ddof=1)) if n > 1 else 0.0,
        "time_mean": float(tm.mean()),
        "time_std": float(tm.std(ddof=1)) if n > 1 else 0.0,
    }


def main():
    cli = parse_args()

    if cli.output_dir:
        out_dir = Path(cli.output_dir)
    else:
        out_dir = Path("results") / "ablation" / "paper_grade" / cli.split
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = (
        "models/best_fcn_imu_subject_split.pth"
        if cli.split == "subject"
        else "models/best_fcn_imu_trial_split.pth"
    )
    norm_path = (
        "models/fcn_imu_subject_normalization.npz"
        if cli.split == "subject"
        else "models/fcn_imu_normalization.npz"
    )

    print("=" * 78)
    print("PAPER-GRADE ABLATION STUDY")
    print("=" * 78)
    print(f"Split       : {cli.split}")
    print(f"n_samples   : {cli.n_samples}")
    print(f"seed        : {cli.seed}")
    print(f"group_level : {cli.group_level}")
    print(f"SA ratios   : {cli.sa_ratios}")
    print(f"LG ratio    : {cli.lg_ratio}")

    model, device = load_model(model_path=model_path)
    imu_processed, labels, subject_ids, imu_mean, imu_std = load_imu_data(norm_stats_path=norm_path)
    predict_fn = create_predict_fn(model, device)
    evaluator = IMUExplainerEvaluator(model, device=device)

    # Create split and select test samples
    print(f"\nLoading data with {cli.split} split...")
    if cli.split == "subject":
        split = create_subject_split(imu_processed, labels, subject_ids, seed=cli.seed)
    else:
        split = create_train_test_split(imu_processed, labels, subject_ids=subject_ids, seed=cli.seed)

    selected = select_test_samples(
        model,
        device,
        split["imu_test"],
        split["y_test"],
        n_samples=cli.n_samples,
        seed=cli.seed,
    )
    print(f"Selected {len(selected)} test samples for evaluation")

    all_results = []

    # ========================================================================
    # M-CELS Baseline
    # ========================================================================
    print("\n[M-CELS] Running baseline...")
    explainer = _build_mcels(split, predict_fn, device)
    results = run_evaluation_loop(
        explainer,
        evaluator,
        split["imu_test"],
        split["y_test"],
        selected,
        imu_mean=imu_mean,
        imu_std=imu_std,
        vis_dir=None,
        vis_count=0,
        change_threshold=0.0001,
        channel_threshold=1e-6,
        verbose=False,
    )
    row = {
        "split": cli.split,
        "group_level": cli.group_level,
        "method": "mcels",
        "config_label": "M-CELS",
        "group_ratio": None,
        "ablation_type": "baseline",
    }
    row.update(_extract_metrics(results))
    all_results.append(row)

    # ========================================================================
    # SA with varying group ratios (SHAP-based)
    # ========================================================================
    print("\nSA RATIO SWEEP (SHAP-based preselection):")
    sa_ratio_results = []
    for ratio in cli.sa_ratios:
        print(f"  [SA SHAP {ratio:.1f}] Running...")
        cfg = {"use_shap": True, "adaptive_weights": True, "group_sparsity_loss": True}
        explainer = _build_sa(split, predict_fn, device, cfg, cli.group_level, ratio)
        results = run_evaluation_loop(
            explainer,
            evaluator,
            split["imu_test"],
            split["y_test"],
            selected,
            imu_mean=imu_mean,
            imu_std=imu_std,
            vis_dir=None,
            vis_count=0,
            change_threshold=0.0001,
            channel_threshold=1e-6,
            verbose=False,
        )
        row = {
            "split": cli.split,
            "group_level": cli.group_level,
            "method": "sa",
            "config_label": f"SA (SHAP r={ratio:.1f})",
            "group_ratio": ratio,
            "ablation_type": "sa_ratio_sweep",
        }
        row.update(_extract_metrics(results))
        all_results.append(row)
        sa_ratio_results.append(row)

    # ========================================================================
    # SA mechanism ablations (fixed ratio)
    # ========================================================================
    print("\nSA MECHANISM ABLATIONS (fixed ratio r=0.8):")
    sa_ablations = [
        {"config_id": "sa_no_shap", "label": "SA (no SHAP, all groups)", 
         "use_shap": False, "adaptive_weights": True, "group_sparsity_loss": True},
        {"config_id": "sa_no_adapt", "label": "SA (no adaptive weights)", 
         "use_shap": True, "adaptive_weights": False, "group_sparsity_loss": True},
        {"config_id": "sa_no_group_loss", "label": "SA (no group loss)", 
         "use_shap": True, "adaptive_weights": True, "group_sparsity_loss": False},
    ]

    for cfg in sa_ablations:
        print(f"  [{cfg['config_id']}] {cfg['label']}...")
        explainer = _build_sa(split, predict_fn, device, cfg, cli.group_level, cli.lg_ratio)
        results = run_evaluation_loop(
            explainer,
            evaluator,
            split["imu_test"],
            split["y_test"],
            selected,
            imu_mean=imu_mean,
            imu_std=imu_std,
            vis_dir=None,
            vis_count=0,
            change_threshold=0.0001,
            channel_threshold=1e-6,
            verbose=False,
        )
        row = {
            "split": cli.split,
            "group_level": cli.group_level,
            "method": "sa",
            "config_label": cfg["label"],
            "group_ratio": cli.lg_ratio,
            "ablation_type": "sa_mechanism",
        }
        row.update(_extract_metrics(results))
        all_results.append(row)

    # ========================================================================
    # LG variants (fixed ratio)
    # ========================================================================
    print("\nLEARNABLE GATE VARIANTS (fixed ratio r=0.8):")
    lg_variants = [
        {"config_id": "lg_shap_pruned", "label": "LG (SHAP pruned)", 
         "use_shap": True, "adaptive_prune": True},
        {"config_id": "lg_no_shap", "label": "LG (no SHAP, from scratch)", 
         "use_shap": False, "adaptive_prune": True},
        {"config_id": "lg_final", "label": "LG final (fixed prune)", 
         "use_shap": True, "adaptive_prune": False},
    ]

    for cfg in lg_variants:
        print(f"  [{cfg['config_id']}] {cfg['label']}...")
        explainer = _build_lg(split, predict_fn, device, cfg, cli.group_level, cli.lg_ratio)
        results = run_evaluation_loop(
            explainer,
            evaluator,
            split["imu_test"],
            split["y_test"],
            selected,
            imu_mean=imu_mean,
            imu_std=imu_std,
            vis_dir=None,
            vis_count=0,
            change_threshold=0.0001,
            channel_threshold=1e-6,
            verbose=False,
        )
        row = {
            "split": cli.split,
            "group_level": cli.group_level,
            "method": "lg",
            "config_label": cfg["label"],
            "group_ratio": cli.lg_ratio,
            "ablation_type": "lg_variant",
        }
        row.update(_extract_metrics(results))
        all_results.append(row)

    # ========================================================================
    # Save results
    # ========================================================================
    df_results = pd.DataFrame(all_results)
    results_csv = out_dir / "ablation_paper_grade_results.csv"
    df_results.to_csv(results_csv, index=False, float_format="%.6f")
    print(f"\n✓ All results saved to {results_csv}")

    # Summary by method (excluding ratio-sweep variants)
    df_summary = df_results[df_results["ablation_type"] != "sa_ratio_sweep"].copy()
    df_summary = df_summary.drop(columns=["split", "group_level", "group_ratio", "ablation_type"])
    summary_csv = out_dir / "ablation_paper_grade_summary.csv"
    df_summary.to_csv(summary_csv, index=False, float_format="%.6f")
    print(f"✓ Summary saved to {summary_csv}")

    # SA ratio analysis (only ratio-sweep variants)
    df_sa_ratio = df_results[df_results["ablation_type"] == "sa_ratio_sweep"].copy()
    sa_ratio_csv = out_dir / "ablation_paper_grade_sa_ratio_analysis.csv"
    df_sa_ratio.to_csv(sa_ratio_csv, index=False, float_format="%.6f")
    print(f"✓ SA ratio analysis saved to {sa_ratio_csv}")

    print("\n" + "=" * 78)
    print("✓ Ablation study complete")
    print(f"  Results    : {results_csv}")
    print(f"  Summary    : {summary_csv}")
    print(f"  SA Ratios  : {sa_ratio_csv}")
    print("=" * 78)


if __name__ == "__main__":
    main()
