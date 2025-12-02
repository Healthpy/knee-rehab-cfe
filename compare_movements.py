#!/usr/bin/env python3
"""
Comparison script for counterfactual analysis results across different movements.

This script compares results between different movement types (squat vs extension)
and provides side-by-side analysis.

Date: November 6, 2025
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List
import seaborn as sns

def load_results(results_dir: str) -> Dict:
    """Load results from explainer_results.json."""
    results_file = Path(results_dir) / 'explainer_results.json'
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        return json.load(f)

def compare_movements(squat_dir: str, extension_dir: str, gait_dir: str, save_dir: str = None):
    """Compare counterfactual results between squat and extension and gait movements."""
    
    # Load results
    squat_results = load_results(squat_dir)
    extension_results = load_results(extension_dir)
    gait_results = load_results(gait_dir)

    # Create comparison directory
    if save_dir is None:
        save_dir = "results/comparisons"
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("MOVEMENT COMPARISON: SQUAT vs EXTENSION vs GAIT")
    print("="*80)
    
    # Extract comparison metrics
    squat_eval = squat_results.get('evaluation', {})
    extension_eval = extension_results.get('evaluation', {})
    gait_eval = gait_results.get('evaluation', {})

    squat_mask = np.array(squat_results['mask'])
    extension_mask = np.array(extension_results['mask'])
    gait_mask = np.array(gait_results['mask'])

    squat_orig = np.array(squat_results['original_input'])
    squat_cf = np.array(squat_results['counterfactual'])
    extension_orig = np.array(extension_results['original_input'])
    extension_cf = np.array(extension_results['counterfactual'])
    gait_orig = np.array(gait_results['original_input'])
    gait_cf = np.array(gait_results['counterfactual'])

    # Calculate metrics
    comparison_data = {
        'Movement': ['Squat', 'Extension', 'Gait'],
        'Success': [squat_eval.get('valid', False), extension_eval.get('valid', False), gait_eval.get('valid', False)],
        'Confidence': [squat_eval.get('confidence', 0), extension_eval.get('confidence', 0), gait_eval.get('confidence', 0)],
        'L2_Distance': [squat_eval.get('l2_distance', 0), extension_eval.get('l2_distance', 0), gait_eval.get('l2_distance', 0)],
        'L1_Distance': [squat_eval.get('l1_distance', 0), extension_eval.get('l1_distance', 0), gait_eval.get('l1_distance', 0)],
        'Mask_Sparsity': [
            1.0 - (np.sum(squat_mask > 0.5) / squat_mask.size),
            1.0 - (np.sum(extension_mask > 0.5) / extension_mask.size),
            1.0 - (np.sum(gait_mask > 0.5) / gait_mask.size)
        ],
        'Selected_Groups': [
            len(squat_results.get('selected_groups', [])),
            len(extension_results.get('selected_groups', [])),
            len(gait_results.get('selected_groups', []))
        ],
        'Optimization_Iterations': [
            squat_results.get('optimization_trace', {}).get('iterations', 0),
            extension_results.get('optimization_trace', {}).get('iterations', 0),
            gait_results.get('optimization_trace', {}).get('iterations', 0)
        ]
    }
    
    # Print comparison table
    df_comparison = pd.DataFrame(comparison_data)
    print("\nQUANTITATIVE COMPARISON:")
    print("-" * 50)
    print(df_comparison.to_string(index=False, float_format='%.3f'))
    
    # Group selection comparison
    print(f"\nGROUP SELECTION COMPARISON:")
    print("-" * 50)
    squat_groups = set(squat_results.get('selected_groups', []))
    extension_groups = set(extension_results.get('selected_groups', []))
    gait_groups = set(gait_results.get('selected_groups', []))

    common_groups = squat_groups.intersection(extension_groups).intersection(gait_groups)
    squat_only = squat_groups - extension_groups - gait_groups
    extension_only = extension_groups - squat_groups - gait_groups
    gait_only = gait_groups - squat_groups - extension_groups

    print(f"Common Groups ({len(common_groups)}): {', '.join(sorted(common_groups))}")
    print(f"Squat Only ({len(squat_only)}): {', '.join(sorted(squat_only))}")
    print(f"Extension Only ({len(extension_only)}): {', '.join(sorted(extension_only))}")
    print(f"Gait Only ({len(gait_only)}): {', '.join(sorted(gait_only))}")
    print("\nVISUAL COMPARISON:")       
    # Create visualization comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Metrics comparison bar chart
    ax1 = axes[0, 0]
    metrics = ['Confidence', 'L2_Distance', 'Mask_Sparsity']
    squat_vals = [comparison_data[m][0] for m in metrics]
    extension_vals = [comparison_data[m][1] for m in metrics]
    gait_vals = [comparison_data[m][2] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    ax1.bar(x - 2*width/3, squat_vals, width, label='Squat', alpha=0.8, color='skyblue')
    ax1.bar(x - width/3, extension_vals, width, label='Extension', alpha=0.8, color='lightcoral')
    ax1.bar(x + width/3, gait_vals, width, label='Gait', alpha=0.8, color='lightgreen') 
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Values')
    ax1.set_title('Key Metrics Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Mask sparsity comparison
    ax2 = axes[0, 1]
    channel_sparsity_squat = np.mean(squat_mask, axis=1)
    channel_sparsity_extension = np.mean(extension_mask, axis=1)
    channel_sparsity_gait = np.mean(gait_mask, axis=1)
    
    channels = np.arange(len(channel_sparsity_squat))
    ax2.plot(channels, channel_sparsity_squat, 'b-', label='Squat', alpha=0.8, linewidth=2)
    ax2.plot(channels, channel_sparsity_extension, 'r-', label='Extension', alpha=0.8, linewidth=2)
    ax2.plot(channels, channel_sparsity_gait, 'g-', label='Gait', alpha=0.8, linewidth=2)
    ax2.set_xlabel('Channel Index')
    ax2.set_ylabel('Average Mask Activation')
    ax2.set_title('Channel-wise Mask Activation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Signal change magnitude comparison
    ax3 = axes[0, 2]
    squat_change = np.abs(squat_cf - squat_orig)
    extension_change = np.abs(extension_cf - extension_orig)
    gait_change = np.abs(gait_cf - gait_orig)
    
    squat_channel_change = np.mean(squat_change, axis=1)
    extension_channel_change = np.mean(extension_change, axis=1)
    gait_channel_change = np.mean(gait_change, axis=1)

    ax3.plot(channels, squat_channel_change, 'b-', label='Squat', alpha=0.8, linewidth=2)
    ax3.plot(channels, extension_channel_change, 'r-', label='Extension', alpha=0.8, linewidth=2)
    ax3.plot(channels, gait_channel_change, 'g-', label='Gait', alpha=0.8, linewidth=2)
    ax3.set_xlabel('Channel Index')
    ax3.set_ylabel('Average Signal Change')
    ax3.set_title('Channel-wise Signal Changes')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Group importance comparison (if available)
    ax4 = axes[1, 0]
    squat_importance = squat_results.get('group_importance', {})
    extension_importance = extension_results.get('group_importance', {})
    gait_importance = gait_results.get('group_importance', {})

    if squat_importance and extension_importance and gait_importance:
        common_groups_list = sorted(common_groups)
        if common_groups_list:
            squat_imp_vals = [squat_importance.get(g, 0) for g in common_groups_list]
            extension_imp_vals = [extension_importance.get(g, 0) for g in common_groups_list]
            gait_imp_vals = [gait_importance.get(g, 0) for g in common_groups_list]

            x = np.arange(len(common_groups_list))
            ax4.bar(x - 2*width/3, squat_imp_vals, width, label='Squat', alpha=0.8, color='skyblue')
            ax4.bar(x - width/3, extension_imp_vals, width, label='Extension', alpha=0.8, color='lightcoral')
            ax4.bar(x + width/3, gait_imp_vals, width, label='Gait', alpha=0.8, color='lightgreen') 
            ax4.set_xlabel('Common Groups')
            ax4.set_ylabel('Importance Score')
            ax4.set_title('Group Importance Comparison')
            ax4.set_xticks(x)
            ax4.set_xticklabels(common_groups_list, rotation=45, ha='right')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No Common Groups', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Group Importance Comparison')
    else:
        ax4.text(0.5, 0.5, 'Group Importance Data\nNot Available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Group Importance Comparison')
    
    # 5. Optimization convergence comparison
    ax5 = axes[1, 1]
    squat_trace = squat_results.get('optimization_trace', {})
    extension_trace = extension_results.get('optimization_trace', {})
    gait_trace = gait_results.get('optimization_trace', {})

    if 'validity_history' in squat_trace and 'validity_history' in extension_trace and 'validity_history' in gait_trace:
        squat_validity = squat_trace['validity_history']
        extension_validity = extension_trace['validity_history']
        gait_validity = gait_trace['validity_history']

        ax5.plot(range(len(squat_validity)), squat_validity, 'b-', label='Squat', alpha=0.8, linewidth=2)
        ax5.plot(range(len(extension_validity)), extension_validity, 'r-', label='Extension', alpha=0.8, linewidth=2)
        ax5.plot(range(len(gait_validity)), gait_validity, 'g-', label='Gait', alpha=0.8, linewidth=2)          
        ax5.axhline(y=0.7, color='gray', linestyle='--', alpha=0.7, label='Threshold')
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Validity Score')
        ax5.set_title('Optimization Convergence')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'Optimization Trace\nNot Available', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Optimization Convergence')
    
    # 6. Group selection Venn diagram style
    ax6 = axes[1, 2]
    
    # Create a simple visualization for group overlap
    total_groups = len(squat_groups.union(extension_groups).union(gait_groups))
    common_ratio = len(common_groups) / max(1, total_groups)
    squat_only_ratio = len(squat_only) / max(1, total_groups)
    extension_only_ratio = len(extension_only) / max(1, total_groups)
    gait_only_ratio = len(gait_only) / max(1, total_groups)

    sizes = [len(common_groups), len(squat_only), len(extension_only), len(gait_only)]
    labels = [f'Common\n({len(common_groups)})', f'Squat Only\n({len(squat_only)})', f'Extension Only\n({len(extension_only)})', f'Gait Only\n({len(gait_only)})']
    colors = ['lightgreen', 'skyblue', 'lightcoral', 'lightblue']

    if sum(sizes) > 0:
        ax6.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax6.set_title('Group Selection Overlap')
    
    plt.suptitle('Squat vs Extension Movement Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save comparison plot
    comparison_plot_path = save_path / 'movement_comparison.png'
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {comparison_plot_path}")
    plt.show()
    
    # Save comparison data
    comparison_csv_path = save_path / 'movement_comparison.csv'
    df_comparison.to_csv(comparison_csv_path, index=False)
    print(f"Comparison data saved to: {comparison_csv_path}")
    
    print("\n" + "="*80)

def main():
    """Main function."""
    squat_dir = "results/adaptive_multi_objective_squat"
    extension_dir = "results/adaptive_multi_objective_extension"
    gait_dir = "results/adaptive_multi_objective_gait"  
    
    print("Comparing Squat vs Extension vs Gait Movement Results...")
    compare_movements(squat_dir, extension_dir, gait_dir)


if __name__ == '__main__':
    main()