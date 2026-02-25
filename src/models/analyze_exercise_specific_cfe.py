"""
Exercise-Specific Counterfactual Analysis

Analyzes LG (SHAP pruned) vs M-CELS on exercise-specific error-to-correct
counterfactual generation for the KneE-PAD dataset.

Generates:
- Per-exercise summary statistics
- Modality group activation frequency comparison
- Heatmaps and LaTeX-ready tables
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Publication-ready style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Exercise mappings
EXERCISE_CLASSES = {
    'Squat': [0, 1, 2],
    'Knee Extension': [3, 4, 5],
    'Gait': [6, 7, 8]
}

EXERCISE_CORRECT_CLASS = {
    'Squat': 0,
    'Knee Extension': 3,
    'Gait': 6
}

# All modality groups in order
ALL_MODALITY_GROUPS = [
    'RF_R_acc', 'RF_R_gyr', 'Ham_R_acc', 'Ham_R_gyr',
    'TA_R_acc', 'TA_R_gyr', 'Gast_R_acc', 'Gast_R_gyr',
    'RF_L_acc', 'RF_L_gyr', 'Ham_L_acc', 'Ham_L_gyr',
    'TA_L_acc', 'TA_L_gyr', 'Gast_L_acc', 'Gast_L_gyr'
]


def load_and_filter_data(csv_path, method_name):
    """Load CSV and filter for error-to-correct counterfactuals."""
    df = pd.read_csv(csv_path)
    
    # Filter for error-to-correct: target_class in {0, 3, 6} and original != target
    correct_targets = [0, 3, 6]
    df_filtered = df[
        (df['target_class'].isin(correct_targets)) &
        (df['original_class'] != df['target_class'])
    ].copy()
    
    # Add exercise label
    def get_exercise(class_id):
        for exercise, classes in EXERCISE_CLASSES.items():
            if class_id in classes:
                return exercise
        return None
    
    df_filtered['exercise'] = df_filtered['original_class'].apply(get_exercise)
    df_filtered['method'] = method_name
    
    return df_filtered


def compute_exercise_statistics(df, exercise):
    """Compute aggregate statistics for a specific exercise."""
    exercise_data = df[df['exercise'] == exercise]
    
    if len(exercise_data) == 0:
        return None
    
    # Compute statistics
    stats = {
        'n_samples': len(exercise_data),
        'success_rate': exercise_data['valid'].sum() / len(exercise_data) * 100,
        'confidence_mean': exercise_data[exercise_data['valid']]['confidence'].mean() * 100,
        'confidence_std': exercise_data[exercise_data['valid']]['confidence'].std() * 100,
        'modality_groups_mean': exercise_data[exercise_data['valid']]['modality_groups_changed'].mean(),
        'modality_groups_std': exercise_data[exercise_data['valid']]['modality_groups_changed'].std(),
        'temporal_grad_mean': exercise_data[exercise_data['valid']]['imu_temporal_grad'].mean(),
        'temporal_grad_std': exercise_data[exercise_data['valid']]['imu_temporal_grad'].std(),
        'time_mean': exercise_data[exercise_data['valid']]['time_seconds'].mean(),
        'time_std': exercise_data[exercise_data['valid']]['time_seconds'].std(),
        'l2_mean': exercise_data[exercise_data['valid']]['imu_l2'].mean(),
        'l2_std': exercise_data[exercise_data['valid']]['imu_l2'].std(),
        'channels_mean': exercise_data[exercise_data['valid']]['imu_channels_changed'].mean(),
        'channels_std': exercise_data[exercise_data['valid']]['imu_channels_changed'].std(),
    }
    
    return stats


def compute_modality_activation_frequency(df, exercise):
    """Compute frequency of each modality group activation for an exercise."""
    exercise_data = df[(df['exercise'] == exercise) & (df['valid'] == True)]
    
    if len(exercise_data) == 0:
        return {group: 0.0 for group in ALL_MODALITY_GROUPS}
    
    n_samples = len(exercise_data)
    activation_counts = Counter()
    
    for _, row in exercise_data.iterrows():
        groups_str = row['changed_modality_groups']
        if pd.notna(groups_str) and groups_str != 'none':
            groups = [g.strip() for g in groups_str.split(';')]
            activation_counts.update(groups)
    
    # Convert to frequency (percentage)
    activation_freq = {group: (activation_counts.get(group, 0) / n_samples) * 100 
                       for group in ALL_MODALITY_GROUPS}
    
    return activation_freq


def generate_exercise_summary_table(lg_data, mcels_data, output_dir):
    """Generate per-exercise summary table comparing LG and M-CELS."""
    
    exercises = ['Squat', 'Knee Extension', 'Gait']
    
    # Collect statistics
    summary_data = []
    for exercise in exercises:
        lg_stats = compute_exercise_statistics(lg_data, exercise)
        mcels_stats = compute_exercise_statistics(mcels_data, exercise)
        
        if lg_stats and mcels_stats:
            summary_data.append({
                'exercise': exercise,
                'method': 'LG (SHAP pruned)',
                **lg_stats
            })
            summary_data.append({
                'exercise': exercise,
                'method': 'M-CELS',
                **mcels_stats
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save CSV
    csv_path = output_dir / 'exercise_specific_summary.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f"✓ Saved exercise summary CSV: {csv_path}")
    
    # Generate LaTeX table
    latex = []
    latex.append("\\begin{table*}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Exercise-Specific Counterfactual Performance: LG (SHAP pruned) vs M-CELS}")
    latex.append("\\label{tab:exercise_specific}")
    latex.append("\\begin{tabular}{llcccccc}")
    latex.append("\\toprule")
    latex.append("Exercise & Method & Success & Confidence & Groups & Temp. Grad. & Time & L2 \\\\")
    latex.append("         &        & Rate (\\%) & (\\%) & Changed & & (s) & Distance \\\\")
    latex.append("\\midrule")
    
    for exercise in exercises:
        exercise_df = summary_df[summary_df['exercise'] == exercise]
        
        for idx, (_, row) in enumerate(exercise_df.iterrows()):
            if idx == 0:
                ex_name = exercise
            else:
                ex_name = ""
            
            latex.append(
                f"{ex_name} & {row['method']} & "
                f"{row['success_rate']:.1f} & "
                f"{row['confidence_mean']:.1f} $\\pm$ {row['confidence_std']:.1f} & "
                f"{row['modality_groups_mean']:.1f} $\\pm$ {row['modality_groups_std']:.1f} & "
                f"{row['temporal_grad_mean']:.4f} & "
                f"{row['time_mean']:.1f} $\\pm$ {row['time_std']:.1f} & "
                f"{row['l2_mean']:.1f} $\\pm$ {row['l2_std']:.1f} \\\\"
            )
        
        if exercise != exercises[-1]:
            latex.append("\\midrule")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table*}")
    
    latex_str = "\n".join(latex)
    latex_path = output_dir / 'exercise_specific_summary.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_str)
    
    print(f"✓ Saved LaTeX table: {latex_path}")
    print("\nLaTeX Table:")
    print(latex_str)
    
    return summary_df


def generate_modality_activation_heatmap(lg_data, mcels_data, output_dir):
    """Generate heatmap comparing modality group activation frequency."""
    
    exercises = ['Squat', 'Knee Extension', 'Gait']
    
    # Compute activation frequencies
    lg_activations = {}
    mcels_activations = {}
    
    for exercise in exercises:
        lg_activations[exercise] = compute_modality_activation_frequency(lg_data, exercise)
        mcels_activations[exercise] = compute_modality_activation_frequency(mcels_data, exercise)
    
    # Create dataframes for heatmap
    lg_df = pd.DataFrame(lg_activations).T
    mcels_df = pd.DataFrame(mcels_activations).T
    
    # Save activation frequencies
    lg_df.to_csv(output_dir / 'lg_modality_activation_frequency.csv')
    mcels_df.to_csv(output_dir / 'mcels_modality_activation_frequency.csv')
    print(f"✓ Saved activation frequency CSVs")
    
    # Create side-by-side heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    
    # LG heatmap
    sns.heatmap(lg_df, annot=True, fmt='.0f', cmap='YlOrRd', 
                vmin=0, vmax=100, cbar_kws={'label': 'Activation (%)'},
                ax=axes[0], linewidths=0.5)
    axes[0].set_title('LG (SHAP pruned): Modality Group Activation', fontweight='bold')
    axes[0].set_xlabel('Modality Group', fontweight='bold')
    axes[0].set_ylabel('Exercise', fontweight='bold')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
    
    # M-CELS heatmap
    sns.heatmap(mcels_df, annot=True, fmt='.0f', cmap='YlOrRd',
                vmin=0, vmax=100, cbar_kws={'label': 'Activation (%)'},
                ax=axes[1], linewidths=0.5)
    axes[1].set_title('M-CELS: Modality Group Activation', fontweight='bold')
    axes[1].set_xlabel('Modality Group', fontweight='bold')
    axes[1].set_ylabel('')
    axes[1].set_yticklabels([])
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    heatmap_path = output_dir / 'exercise_modality_activation_heatmap.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight') 
    print(f"✓ Saved activation heatmap: {heatmap_path}")
    plt.close()
    
    # Generate difference heatmap (LG - M-CELS)
    fig, ax = plt.subplots(figsize=(12, 4))
    diff_df = lg_df - mcels_df
    
    sns.heatmap(diff_df, annot=True, fmt='.0f', cmap='RdBu_r', center=0,
                vmin=-50, vmax=50, cbar_kws={'label': 'Difference (LG - M-CELS) %'},
                ax=ax, linewidths=0.5)
    ax.set_title('Modality Group Activation Difference: LG vs M-CELS', fontweight='bold')
    ax.set_xlabel('Modality Group', fontweight='bold')
    ax.set_ylabel('Exercise', fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    diff_path = output_dir / 'exercise_modality_activation_difference.png'
    plt.savefig(diff_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved difference heatmap: {diff_path}")
    plt.close()
    
    return lg_df, mcels_df, diff_df


def generate_qualitative_summary(lg_activations, mcels_activations, output_dir):
    """Generate qualitative biomechanical interpretation for each exercise."""
    
    exercises = ['Squat', 'Knee Extension', 'Gait']
    
    summaries = []
    
    # Get top activated groups per exercise
    for exercise in exercises:
        lg_freq = lg_activations.loc[exercise].sort_values(ascending=False)
        top_lg = lg_freq[lg_freq > 20].index.tolist()[:6]  # Top groups with >20% activation
        
        summaries.append(f"\n{'='*60}")
        summaries.append(f"{exercise.upper()}")
        summaries.append(f"{'='*60}")
        summaries.append(f"\nMost frequently modified groups (LG):")
        summaries.append(f"{', '.join(top_lg)}")
        
        # Exercise-specific interpretation
        if exercise == 'Squat':
            summaries.append("\nBiomechanical Interpretation:")
            summaries.append("Squat corrections emphasize bilateral quadriceps (RF) and hamstrings (Ham)")
            summaries.append("for symmetric knee flexion control and rotational stability. Modifications")
            summaries.append("to both accelerometer and gyroscope signals indicate corrections to both")
            summaries.append("linear and angular motion patterns, ensuring balanced weight distribution")
            summaries.append("during descent and ascent phases.")
            
        elif exercise == 'Knee Extension':
            summaries.append("\nBiomechanical Interpretation:")
            summaries.append("Knee extension corrections highlight injured-side quadriceps (RF) engagement")
            summaries.append("and reduced hamstring co-contraction to address range-of-motion deficits.")
            summaries.append("Tibialis anterior (TA) and gastrocnemius (Gast) modifications suggest")
            summaries.append("adjustments to ankle stabilization during the extension phase, preventing")
            summaries.append("compensatory movements that limit full knee extension.")
            
        else:  # Gait
            summaries.append("\nBiomechanical Interpretation:")
            summaries.append("Gait corrections involve coordinated quadriceps-hamstrings control during")
            summaries.append("stance phase and tibialis anterior-gastrocnemius activation for swing")
            summaries.append("clearance and push-off. High activation across all bilateral muscle groups")
            summaries.append("reflects the complex inter-limb coordination required for normal gait")
            summaries.append("kinematics, particularly in correcting asymmetric loading patterns.")
    
    summary_text = "\n".join(summaries)
    
    summary_path = output_dir / 'exercise_qualitative_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    
    print(f"✓ Saved qualitative summary: {summary_path}")
    print(summary_text)
    
    return summary_text


def generate_per_exercise_plots(summary_df, output_dir):
    """Generate per-exercise comparison plots."""
    
    exercises = ['Squat', 'Knee Extension', 'Gait']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Exercise-Specific Performance: LG vs M-CELS', 
                 fontsize=14, fontweight='bold')
    
    metrics = [
        ('success_rate', 'Success Rate (%)', axes[0]),
        ('modality_groups_mean', 'Modality Groups Changed', axes[1]),
        ('time_mean', 'Generation Time (s)', axes[2])
    ]
    
    for metric, ylabel, ax in metrics:
        x_pos = np.arange(len(exercises))
        width = 0.35
        
        lg_vals = []
        mcels_vals = []
        lg_stds = []
        mcels_stds = []
        
        for exercise in exercises:
            exercise_df = summary_df[summary_df['exercise'] == exercise]
            lg_row = exercise_df[exercise_df['method'] == 'LG (SHAP pruned)'].iloc[0]
            mcels_row = exercise_df[exercise_df['method'] == 'M-CELS'].iloc[0]
            
            lg_vals.append(lg_row[metric])
            mcels_vals.append(mcels_row[metric])
            
            if metric != 'success_rate':
                lg_stds.append(lg_row[metric.replace('_mean', '_std')])
                mcels_stds.append(mcels_row[metric.replace('_mean', '_std')])
            else:
                lg_stds.append(0)
                mcels_stds.append(0)
        
        bars1 = ax.bar(x_pos - width/2, lg_vals, width, yerr=lg_stds if any(lg_stds) else None,
                       capsize=5, label='LG (SHAP pruned)', 
                       color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=1.2)
        bars2 = ax.bar(x_pos + width/2, mcels_vals, width, yerr=mcels_stds if any(mcels_stds) else None,
                       capsize=5, label='M-CELS',
                       color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.2)
        
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(exercises, rotation=20, ha='right')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = output_dir / 'exercise_comparison_metrics.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved exercise comparison plot: {plot_path}")
    plt.close()


def main():
    """Main analysis workflow."""
    print("\n" + "="*70)
    print("EXERCISE-SPECIFIC COUNTERFACTUAL ANALYSIS")
    print("="*70 + "\n")
    
    # Paths
    lg_path = Path("results/evaluation/learnable_gate_subject_split/fcn_imu_learnable_gate_subject_split_evaluation.csv")
    mcels_path = Path("results/evaluation/mcels_subject_split/fcn_imu_mcels_subject_split_evaluation.csv")
    output_dir = Path("results/experiments/exercise_specific")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading evaluation data...")
    lg_data = load_and_filter_data(lg_path, "LG (SHAP pruned)")
    mcels_data = load_and_filter_data(mcels_path, "M-CELS")
    
    print(f"✓ Loaded LG data: {len(lg_data)} error-to-correct samples")
    print(f"✓ Loaded M-CELS data: {len(mcels_data)} error-to-correct samples")
    
    # Generate summary table
    print("\n" + "-"*70)
    print("1. Generating exercise summary table...")
    print("-"*70)
    summary_df = generate_exercise_summary_table(lg_data, mcels_data, output_dir)
    
    # Generate activation heatmaps
    print("\n" + "-"*70)
    print("2. Generating modality activation heatmaps...")
    print("-"*70)
    lg_activations, mcels_activations, diff_activations = generate_modality_activation_heatmap(
        lg_data, mcels_data, output_dir
    )
    
    # Generate per-exercise plots
    print("\n" + "-"*70)
    print("3. Generating per-exercise comparison plots...")
    print("-"*70)
    generate_per_exercise_plots(summary_df, output_dir)
    
    # Generate qualitative summary
    print("\n" + "-"*70)
    print("4. Generating qualitative biomechanical summary...")
    print("-"*70)
    generate_qualitative_summary(lg_activations, mcels_activations, output_dir)
    
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE")
    print(f"  Output directory: {output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
