#!/usr/bin/env python3
"""
Visualization script for Adaptive Multi-Objective Counterfactual Explanations.

This script creates comprehensive visualizations of:
1. Original vs Counterfactual signals
2. Saliency masks
3. Channel-wise comparisons
4. Group-level analysis
5. Statistical summaries

Author: XAI Analysis Team
Date: November 6, 2025
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional
import argparse

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CounterfactualVisualizer:
    """Visualize counterfactual explanations from adaptive multi-objective explainer."""
    
    def __init__(self, results_path: str):
        """Initialize visualizer with results path."""
        self.results_path = Path(results_path)
        self.results = self._load_results()
        
        # Sensor configuration (48 channels: 8 sensors × 6 channels each)
        self.sensor_names = ['R_RF', 'R_HAM', 'R_TA', 'R_GAS', 'L_RF', 'L_HAM', 'L_TA', 'L_GAS']
        self.channel_types = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
        
        # Create modality groups
        self.modality_groups = {
            'R_RF_acc': list(range(0, 3)),    # Channels 0-2
            'R_RF_gyr': list(range(3, 6)),    # Channels 3-5
            'R_HAM_acc': list(range(6, 9)),    # Channels 6-8
            'R_HAM_gyr': list(range(9, 12)),   # Channels 9-11
            'R_TA_acc': list(range(12, 15)),   # Channels 12-14
            'R_TA_gyr': list(range(15, 18)),   # Channels 15-17
            'R_GAS_acc': list(range(18, 21)),  # Channels 18-20
            'R_GAS_gyr': list(range(21, 24)),  # Channels 21-23
            'L_RF_acc': list(range(24, 27)),   # Channels 24-26
            'L_RF_gyr': list(range(27, 30)),   # Channels 27-29
            'L_HAM_acc': list(range(30, 33)),  # Channels 30-32
            'L_HAM_gyr': list(range(33, 36)),  # Channels 33-35
            'L_TA_acc': list(range(36, 39)),   # Channels 36-38
            'L_TA_gyr': list(range(39, 42)),   # Channels 39-41
            'L_GAS_acc': list(range(42, 45)),  # Channels 42-44
            'L_GAS_gyr': list(range(45, 48))   # Channels 45-47
        }
        
    def _load_results(self) -> Dict:
        """Load results from JSON file."""
        results_file = self.results_path / 'explainer_results.json'
        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def plot_signal_comparison(self, save_path: Optional[str] = None, sample_channels: int = 16) -> None:
        """Plot original vs counterfactual signals."""
        original = np.array(self.results['original_input'])
        counterfactual = np.array(self.results['counterfactual'])
        mask = np.array(self.results['mask'])
        
        # Sample channels for visualization (too many to show all 48)
        n_channels, n_timesteps = original.shape
        channel_indices = np.linspace(0, n_channels-1, sample_channels, dtype=int)
        
        fig, axes = plt.subplots(sample_channels, 1, figsize=(15, 2*sample_channels))
        if sample_channels == 1:
            axes = [axes]
        
        for i, ch_idx in enumerate(channel_indices):
            ax = axes[i]
            
            # Plot signals
            time_steps = np.arange(n_timesteps)
            ax.plot(time_steps, original[ch_idx], 'b-', label='Original', alpha=0.7, linewidth=1.5)
            ax.plot(time_steps, counterfactual[ch_idx], 'r-', label='Counterfactual', alpha=0.8, linewidth=1.5)
            
            # Highlight modified regions with mask
            modified_regions = mask[ch_idx] > 0.5
            if np.any(modified_regions):
                ax.fill_between(time_steps, 
                               ax.get_ylim()[0], ax.get_ylim()[1], 
                               where=modified_regions, alpha=0.2, color='orange',
                               label='Modified Region')
            
            # Get sensor info
            sensor_idx = ch_idx // 6
            channel_type_idx = ch_idx % 6
            sensor_name = self.sensor_names[sensor_idx] if sensor_idx < len(self.sensor_names) else f"Sensor_{sensor_idx}"
            channel_type = self.channel_types[channel_type_idx] if channel_type_idx < len(self.channel_types) else f"ch_{channel_type_idx}"
            
            ax.set_title(f'Channel {ch_idx}: {sensor_name}_{channel_type}', fontsize=10)
            ax.set_ylabel('Signal Value', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.legend(loc='upper right', fontsize=9)
            if i == len(channel_indices) - 1:
                ax.set_xlabel('Time Steps', fontsize=10)
        
        plt.suptitle('Original vs Counterfactual Signals (Sample Channels)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Signal comparison saved to: {save_path}")
        plt.show()
    
    def plot_mask_analysis(self, save_path: Optional[str] = None) -> None:
        """Plot detailed mask analysis."""
        mask = np.array(self.results['mask'])
        n_channels, n_timesteps = mask.shape
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Heatmap of entire mask
        ax1 = axes[0, 0]
        im1 = ax1.imshow(mask, aspect='auto', cmap='viridis', interpolation='nearest')
        ax1.set_title('Complete Mask Heatmap', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Channels')
        plt.colorbar(im1, ax=ax1, label='Mask Value')
        
        # 2. Channel-wise mask activation
        ax2 = axes[0, 1]
        channel_activation = np.mean(mask, axis=1)
        channels = np.arange(n_channels)
        bars = ax2.bar(channels, channel_activation, alpha=0.7, color='steelblue')
        ax2.set_title('Average Mask Activation per Channel', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Channel Index')
        ax2.set_ylabel('Average Activation')
        ax2.grid(True, alpha=0.3)
        
        # Highlight top activated channels
        top_channels = np.argsort(channel_activation)[-5:]
        for ch in top_channels:
            bars[ch].set_color('orange')
        
        # 3. Temporal mask activation
        ax3 = axes[1, 0]
        temporal_activation = np.mean(mask, axis=0)
        time_steps = np.arange(n_timesteps)
        ax3.plot(time_steps, temporal_activation, 'g-', linewidth=2)
        ax3.fill_between(time_steps, temporal_activation, alpha=0.3, color='green')
        ax3.set_title('Average Mask Activation over Time', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Average Activation')
        ax3.grid(True, alpha=0.3)
        
        # 4. Group-level activation
        ax4 = axes[1, 1]
        group_activations = {}
        for group_name, channel_indices in self.modality_groups.items():
            if all(ch < n_channels for ch in channel_indices):
                group_activations[group_name] = np.mean(mask[channel_indices, :])
        
        if group_activations:
            groups = list(group_activations.keys())
            activations = list(group_activations.values())
            
            bars = ax4.bar(range(len(groups)), activations, alpha=0.7, color='coral')
            ax4.set_title('Group-wise Mask Activation', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Sensor Groups')
            ax4.set_ylabel('Average Activation')
            ax4.set_xticks(range(len(groups)))
            ax4.set_xticklabels(groups, rotation=45, ha='right', fontsize=9)
            ax4.grid(True, alpha=0.3)
            
            # Highlight top groups
            if len(activations) > 0:
                top_groups = np.argsort(activations)[-3:]
                for idx in top_groups:
                    bars[idx].set_color('darkred')
        
        plt.suptitle('Mask Analysis Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Mask analysis saved to: {save_path}")
        plt.show()
    
    def plot_difference_analysis(self, save_path: Optional[str] = None) -> None:
        """Plot difference between original and counterfactual."""
        original = np.array(self.results['original_input'])
        counterfactual = np.array(self.results['counterfactual'])
        
        difference = counterfactual - original
        abs_difference = np.abs(difference)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Difference heatmap
        ax1 = axes[0, 0]
        im1 = ax1.imshow(difference, aspect='auto', cmap='RdBu_r', interpolation='nearest')
        ax1.set_title('Signal Difference Heatmap\n(Counterfactual - Original)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Channels')
        plt.colorbar(im1, ax=ax1, label='Difference')
        
        # 2. Absolute difference heatmap
        ax2 = axes[0, 1]
        im2 = ax2.imshow(abs_difference, aspect='auto', cmap='Reds', interpolation='nearest')
        ax2.set_title('Absolute Difference Heatmap', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Channels')
        plt.colorbar(im2, ax=ax2, label='|Difference|')
        
        # 3. Channel-wise difference distribution
        ax3 = axes[1, 0]
        channel_diff = np.mean(abs_difference, axis=1)
        channels = np.arange(len(channel_diff))
        bars = ax3.bar(channels, channel_diff, alpha=0.7, color='purple')
        ax3.set_title('Average Absolute Difference per Channel', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Channel Index')
        ax3.set_ylabel('Average |Difference|')
        ax3.grid(True, alpha=0.3)
        
        # Highlight channels with largest changes
        top_diff_channels = np.argsort(channel_diff)[-5:]
        for ch in top_diff_channels:
            bars[ch].set_color('red')
        
        # 4. Temporal difference distribution
        ax4 = axes[1, 1]
        temporal_diff = np.mean(abs_difference, axis=0)
        time_steps = np.arange(len(temporal_diff))
        ax4.plot(time_steps, temporal_diff, 'r-', linewidth=2)
        ax4.fill_between(time_steps, temporal_diff, alpha=0.3, color='red')
        ax4.set_title('Average Absolute Difference over Time', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('Average |Difference|')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Signal Difference Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Difference analysis saved to: {save_path}")
        plt.show()
    
    def plot_group_importance(self, save_path: Optional[str] = None) -> None:
        """Plot group importance and selection analysis."""
        if 'group_importance' not in self.results or 'selected_groups' not in self.results:
            print("Group importance data not available in results.")
            return
        
        group_importance = self.results['group_importance']
        selected_groups = self.results['selected_groups']
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 1. Group importance ranking
        ax1 = axes[0]
        groups = list(group_importance.keys())
        importance_scores = list(group_importance.values())
        
        # Sort by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        sorted_groups = [groups[i] for i in sorted_indices]
        sorted_scores = [importance_scores[i] for i in sorted_indices]
        
        # Color bars based on selection
        colors = ['darkgreen' if group in selected_groups else 'lightblue' for group in sorted_groups]
        
        bars = ax1.bar(range(len(sorted_groups)), sorted_scores, color=colors, alpha=0.8)
        ax1.set_title('Group Importance Scores (Shapley Values)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Sensor Groups')
        ax1.set_ylabel('Importance Score')
        ax1.set_xticks(range(len(sorted_groups)))
        ax1.set_xticklabels(sorted_groups, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='darkgreen', label='Selected for Optimization'),
                          Patch(facecolor='lightblue', label='Not Selected')]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # 2. Selected vs Available groups
        ax2 = axes[1]
        total_groups = len(groups)
        selected_count = len(selected_groups)
        not_selected_count = total_groups - selected_count
        
        sizes = [selected_count, not_selected_count]
        labels = [f'Selected\n({selected_count} groups)', f'Not Selected\n({not_selected_count} groups)']
        colors_pie = ['darkgreen', 'lightcoral']
        
        wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                                          startangle=90, textprops={'fontsize': 12})
        ax2.set_title('Group Selection Summary', fontsize=14, fontweight='bold')
        
        plt.suptitle('Group-Level Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Group importance analysis saved to: {save_path}")
        plt.show()
    
    def plot_optimization_trace(self, save_path: Optional[str] = None) -> None:
        """Plot optimization trace if available."""
        if 'optimization_trace' not in self.results:
            print("Optimization trace not available in results.")
            return
        
        trace = self.results['optimization_trace']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Loss history
        if 'loss_history' in trace:
            ax1 = axes[0, 0]
            iterations = range(len(trace['loss_history']))
            ax1.plot(iterations, trace['loss_history'], 'b-', linewidth=2)
            ax1.set_title('Total Loss During Optimization', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Loss Value')
            ax1.grid(True, alpha=0.3)
        
        # 2. Validity history
        if 'validity_history' in trace:
            ax2 = axes[0, 1]
            iterations = range(len(trace['validity_history']))
            ax2.plot(iterations, trace['validity_history'], 'g-', linewidth=2)
            ax2.axhline(y=0.7, color='r', linestyle='--', alpha=0.7, label='Min Threshold')
            ax2.set_title('Target Validity During Optimization', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Validity Score')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Sparsity history
        if 'sparsity_history' in trace:
            ax3 = axes[1, 0]
            iterations = range(len(trace['sparsity_history']))
            ax3.plot(iterations, trace['sparsity_history'], 'orange', linewidth=2)
            ax3.set_title('Sparsity During Optimization', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Sparsity Score')
            ax3.grid(True, alpha=0.3)
        
        # 4. Component losses
        if 'component_losses' in trace and trace['component_losses']:
            ax4 = axes[1, 1]
            component_data = trace['component_losses']
            
            # Extract component names and plot each
            if len(component_data) > 0:
                components = list(component_data[0].keys())
                for comp in components:
                    values = [step.get(comp, 0) for step in component_data]
                    ax4.plot(range(len(values)), values, label=comp, linewidth=1.5)
                
                ax4.set_title('Loss Components During Optimization', fontsize=12, fontweight='bold')
                ax4.set_xlabel('Iteration')
                ax4.set_ylabel('Component Loss')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Optimization Trace Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Optimization trace saved to: {save_path}")
        plt.show()
    
    def create_summary_report(self, save_path: Optional[str] = None) -> None:
        """Create a comprehensive summary report."""
        original = np.array(self.results['original_input'])
        counterfactual = np.array(self.results['counterfactual'])
        mask = np.array(self.results['mask'])
        
        # Calculate statistics
        mask_sparsity = 1.0 - (np.sum(mask > 0.5) / mask.size)
        l2_distance = np.linalg.norm(counterfactual - original)
        l1_distance = np.sum(np.abs(counterfactual - original))
        
        # Get evaluation metrics
        evaluation = self.results.get('evaluation', {})
        
        print("="*60)
        print("ADAPTIVE MULTI-OBJECTIVE COUNTERFACTUAL SUMMARY")
        print("="*60)
        print(f"Input Shape: {original.shape}")
        print(f"Target Class: {self.results.get('target_class', 'Unknown')}")
        print(f"Method: {self.results.get('method', 'Adaptive Multi-Objective')}")
        print()
        
        print("OPTIMIZATION RESULTS:")
        print(f"  Success: {evaluation.get('valid', 'Unknown')}")
        print(f"  Final Confidence: {evaluation.get('confidence', 'Unknown'):.3f}")
        print(f"  Predicted Class: {evaluation.get('predicted_class', 'Unknown')}")
        print()
        
        print("SPARSITY & DISTANCES:")
        print(f"  Mask Sparsity: {mask_sparsity:.3f}")
        print(f"  L1 Distance: {l1_distance:.3f}")
        print(f"  L2 Distance: {l2_distance:.3f}")
        print()
        
        if 'selected_groups' in self.results:
            print("GROUP SELECTION:")
            print(f"  Selected Groups: {len(self.results['selected_groups'])}")
            print(f"  Total Available: {len(self.results.get('group_importance', {}))}")
            print(f"  Selection Ratio: {len(self.results['selected_groups']) / max(1, len(self.results.get('group_importance', {}))):.1%}")
            print(f"  Groups: {', '.join(self.results['selected_groups'])}")
            print()
        
        if 'optimization_trace' in self.results:
            trace = self.results['optimization_trace']
            print("OPTIMIZATION DETAILS:")
            print(f"  Iterations: {trace.get('iterations', 'Unknown')}")
            print(f"  Final Validity: {trace.get('final_validity', 'Unknown'):.3f}")
            print()
        
        print("="*60)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write("ADAPTIVE MULTI-OBJECTIVE COUNTERFACTUAL SUMMARY\n")
                f.write("="*60 + "\n")
                f.write(f"Input Shape: {original.shape}\n")
                f.write(f"Target Class: {self.results.get('target_class', 'Unknown')}\n")
                f.write(f"Method: {self.results.get('method', 'Adaptive Multi-Objective')}\n\n")
                
                f.write("OPTIMIZATION RESULTS:\n")
                f.write(f"  Success: {evaluation.get('valid', 'Unknown')}\n")
                f.write(f"  Final Confidence: {evaluation.get('confidence', 'Unknown'):.3f}\n")
                f.write(f"  Predicted Class: {evaluation.get('predicted_class', 'Unknown')}\n\n")
                
                f.write("SPARSITY & DISTANCES:\n")
                f.write(f"  Mask Sparsity: {mask_sparsity:.3f}\n")
                f.write(f"  L1 Distance: {l1_distance:.3f}\n")
                f.write(f"  L2 Distance: {l2_distance:.3f}\n\n")
                
                if 'selected_groups' in self.results:
                    f.write("GROUP SELECTION:\n")
                    f.write(f"  Selected Groups: {len(self.results['selected_groups'])}\n")
                    f.write(f"  Total Available: {len(self.results.get('group_importance', {}))}\n")
                    f.write(f"  Selection Ratio: {len(self.results['selected_groups']) / max(1, len(self.results.get('group_importance', {}))):.1%}\n")
                    f.write(f"  Groups: {', '.join(self.results['selected_groups'])}\n\n")
            
            print(f"Summary report saved to: {save_path}")
    
    def create_all_visualizations(self, output_dir: Optional[str] = None) -> None:
        """Create all visualizations and save to output directory."""
        if output_dir is None:
            output_dir = self.results_path / 'visualizations'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        print("Creating comprehensive visualizations...")
        
        # 1. Signal comparison
        print("1. Plotting signal comparison...")
        self.plot_signal_comparison(output_dir / 'signal_comparison.png')
        
        # 2. Mask analysis
        print("2. Plotting mask analysis...")
        self.plot_mask_analysis(output_dir / 'mask_analysis.png')
        
        # 3. Difference analysis
        print("3. Plotting difference analysis...")
        self.plot_difference_analysis(output_dir / 'difference_analysis.png')
        
        # 4. Group importance (if available)
        print("4. Plotting group importance...")
        self.plot_group_importance(output_dir / 'group_importance.png')
        
        # 5. Optimization trace (if available)
        print("5. Plotting optimization trace...")
        self.plot_optimization_trace(output_dir / 'optimization_trace.png')
        
        # 6. Summary report
        print("6. Creating summary report...")
        self.create_summary_report(output_dir / 'summary_report.txt')
        
        print(f"\nAll visualizations saved to: {output_dir}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Visualize Adaptive Multi-Objective Counterfactual Explanations')
    parser.add_argument('results_path', type=str, help='Path to results directory')
    parser.add_argument('--output', type=str, help='Output directory for visualizations')
    parser.add_argument('--plot', type=str, choices=['all', 'signals', 'mask', 'difference', 'groups', 'trace'], 
                       default='all', help='Which plots to generate')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = CounterfactualVisualizer(args.results_path)
    
    # Generate plots based on selection
    if args.plot == 'all':
        visualizer.create_all_visualizations(args.output)
    elif args.plot == 'signals':
        visualizer.plot_signal_comparison()
    elif args.plot == 'mask':
        visualizer.plot_mask_analysis()
    elif args.plot == 'difference':
        visualizer.plot_difference_analysis()
    elif args.plot == 'groups':
        visualizer.plot_group_importance()
    elif args.plot == 'trace':
        visualizer.plot_optimization_trace()


if __name__ == '__main__':
    main()