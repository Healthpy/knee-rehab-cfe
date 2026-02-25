"""
Clinician-Level Visualization

Detailed, technical visualizations for clinicians and researchers.
Shows full multi-channel data, counterfactual heatmaps, and validation tools.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List
from matplotlib.gridspec import GridSpec


class ClinicianVisualizer:
    """
    Comprehensive visualization for clinicians
    
    Features:
    - Multi-channel time series
    - Counterfactual difference heatmap
    - Channel-wise analysis
    - Model validation tools
    """
    
    def __init__(self):
        # Channel names for 8 Delsys Trigno Avanti sensors (56 channels total)
        # Each sensor: 1 EMG + 6 IMU (3 accel + 3 gyro)
        sensor_muscles = [
            'R_RectFem', 'R_Hamstring', 'R_TibAnt', 'R_Gastro',
            'L_RectFem', 'L_Hamstring', 'L_TibAnt', 'L_Gastro'
        ]
        
        self.channel_names = []
        for muscle in sensor_muscles:
            self.channel_names.append(f'{muscle}_EMG')
            self.channel_names.append(f'{muscle}_Accel_X')
            self.channel_names.append(f'{muscle}_Accel_Y')
            self.channel_names.append(f'{muscle}_Accel_Z')
            self.channel_names.append(f'{muscle}_Gyro_X')
            self.channel_names.append(f'{muscle}_Gyro_Y')
            self.channel_names.append(f'{muscle}_Gyro_Z')
    
    def show_full_comparison(
        self,
        X_patient: np.ndarray,
        X_counterfactual: np.ndarray,
        channel_groups: Optional[Dict[str, List[int]]] = None,
        save_path: Optional[str] = None
    ):
        """
        Show all channels with patient vs counterfactual
        
        Args:
            X_patient: Patient data [T, C]
            X_counterfactual: Counterfactual data [T, C]
            channel_groups: Dictionary grouping channels
            save_path: Save path
        """
        if channel_groups is None:
            channel_groups = {
                'Thigh IMU': list(range(0, 6)),
                'Shank IMU': list(range(6, 12)),
                'EMG': list(range(12, 20))
            }
        
        n_groups = len(channel_groups)
        fig, axes = plt.subplots(n_groups, 1, figsize=(14, 4 * n_groups))
        
        if n_groups == 1:
            axes = [axes]
        
        time_percent = np.linspace(0, 100, len(X_patient))
        
        for ax, (group_name, channel_indices) in zip(axes, channel_groups.items()):
            for idx in channel_indices:
                if idx < X_patient.shape[1]:
                    # Patient
                    ax.plot(time_percent, X_patient[:, idx],
                           alpha=0.5, linewidth=1, label=f'{self.channel_names[idx]} (Patient)')
                    # Counterfactual
                    ax.plot(time_percent, X_counterfactual[:, idx],
                           alpha=0.5, linewidth=1, linestyle='--',
                           label=f'{self.channel_names[idx]} (CF)')
            
            ax.set_title(f'{group_name} Channels', fontsize=12, fontweight='bold')
            ax.set_ylabel('Signal Value', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, ncol=2, loc='upper right')
            ax.set_xlim(0, 100)
        
        axes[-1].set_xlabel('Movement Progress (%)', fontsize=11, fontweight='bold')
        
        plt.suptitle('Full Multi-Channel Comparison', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def show_difference_heatmap(
        self,
        X_patient: np.ndarray,
        X_counterfactual: np.ndarray,
        channel_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Heatmap of Δ(t, channel) showing where and when changes are needed
        
        Args:
            X_patient: Patient data [T, C]
            X_counterfactual: Counterfactual data [T, C]
            channel_names: Optional custom channel names
            save_path: Save path
        """
        delta = X_counterfactual - X_patient
        
        if channel_names is None:
            channel_names = [self.channel_names[i] for i in range(delta.shape[1])]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                        gridspec_kw={'height_ratios': [3, 1]})
        
        # Main heatmap
        im = ax1.imshow(
            delta.T,
            aspect='auto',
            cmap='RdBu_r',
            vmin=-np.abs(delta).max(),
            vmax=np.abs(delta).max(),
            interpolation='bilinear'
        )
        
        # Time axis
        time_ticks = np.linspace(0, delta.shape[0]-1, 6).astype(int)
        time_labels = [f"{i:.0f}%" for i in np.linspace(0, 100, 6)]
        ax1.set_xticks(time_ticks)
        ax1.set_xticklabels(time_labels)
        
        # Channel axis
        ax1.set_yticks(range(len(channel_names)))
        ax1.set_yticklabels(channel_names, fontsize=8)
        
        ax1.set_xlabel('Movement Progress (%)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Channel', fontsize=11, fontweight='bold')
        ax1.set_title('Counterfactual Difference Heatmap: Δ(time, channel)',
                     fontsize=13, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Change Magnitude', fontsize=10)
        
        # Temporal summary
        temporal_magnitude = np.sqrt(np.mean(delta**2, axis=1))
        ax2.fill_between(
            range(len(temporal_magnitude)),
            temporal_magnitude,
            alpha=0.6,
            color='steelblue'
        )
        ax2.set_xlim(0, len(delta)-1)
        ax2.set_xticks(time_ticks)
        ax2.set_xticklabels(time_labels)
        ax2.set_xlabel('Movement Progress (%)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Overall\nChange', fontsize=10)
        ax2.set_title('Temporal Change Profile', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def show_channel_importance(
        self,
        delta: np.ndarray,
        channel_names: Optional[List[str]] = None,
        top_k: int = 10,
        save_path: Optional[str] = None
    ):
        """
        Bar chart showing which channels changed most
        
        Args:
            delta: Difference tensor [T, C]
            channel_names: Channel names
            top_k: Number of top channels to show
            save_path: Save path
        """
        # Compute channel-wise magnitude
        channel_magnitude = np.sqrt(np.mean(delta**2, axis=0))
        
        if channel_names is None:
            channel_names = [self.channel_names[i] for i in range(len(channel_magnitude))]
        
        # Sort and select top k
        sorted_indices = np.argsort(channel_magnitude)[::-1][:top_k]
        top_magnitudes = channel_magnitude[sorted_indices]
        top_names = [channel_names[i] for i in sorted_indices]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.barh(range(len(top_names)), top_magnitudes, color='steelblue', alpha=0.7)
        
        # Color code by magnitude
        for i, bar in enumerate(bars):
            if top_magnitudes[i] > np.mean(top_magnitudes) * 1.5:
                bar.set_color('firebrick')
            elif top_magnitudes[i] > np.mean(top_magnitudes):
                bar.set_color('orange')
        
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names, fontsize=10)
        ax.set_xlabel('RMS Change Magnitude', fontsize=11, fontweight='bold')
        ax.set_title(f'Top {top_k} Changed Channels', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def show_phase_analysis(
        self,
        X_patient: np.ndarray,
        X_counterfactual: np.ndarray,
        n_phases: int = 4,
        save_path: Optional[str] = None
    ):
        """
        Analyze changes by movement phase
        
        Args:
            X_patient: Patient data [T, C]
            X_counterfactual: Counterfactual data [T, C]
            n_phases: Number of phases to divide movement into
            save_path: Save path
        """
        delta = X_counterfactual - X_patient
        T = len(delta)
        phase_length = T // n_phases
        
        # Compute phase-wise statistics
        phase_magnitudes = []
        phase_labels = []
        
        for i in range(n_phases):
            start = i * phase_length
            end = start + phase_length if i < n_phases - 1 else T
            phase_delta = delta[start:end, :]
            
            magnitude = np.sqrt(np.mean(phase_delta**2))
            phase_magnitudes.append(magnitude)
            phase_labels.append(f"{i*100//n_phases}-{(i+1)*100//n_phases}%")
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Phase magnitudes
        ax1.bar(range(n_phases), phase_magnitudes, color='steelblue', alpha=0.7)
        ax1.set_xticks(range(n_phases))
        ax1.set_xticklabels(phase_labels)
        ax1.set_xlabel('Movement Phase', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Change Magnitude', fontsize=11, fontweight='bold')
        ax1.set_title('Change Magnitude by Phase', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Phase-wise channel breakdown
        phase_channel_data = []
        for i in range(n_phases):
            start = i * phase_length
            end = start + phase_length if i < n_phases - 1 else T
            phase_delta = delta[start:end, :]
            channel_mag = np.sqrt(np.mean(phase_delta**2, axis=0))
            phase_channel_data.append(channel_mag)
        
        phase_channel_array = np.array(phase_channel_data).T
        
        im = ax2.imshow(phase_channel_array[:10, :], aspect='auto', cmap='YlOrRd')
        ax2.set_xticks(range(n_phases))
        ax2.set_xticklabels(phase_labels)
        ax2.set_yticks(range(10))
        ax2.set_yticklabels([self.channel_names[i] for i in range(10)], fontsize=9)
        ax2.set_xlabel('Movement Phase', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Channel (Top 10)', fontsize=11, fontweight='bold')
        ax2.set_title('Channel × Phase Change Map', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax2, label='Magnitude')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_comprehensive_report(
        self,
        X_patient: np.ndarray,
        X_counterfactual: np.ndarray,
        explanation: Dict,
        save_dir: str
    ):
        """
        Generate complete clinician report
        
        Args:
            X_patient: Patient data [T, C]
            X_counterfactual: Counterfactual data [T, C]
            explanation: Explanation metadata
            save_dir: Directory to save figures
        """
        delta = X_counterfactual - X_patient
        
        # 1. Full comparison
        self.show_full_comparison(
            X_patient, X_counterfactual,
            save_path=f"{save_dir}/full_comparison.png"
        )
        
        # 2. Difference heatmap
        self.show_difference_heatmap(
            X_patient, X_counterfactual,
            save_path=f"{save_dir}/difference_heatmap.png"
        )
        
        # 3. Channel importance
        self.show_channel_importance(
            delta,
            save_path=f"{save_dir}/channel_importance.png"
        )
        
        # 4. Phase analysis
        self.show_phase_analysis(
            X_patient, X_counterfactual,
            save_path=f"{save_dir}/phase_analysis.png"
        )
        
        print(f"Comprehensive clinician report saved to {save_dir}")


if __name__ == "__main__":
    # Test visualization
    viz = ClinicianVisualizer()
    
    # Simulate data
    T, C = 100, 20
    X_patient = np.random.randn(T, C)
    X_counterfactual = X_patient + np.random.randn(T, C) * 0.3
    
    fig = viz.show_difference_heatmap(X_patient, X_counterfactual)
    plt.show()
    
    print("Clinician visualizer test complete")
