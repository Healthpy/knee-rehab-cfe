"""
Patient-Level Visualization

Minimal, intuitive visualizations for patients.
Goal: Understand what to change without overwhelming detail.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Optional, Dict, List, Tuple
import seaborn as sns


class PatientVisualizer:
    """
    Simple, patient-friendly visualizations
    
    Shows:
    - You vs Correct overlays
    - Highlighted correction zones
    - Key body parts only (knee, quad)
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Args:
            style: Matplotlib style
        """
        plt.style.use('default')
        sns.set_palette("husl")
        self.colors = {
            'patient': '#e74c3c',  # Red
            'correct': '#2ecc71',  # Green
            'highlight': '#f39c12',  # Yellow/Orange
            'background': '#ecf0f1'  # Light gray
        }
    
    def show_comparison(
        self,
        X_patient: np.ndarray,
        X_correct: np.ndarray,
        body_part: str = 'knee',
        highlight_phases: Optional[List[Tuple[float, float]]] = None,
        save_path: Optional[str] = None
    ):
        """
        Show patient vs correct execution with highlighted correction zones
        
        Args:
            X_patient: Patient execution [T, C]
            X_correct: Correct/counterfactual execution [T, C]
            body_part: Which body part to show ('knee', 'quad', 'hamstring')
            highlight_phases: List of (start%, end%) to highlight
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get channel data
        time_percent = np.linspace(0, 100, len(X_patient))
        patient_data, correct_data, ylabel = self._extract_body_part_data(
            X_patient, X_correct, body_part
        )
        
        # Plot comparison
        ax.plot(
            time_percent, patient_data,
            color=self.colors['patient'],
            linewidth=3,
            label='Your Execution',
            alpha=0.8
        )
        ax.plot(
            time_percent, correct_data,
            color=self.colors['correct'],
            linewidth=3,
            label='Correct Pattern',
            linestyle='--',
            alpha=0.8
        )
        
        # Highlight correction zones
        if highlight_phases:
            for phase_start, phase_end in highlight_phases:
                ax.axvspan(
                    phase_start, phase_end,
                    alpha=0.2,
                    color=self.colors['highlight'],
                    label='Change Here' if phase_start == highlight_phases[0][0] else ''
                )
        
        # Styling
        ax.set_xlabel('Movement Progress (%)', fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax.set_title(
            f'{body_part.capitalize()} Movement Pattern',
            fontsize=16,
            fontweight='bold',
            pad=20
        )
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        
        # Add annotation for first highlight
        if highlight_phases:
            first_phase = highlight_phases[0]
            mid_phase = (first_phase[0] + first_phase[1]) / 2
            ax.annotate(
                'Focus on\nthis phase',
                xy=(mid_phase, ax.get_ylim()[1] * 0.9),
                xytext=(mid_phase, ax.get_ylim()[1] * 0.95),
                ha='center',
                fontsize=11,
                bbox=dict(boxstyle='round,pad=0.5', facecolor=self.colors['highlight'], alpha=0.7),
                arrowprops=dict(arrowstyle='->', color=self.colors['highlight'], lw=2)
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def show_simple_difference(
        self,
        X_patient: np.ndarray,
        X_correct: np.ndarray,
        body_parts: List[str] = ['knee', 'quad'],
        highlight_phases: Optional[List[Tuple[float, float]]] = None,
        save_path: Optional[str] = None
    ):
        """
        Show multiple body parts in subplots
        
        Args:
            X_patient: Patient execution [T, C]
            X_correct: Correct execution [T, C]
            body_parts: List of body parts to visualize
            highlight_phases: Correction phases
            save_path: Save path
        """
        n_parts = len(body_parts)
        fig, axes = plt.subplots(n_parts, 1, figsize=(12, 4 * n_parts))
        
        if n_parts == 1:
            axes = [axes]
        
        time_percent = np.linspace(0, 100, len(X_patient))
        
        for ax, body_part in zip(axes, body_parts):
            patient_data, correct_data, ylabel = self._extract_body_part_data(
                X_patient, X_correct, body_part
            )
            
            # Plot
            ax.plot(time_percent, patient_data, color=self.colors['patient'],
                   linewidth=2.5, label='You', alpha=0.8)
            ax.plot(time_percent, correct_data, color=self.colors['correct'],
                   linewidth=2.5, label='Target', linestyle='--', alpha=0.8)
            
            # Highlight zones
            if highlight_phases:
                for phase_start, phase_end in highlight_phases:
                    ax.axvspan(phase_start, phase_end, alpha=0.15,
                              color=self.colors['highlight'])
            
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.set_title(body_part.capitalize(), fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 100)
        
        axes[-1].set_xlabel('Movement Progress (%)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def show_instruction_card(
        self,
        instructions: List[str],
        summary: str,
        body_parts: List[str],
        save_path: Optional[str] = None
    ):
        """
        Create a simple instruction card for the patient
        
        Args:
            instructions: List of instruction strings
            summary: Overall summary
            body_parts: Affected body parts
            save_path: Save path
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        
        # Title
        title_text = "Your Personalized Correction"
        ax.text(0.5, 0.95, title_text, ha='center', va='top',
               fontsize=20, fontweight='bold', transform=ax.transAxes)
        
        # Summary
        ax.text(0.5, 0.85, summary, ha='center', va='top',
               fontsize=14, transform=ax.transAxes,
               bbox=dict(boxstyle='round,pad=0.8', facecolor=self.colors['highlight'], alpha=0.3))
        
        # Body parts affected
        body_parts_str = ", ".join(body_parts)
        ax.text(0.5, 0.75, f"Focus on: {body_parts_str}", ha='center', va='top',
               fontsize=12, style='italic', transform=ax.transAxes)
        
        # Instructions
        y_pos = 0.65
        for i, instruction in enumerate(instructions, 1):
            ax.text(0.1, y_pos, f"{i}.", ha='left', va='top',
                   fontsize=14, fontweight='bold', transform=ax.transAxes)
            ax.text(0.15, y_pos, instruction, ha='left', va='top',
                   fontsize=12, transform=ax.transAxes, wrap=True)
            y_pos -= 0.15
        
        # Footer
        ax.text(0.5, 0.05, "Practice these corrections slowly and focus on quality",
               ha='center', va='bottom', fontsize=10, style='italic',
               transform=ax.transAxes, color='gray')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _extract_body_part_data(
        self,
        X_patient: np.ndarray,
        X_correct: np.ndarray,
        body_part: str
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Extract relevant channel data for body part
        
        Channel mapping (8 sensors × 7 channels each = 56 total):
        - Sensor i: EMG at index i*7, IMU at i*7+1 to i*7+6
        - IMU layout: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
        
        Sensor placements (0-7):
        0: R Rectus Femoris, 1: R Hamstrings, 2: R Tibialis Anterior, 3: R Gastrocnemius
        4: L Rectus Femoris, 5: L Hamstrings, 6: L Tibialis Anterior, 7: L Gastrocnemius
        
        Args:
            X_patient: Patient data [T, C]
            X_correct: Correct data [T, C]
            body_part: Body part name
            
        Returns:
            patient_data, correct_data, ylabel
        """
        if body_part == 'knee':
            # Knee angle proxy (integrate right thigh gyro Y)
            # Right rectus femoris gyro Y is at index 0*7 + 5 = 5
            patient_data = np.cumsum(X_patient[:, 5])
            correct_data = np.cumsum(X_correct[:, 5])
            ylabel = 'Knee Angle Proxy (degrees)'
        
        elif body_part == 'quad':
            # Right rectus femoris EMG at index 0
            patient_data = X_patient[:, 0]
            correct_data = X_correct[:, 0]
            ylabel = 'Quadriceps Activation'
        
        elif body_part == 'hamstring':
            # Right hamstrings EMG at index 7 (sensor 1 * 7)
            patient_data = X_patient[:, 7]
            correct_data = X_correct[:, 7]
            ylabel = 'Hamstring Activation'
        
        elif body_part == 'angular_velocity':
            # Right rectus femoris gyro xyz at indices 4, 5, 6
            gyro_indices = [4, 5, 6]
            patient_data = np.linalg.norm(X_patient[:, gyro_indices], axis=1)
            correct_data = np.linalg.norm(X_correct[:, gyro_indices], axis=1)
            ylabel = 'Movement Speed (deg/s)'
        
        else:
            # Default to first channel
            patient_data = X_patient[:, 0]
            correct_data = X_correct[:, 0]
            ylabel = 'Signal Value'
        
        return patient_data, correct_data, ylabel


def create_patient_report(
    X_patient: np.ndarray,
    X_correct: np.ndarray,
    explanation: Dict,
    save_dir: str
):
    """
    Generate a complete patient report with multiple visualizations
    
    Args:
        X_patient: Patient execution [T, C]
        X_correct: Counterfactual [T, C]
        explanation: Explanation dictionary
        save_dir: Directory to save figures
    """
    viz = PatientVisualizer()
    
    # Extract phases
    highlight_phases = [
        (phase[0], phase[1])
        for phase in explanation.get('affected_phases', [])[:2]
    ]
    
    # 1. Knee comparison
    viz.show_comparison(
        X_patient, X_correct,
        body_part='knee',
        highlight_phases=highlight_phases,
        save_path=f"{save_dir}/knee_comparison.png"
    )
    
    # 2. Multi-part view
    viz.show_simple_difference(
        X_patient, X_correct,
        body_parts=['knee', 'quad', 'hamstring'],
        highlight_phases=highlight_phases,
        save_path=f"{save_dir}/multi_part_view.png"
    )
    
    # 3. Instruction card (placeholder)
    instructions = [
        "At 50% of movement: Lower your body more",
        "At 65% of movement: Activate quadriceps strongly"
    ]
    viz.show_instruction_card(
        instructions=instructions,
        summary="2 corrections needed for proper form",
        body_parts=['Knee', 'Quadriceps'],
        save_path=f"{save_dir}/instruction_card.png"
    )
    
    print(f"Patient report saved to {save_dir}")


if __name__ == "__main__":
    # Test visualization
    viz = PatientVisualizer()
    
    # Simulate data
    T = 100
    X_patient = np.random.randn(T, 20)
    X_correct = X_patient + np.random.randn(T, 20) * 0.3
    
    fig = viz.show_comparison(
        X_patient, X_correct,
        body_part='knee',
        highlight_phases=[(40, 70)]
    )
    plt.show()
    
    print("Patient visualizer test complete")
