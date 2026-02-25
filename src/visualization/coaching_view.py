"""
Coaching View for Real-Time AR/VR Feedback

Provides simple, actionable cues during movement execution.
Designed for AR glasses or mobile apps.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Arrow
from typing import Optional, List, Tuple, Dict
import matplotlib.patches as mpatches


class CoachingVisualizer:
    """
    Real-time coaching visualizations
    
    Features:
    - Progress bar with phase indicators
    - Directional cues (arrows, text)
    - Simple animated guidance
    """
    
    def __init__(self):
        self.colors = {
            'progress': '#3498db',  # Blue
            'cue': '#e74c3c',  # Red
            'success': '#2ecc71',  # Green
            'warning': '#f39c12'  # Orange
        }
    
    def show_progress_cue(
        self,
        current_progress: float,
        cue_phase: Tuple[float, float],
        instruction: str,
        save_path: Optional[str] = None
    ):
        """
        Show movement progress with correction cue
        
        Args:
            current_progress: Current % of movement (0-100)
            cue_phase: (start%, end%) where correction needed
            instruction: Text instruction
            save_path: Save path
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        
        # Progress bar
        bar_y = 0.7
        bar_height = 0.1
        bar_width = 0.8
        bar_x = 0.1
        
        # Background bar
        bg_rect = FancyBboxPatch(
            (bar_x, bar_y), bar_width, bar_height,
            boxstyle="round,pad=0.01",
            facecolor='lightgray',
            edgecolor='black',
            linewidth=2,
            transform=ax.transAxes
        )
        ax.add_patch(bg_rect)
        
        # Progress fill
        progress_width = bar_width * (current_progress / 100)
        progress_rect = FancyBboxPatch(
            (bar_x, bar_y), progress_width, bar_height,
            boxstyle="round,pad=0.01",
            facecolor=self.colors['progress'],
            alpha=0.7,
            transform=ax.transAxes
        )
        ax.add_patch(progress_rect)
        
        # Cue zone
        cue_start_x = bar_x + bar_width * (cue_phase[0] / 100)
        cue_width = bar_width * ((cue_phase[1] - cue_phase[0]) / 100)
        cue_rect = FancyBboxPatch(
            (cue_start_x, bar_y), cue_width, bar_height,
            boxstyle="round,pad=0.01",
            facecolor=self.colors['warning'],
            alpha=0.5,
            edgecolor=self.colors['cue'],
            linewidth=3,
            transform=ax.transAxes
        )
        ax.add_patch(cue_rect)
        
        # Current position indicator
        current_x = bar_x + bar_width * (current_progress / 100)
        circle = Circle(
            (current_x, bar_y + bar_height/2),
            0.02,
            facecolor=self.colors['success'],
            edgecolor='black',
            linewidth=2,
            transform=ax.transAxes,
            zorder=10
        )
        ax.add_patch(circle)
        
        # Progress text
        ax.text(
            0.5, bar_y + bar_height + 0.05,
            f"Movement Progress: {current_progress:.0f}%",
            ha='center', va='bottom',
            fontsize=16, fontweight='bold',
            transform=ax.transAxes
        )
        
        # Instruction box
        in_cue_zone = cue_phase[0] <= current_progress <= cue_phase[1]
        
        if in_cue_zone:
            # Active cue
            ax.text(
                0.5, 0.4,
                "⚠️ CORRECTION NEEDED",
                ha='center', va='top',
                fontsize=20, fontweight='bold',
                color=self.colors['cue'],
                transform=ax.transAxes
            )
            
            ax.text(
                0.5, 0.3,
                instruction,
                ha='center', va='top',
                fontsize=16,
                transform=ax.transAxes,
                bbox=dict(
                    boxstyle='round,pad=1',
                    facecolor=self.colors['warning'],
                    alpha=0.3,
                    edgecolor=self.colors['cue'],
                    linewidth=3
                )
            )
        else:
            # No active cue
            ax.text(
                0.5, 0.35,
                "Keep going!",
                ha='center', va='top',
                fontsize=18,
                color='gray',
                transform=ax.transAxes
            )
        
        # Phase labels
        ax.text(bar_x, bar_y - 0.05, "Start", ha='left', va='top',
               fontsize=10, transform=ax.transAxes)
        ax.text(bar_x + bar_width, bar_y - 0.05, "End", ha='right', va='top',
               fontsize=10, transform=ax.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def show_directional_cue(
        self,
        direction: str,
        body_part: str,
        magnitude: str = 'moderate',
        save_path: Optional[str] = None
    ):
        """
        Show directional guidance (up, down, left, right, etc.)
        
        Args:
            direction: 'up', 'down', 'increase', 'decrease'
            body_part: Which body part
            magnitude: 'slight', 'moderate', 'significant'
            save_path: Save path
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Body part label
        ax.text(
            5, 8,
            body_part.upper(),
            ha='center', va='center',
            fontsize=24, fontweight='bold',
            transform=ax.transData
        )
        
        # Direction arrow
        arrow_props = dict(
            facecolor=self.colors['cue'],
            edgecolor='black',
            linewidth=2,
            width=1.5,
            head_width=2.5,
            head_length=1.5
        )
        
        if direction in ['down', 'lower', 'decrease']:
            arrow = mpatches.FancyArrow(
                5, 6, 0, -2,
                **arrow_props
            )
        elif direction in ['up', 'raise', 'increase']:
            arrow = mpatches.FancyArrow(
                5, 4, 0, 2,
                **arrow_props
            )
        elif direction == 'left':
            arrow = mpatches.FancyArrow(
                6, 5, -2, 0,
                **arrow_props
            )
        elif direction == 'right':
            arrow = mpatches.FancyArrow(
                4, 5, 2, 0,
                **arrow_props
            )
        
        ax.add_patch(arrow)
        
        # Magnitude text
        magnitude_text = {
            'slight': 'Slightly',
            'moderate': 'More',
            'significant': 'Much more'
        }
        
        ax.text(
            5, 2,
            magnitude_text.get(magnitude, 'More'),
            ha='center', va='center',
            fontsize=20, fontweight='bold',
            color=self.colors['cue']
        )
        
        # Action verb
        action = direction.capitalize()
        ax.text(
            5, 1,
            action,
            ha='center', va='center',
            fontsize=18,
            style='italic'
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def show_timing_cue(
        self,
        cue_times: List[Tuple[float, str]],
        current_time: float,
        total_duration: float = 100.0,
        save_path: Optional[str] = None
    ):
        """
        Timeline view with multiple cues
        
        Args:
            cue_times: List of (time%, instruction) pairs
            current_time: Current time %
            total_duration: Total duration (default 100%)
            save_path: Save path
        """
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Timeline
        ax.plot([0, total_duration], [0, 0], 'k-', linewidth=3)
        
        # Current position
        ax.plot(current_time, 0, 'o', color=self.colors['progress'],
               markersize=20, zorder=10)
        ax.text(current_time, -0.3, 'NOW', ha='center', va='top',
               fontsize=12, fontweight='bold')
        
        # Cue markers
        for cue_time, instruction in cue_times:
            # Marker
            color = self.colors['success'] if cue_time < current_time else self.colors['cue']
            ax.plot(cue_time, 0, 's', color=color, markersize=15, zorder=5)
            
            # Label
            y_offset = 0.5 if len(cue_times) % 2 else -0.5
            ax.text(cue_time, y_offset, instruction,
                   ha='center', va='bottom' if y_offset > 0 else 'top',
                   fontsize=10, rotation=0,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
        
        ax.set_xlim(-5, total_duration + 5)
        ax.set_ylim(-1, 1)
        ax.set_xlabel('Movement Progress (%)', fontsize=12, fontweight='bold')
        ax.set_title('Correction Timeline', fontsize=14, fontweight='bold')
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


class ARInterface:
    """
    Simple AR interface data structure
    
    Provides data in format suitable for AR/VR rendering engines
    """
    
    @staticmethod
    def generate_cue_data(
        current_progress: float,
        cue_phases: List[Tuple[float, float, str]],
        body_parts: List[str]
    ) -> Dict:
        """
        Generate AR-ready cue data
        
        Args:
            current_progress: Current movement progress (0-100%)
            cue_phases: List of (start%, end%, instruction)
            body_parts: Affected body parts
            
        Returns:
            Dictionary with AR rendering data
        """
        active_cues = []
        
        for start, end, instruction in cue_phases:
            if start <= current_progress <= end:
                active_cues.append({
                    'instruction': instruction,
                    'urgency': 'high',
                    'body_parts': body_parts
                })
        
        return {
            'progress': current_progress,
            'active_cues': active_cues,
            'status': 'correcting' if active_cues else 'good',
            'timestamp': current_progress
        }


if __name__ == "__main__":
    # Test coaching visualizations
    viz = CoachingVisualizer()
    
    # Progress cue
    fig1 = viz.show_progress_cue(
        current_progress=55,
        cue_phase=(40, 70),
        instruction="Lower your body more\n(Increase knee flexion)"
    )
    
    # Directional cue
    fig2 = viz.show_directional_cue(
        direction='down',
        body_part='Knee',
        magnitude='moderate'
    )
    
    plt.show()
    
    print("Coaching visualizer test complete")
