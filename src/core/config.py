"""
Centralized Configuration and Constants

This module contains all shared constants, configurations, and common utilities
to eliminate redundancy across the codebase.
"""

from typing import Dict, List, Tuple, Any
import numpy as np

# ============================================================================
# INJURY SUBJECT MAPPING - Single Source of Truth
# ============================================================================

LEFT_INJURY_SUBJECTS = [1, 2, 3, 6, 7, 12, 16, 17, 18, 20, 21, 23, 24, 27, 30]
RIGHT_INJURY_SUBJECTS = [4, 5, 8, 9, 10, 11, 13, 14, 15, 19, 20, 21, 22, 25, 26, 28, 29, 31]

def get_injury_side(subject_id: int) -> str:
    """
    Determine injury side based on subject ID.
    
    Args:
        subject_id: Subject identifier
        
    Returns:
        'Left', 'Right', or 'Unknown'
    """
    if subject_id in LEFT_INJURY_SUBJECTS:
        return "Left"
    elif subject_id in RIGHT_INJURY_SUBJECTS:
        return "Right"
    return "Unknown"

# ============================================================================
# SENSOR SPECIFICATIONS - Single Source of Truth
# ============================================================================

class SensorSpecifications:
    """Centralized sensor specifications for IMU and sEMG sensors."""
    
    # IMU Specifications
    IMU_SAMPLING_RATE = 148.148  # Hz
    IMU_ACC_RANGE = (-2.0, 2.0)  # ±2g
    IMU_GYRO_RANGE = (-250.0, 250.0)  # ±250 deg/s
    IMU_ACC_UNIT = 'g'
    IMU_GYRO_UNIT = 'deg/s'
    
    # sEMG Specifications (for future use)
    SEMG_SAMPLING_RATE = 1259.259  # Hz
    SEMG_RANGE = (-11.0, 11.0)  # ±11 mV
    SEMG_UNIT = 'mV'
    
    # System Constants
    NUM_SENSORS = 8
    CHANNELS_PER_SENSOR = 6
    TOTAL_CHANNELS = NUM_SENSORS * CHANNELS_PER_SENSOR  # 48
    TARGET_CLASS = 0  # Healthy class
    
    @classmethod
    def get_sensor_specs(cls) -> Dict[str, Dict]:
        """Get complete sensor specifications."""
        return {
            'accelerometer': {
                'sampling_rate': cls.IMU_SAMPLING_RATE,
                'unit': cls.IMU_ACC_UNIT,
                'range': cls.IMU_ACC_RANGE,
                'bandwidth': (24, 470)
            },
            'gyroscope': {
                'sampling_rate': cls.IMU_SAMPLING_RATE,
                'unit': cls.IMU_GYRO_UNIT,
                'range': cls.IMU_GYRO_RANGE,
                'bandwidth': (24, 360)
            },
            'sEMG': {
                'sampling_rate': cls.SEMG_SAMPLING_RATE,
                'unit': cls.SEMG_UNIT,
                'range': cls.SEMG_RANGE,
                'bandwidth': (20, 450)
            }
        }
    
    @classmethod
    def get_channel_info(cls, channel_idx: int) -> Dict[str, Any]:
        """
        Get information for a specific channel.
        
        Args:
            channel_idx: Channel index (0-47)
            
        Returns:
            Dictionary with sensor, axis, unit information
        """
        if not 0 <= channel_idx < cls.TOTAL_CHANNELS:
            raise ValueError(f"Channel index {channel_idx} out of range [0, {cls.TOTAL_CHANNELS-1}]")
        
        sensor_idx = channel_idx // cls.CHANNELS_PER_SENSOR
        channel_within_sensor = channel_idx % cls.CHANNELS_PER_SENSOR
        
        # Sensor names
        sensor_names = ['R-RF', 'R-Ham', 'R-TA', 'R-Gas', 'L-RF', 'L-Ham', 'L-TA', 'L-Gas']
        
        # Channel types and axes
        if channel_within_sensor < 3:  # Accelerometer
            axis_names = ['x', 'y', 'z']
            sensor_type = 'accelerometer'
            unit = cls.IMU_ACC_UNIT
            range_vals = cls.IMU_ACC_RANGE
        else:  # Gyroscope
            axis_names = ['x', 'y', 'z']
            sensor_type = 'gyroscope'
            unit = cls.IMU_GYRO_UNIT
            range_vals = cls.IMU_GYRO_RANGE
        
        axis_idx = channel_within_sensor % 3
        
        return {
            'sensor': sensor_names[sensor_idx],
            'sensor_idx': sensor_idx,
            'axis': f'{sensor_type[:3]}_{axis_names[axis_idx]}',
            'unit': unit,
            'sensor_type': sensor_type,
            'range': range_vals,
            'sampling_rate': cls.IMU_SAMPLING_RATE
        }
    
    @classmethod
    def create_realistic_imu_data(cls, n_samples: int, n_timesteps: int) -> np.ndarray:
        """
        Create realistic IMU data within sensor specifications.
        
        Args:
            n_samples: Number of samples
            n_timesteps: Number of timesteps
            
        Returns:
            Array of shape (n_samples, 48, n_timesteps)
        """
        np.random.seed(42)  # For reproducibility
        data = np.zeros((n_samples, cls.TOTAL_CHANNELS, n_timesteps))
        
        for sample_idx in range(n_samples):
            for sensor_idx in range(cls.NUM_SENSORS):
                # Accelerometer channels (±2g)
                acc_start = sensor_idx * cls.CHANNELS_PER_SENSOR
                acc_end = acc_start + 3
                acc_data = np.random.normal(0, 0.5, (3, n_timesteps))
                data[sample_idx, acc_start:acc_end, :] = np.clip(
                    acc_data, cls.IMU_ACC_RANGE[0], cls.IMU_ACC_RANGE[1]
                )
                
                # Gyroscope channels (±250 deg/s)
                gyro_start = acc_end
                gyro_end = gyro_start + 3
                gyro_data = np.random.normal(0, 20, (3, n_timesteps))
                data[sample_idx, gyro_start:gyro_end, :] = np.clip(
                    gyro_data, cls.IMU_GYRO_RANGE[0], cls.IMU_GYRO_RANGE[1]
                )
        
        return data

# ============================================================================
# SENSOR GROUPINGS - Single Source of Truth
# ============================================================================

class SensorGroups:
    """Centralized sensor grouping definitions."""
    
    @classmethod
    def get_biomechanical_groups(cls) -> Dict[str, Dict[str, Any]]:
        """Get biomechanically meaningful sensor groups."""
        return {
            'right_thigh': {
                'channels': list(range(0, 12)),  # R-RF + R-Ham
                'muscles': ['R-RF', 'R-Ham'],
                'description': 'Right thigh muscles (primary for squats)'
            },
            'left_thigh': {
                'channels': list(range(24, 36)),  # L-RF + L-Ham
                'muscles': ['L-RF', 'L-Ham'],
                'description': 'Left thigh muscles (primary for squats)'
            },
            'right_shank': {
                'channels': list(range(12, 24)),  # R-TA + R-Gas
                'muscles': ['R-TA', 'R-Gas'],
                'description': 'Right shank muscles (stabilization)'
            },
            'left_shank': {
                'channels': list(range(36, 48)),  # L-TA + L-Gas
                'muscles': ['L-TA', 'L-Gas'],
                'description': 'Left shank muscles (stabilization)'
            },
            'right_leg': {
                'channels': list(range(0, 24)),
                'muscles': ['R-RF', 'R-Ham', 'R-TA', 'R-Gas'],
                'description': 'Complete right leg'
            },
            'left_leg': {
                'channels': list(range(24, 48)),
                'muscles': ['L-RF', 'L-Ham', 'L-TA', 'L-Gas'],
                'description': 'Complete left leg'
            },
            'thigh_muscles': {
                'channels': list(range(0, 12)) + list(range(24, 36)),
                'muscles': ['R-RF', 'R-Ham', 'L-RF', 'L-Ham'],
                'description': 'All thigh muscles (primary movers)'
            },
            'shank_muscles': {
                'channels': list(range(12, 24)) + list(range(36, 48)),
                'muscles': ['R-TA', 'R-Gas', 'L-TA', 'L-Gas'],
                'description': 'All shank muscles (stabilizers)'
            }
        }
    
    @classmethod
    def get_technical_groups(cls) -> Dict[str, List[int]]:
        """Get technical sensor groups (accelerometer vs gyroscope)."""
        return {
            'accelerometers': [i for i in range(48) if i % 6 < 3],
            'gyroscopes': [i for i in range(48) if i % 6 >= 3],
            'sensor_0': list(range(0, 6)),   # R-RF
            'sensor_1': list(range(6, 12)),  # R-Ham
            'sensor_2': list(range(12, 18)), # R-TA
            'sensor_3': list(range(18, 24)), # R-Gas
            'sensor_4': list(range(24, 30)), # L-RF
            'sensor_5': list(range(30, 36)), # L-Ham
            'sensor_6': list(range(36, 42)), # L-TA
            'sensor_7': list(range(42, 48))  # L-Gas
        }
    
    @classmethod
    def get_expert_weights_for_movement(cls, movement_type: str) -> Dict[str, float]:
        """
        Get expert-defined weights for different movement types.
        
        Args:
            movement_type: 'squat', 'extension', or 'gait'
            
        Returns:
            Dictionary of group weights
        """
        if movement_type.lower() == 'squat':
            return {
                'right_thigh': 1.2,    # Primary muscles
                'left_thigh': 1.2,     # Primary muscles
                'right_shank': 0.8,    # Secondary
                'left_shank': 0.8,     # Secondary
                'accelerometers': 1.1, # Motion patterns important
                'gyroscopes': 0.9      # Rotation less critical
            }
        elif movement_type.lower() == 'extension':
            return {
                'right_thigh': 1.3,    # Very important for extension
                'left_thigh': 1.3,     # Very important for extension
                'right_shank': 0.7,    # Less involved
                'left_shank': 0.7,     # Less involved
                'accelerometers': 1.0, # Balanced importance
                'gyroscopes': 1.0      # Balanced importance
            }
        elif movement_type.lower() == 'gait':
            return {
                'right_thigh': 1.0,    # Balanced for walking
                'left_thigh': 1.0,     # Balanced for walking
                'right_shank': 1.1,    # Important for gait
                'left_shank': 1.1,     # Important for gait
                'accelerometers': 1.2, # Critical for gait analysis
                'gyroscopes': 1.1      # Important for balance
            }
        else:
            # Default weights
            return {
                'right_thigh': 1.0, 'left_thigh': 1.0,
                'right_shank': 1.0, 'left_shank': 1.0,
                'accelerometers': 1.0, 'gyroscopes': 1.0
            }

# ============================================================================
# COMMON UTILITIES
# ============================================================================

def samples_to_time(samples: np.ndarray, sampling_rate: float = SensorSpecifications.IMU_SAMPLING_RATE) -> np.ndarray:
    """Convert sample indices to time in seconds."""
    return samples / sampling_rate

def time_to_samples(time_seconds: float, sampling_rate: float = SensorSpecifications.IMU_SAMPLING_RATE) -> int:
    """Convert time in seconds to sample index."""
    return int(time_seconds * sampling_rate)

def validate_channel_index(channel_idx: int) -> bool:
    """Validate channel index is within valid range."""
    return 0 <= channel_idx < SensorSpecifications.TOTAL_CHANNELS

def get_sensor_name_from_channel(channel_idx: int) -> str:
    """Get sensor name from channel index."""
    if not validate_channel_index(channel_idx):
        raise ValueError(f"Invalid channel index: {channel_idx}")
    
    sensor_names = ['R-RF', 'R-Ham', 'R-TA', 'R-Gas', 'L-RF', 'L-Ham', 'L-TA', 'L-Gas']
    sensor_idx = channel_idx // SensorSpecifications.CHANNELS_PER_SENSOR
    return sensor_names[sensor_idx]

# ============================================================================
# ALGORITHM CONSTANTS
# ============================================================================

class AlgorithmConstants:
    """Constants for counterfactual algorithms."""
    
    # General thresholds
    CONFIDENCE_THRESHOLD = 0.5
    MIN_TARGET_PROBABILITY = 0.5
    MASK_THRESHOLD = 0.5
    INITIAL_EPS = 1.0
    
    # Adaptive explainer specific
    EPS_DECAY = 0.9991
    MAX_RETRY_ATTEMPTS = 3
    MIN_CHANNELS = 3
    NON_SELECTED_MASK_MULTIPLIER = 0.1
    
    # Early stopping
    MAX_ITERATIONS_WITHOUT_IMPROVEMENT = 50
    
    # SHAP parameters
    DEFAULT_SHAP_SAMPLES = 100

    # Marginal importance parameters
    DEFAULT_MARGINAL_SAMPLES = 100

# ============================================================================
# VISUALIZATION CONSTANTS
# ============================================================================

class VisualizationConfig:
    """Constants for consistent visualization across the framework."""
    
    # Colors
    COLORS = {
        'original': 'blue',
        'counterfactual': 'green', 
        'mask': 'orange',
        'accelerometer': 'red',
        'gyroscope': 'purple',
        'left_leg': 'cyan',
        'right_leg': 'magenta'
    }
    
    # Figure sizes
    FIGURE_SIZES = {
        'single_plot': (12, 6),
        'comparison': (16, 10),
        'detailed_analysis': (20, 12)
    }
    
    # Plot settings
    DPI = 300
    LINEWIDTH = 1.5
    ALPHA = 0.7
    GRID_ALPHA = 0.3
