"""
Shared Evaluation Utilities for Counterfactual Explainer Evaluation

Contains common components used across M-CELS, Shapley-Adaptive, and Ablation evaluations:
- IMUExplainerEvaluator: Validity, proximity, sparsity, and continuity metrics
- visualize_imu_counterfactual: 8×2 sensor grid visualization
- Data loading, normalization, and train/val/test split pipeline
- Test sample selection (correctly predicted incorrect-exercise classes)
- Results aggregation and CSV saving
"""

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time

from src.architectures.fcn_model import FCN
from src.data.preprocessing import preprocess_imu, normalize_data, denormalize_data


# ============================================================================
# Constants
# ============================================================================

EXERCISE_NAMES = [
    "Squat - Correct", "Squat - Weight transfer", "Squat - Injured leg forward",
    "Leg Extension - Correct", "Leg Extension - Limited ROM", "Leg Extension - Lifting limb",
    "Walking - Correct", "Walking - No full extension", "Walking - Hip abduction"
]

SENSOR_NAMES = ['RF_R', 'Ham_R', 'TA_R', 'Gast_R', 'RF_L', 'Ham_L', 'TA_L', 'Gast_L']

INCORRECT_CLASSES = [1, 2, 4, 5, 7, 8]

TARGET_CLASS_MAP = {1: 0, 2: 0, 4: 3, 5: 3, 7: 6, 8: 6}

SAMPLING_RATE_HZ = 148.15

# Subject ID -> injured leg (from metadata.txt)
SUBJECT_INJURED_LEG = {
    1: 'left', 2: 'left', 3: 'left', 4: 'right', 5: 'right',
    6: 'left', 7: 'right', 8: 'right', 9: 'right', 10: 'right',
    11: 'right', 12: 'right', 13: 'right', 14: 'right', 15: 'right',
    16: 'left', 17: 'left', 18: 'left', 19: 'right', 20: 'left',
    21: 'right', 22: 'right', 23: 'right', 24: 'left', 25: 'right',
    26: 'right', 27: 'left', 28: 'right', 29: 'right', 30: 'left',
    31: 'right',
}


# ============================================================================
# IMUExplainerEvaluator
# ============================================================================

class IMUExplainerEvaluator:
    """Evaluate counterfactual explanations for IMU-only model.
    
    Provides four evaluation metrics:
    - Validity: Does the counterfactual achieve the target class?
    - Proximity: How far is the counterfactual from the original?
    - Channel sparsity: How many channels were modified?
    - Temporal continuity: How smooth are the changes over time?
    - Group modifications: Which sensor/modality groups were modified?
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def evaluate_validity(self, imu_cf, target_class):
        """Check if counterfactual achieves target class.
        
        Returns:
            tuple: (is_valid, confidence, predicted_class)
        """
        with torch.no_grad():
            imu_tensor = torch.FloatTensor(imu_cf).unsqueeze(0).to(self.device)
            logits = self.model(imu_tensor)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0, target_class].item()
        return pred == target_class, confidence, pred
    
    def evaluate_proximity(self, imu_orig, imu_cf, change_threshold=0.0001):
        """Measure distance from original.
        
        Args:
            change_threshold: Minimum absolute difference to count as changed.
        """
        imu_l2 = np.linalg.norm(imu_cf - imu_orig)
        imu_l1 = np.abs(imu_cf - imu_orig).sum()
        imu_changed_pct = (np.abs(imu_cf - imu_orig) > change_threshold).mean() * 100
        return {
            'imu_l2': imu_l2,
            'imu_l1': imu_l1,
            'imu_changed_pct': imu_changed_pct
        }
    
    def evaluate_temporal_continuity(self, imu_delta):
        """Measure temporal smoothness of changes."""
        imu_temporal_grad = np.abs(np.diff(imu_delta, axis=1)).mean()
        return {'imu_temporal_grad': imu_temporal_grad}
    
    def evaluate_channel_sparsity(self, imu_delta, channel_threshold=1e-6):
        """Count how many channels were modified (strict).
        
        A channel counts as changed if ANY timestep has a non-zero delta.
        
        Args:
            channel_threshold: Near-zero tolerance for floating-point noise.
        """
        imu_channel_change = np.abs(imu_delta).max(axis=1)  # [48] — max across timesteps
        imu_changed = (imu_channel_change > channel_threshold).sum()
        return {
            'imu_channels_changed': imu_changed,
            'imu_channel_magnitudes': imu_channel_change
        }
    
    def evaluate_group_sparsity(self, imu_delta, group_level='sensor', channel_threshold=1e-6):
        """Count how many sensor/modality groups were modified (strict).
        
        A group counts as changed if ANY channel in the group has ANY timestep
        with a non-zero delta (matching the strict channel sparsity approach).
        
        Args:
            imu_delta: Change array [48, T].
            group_level: 'sensor' (8 groups) or 'modality' (16 groups).
            channel_threshold: Near-zero tolerance for floating-point noise.
        
        Returns:
            dict with keys:
                imu_groups_changed: Number of groups with any change.
                imu_total_groups: Total number of groups (8 or 16).
                imu_group_details: Dict mapping group name -> bool (changed).
        """
        group_details = {}
        if group_level == 'sensor':
            for i, sensor in enumerate(SENSOR_NAMES):
                base_idx = i * 6
                sensor_delta = imu_delta[base_idx:base_idx + 6]
                group_details[sensor] = bool(np.abs(sensor_delta).max() > channel_threshold)
        else:  # modality
            for i, sensor in enumerate(SENSOR_NAMES):
                base_idx = i * 6
                acc_delta = imu_delta[base_idx:base_idx + 3]
                gyr_delta = imu_delta[base_idx + 3:base_idx + 6]
                group_details[f'{sensor}_acc'] = bool(np.abs(acc_delta).max() > channel_threshold)
                group_details[f'{sensor}_gyr'] = bool(np.abs(gyr_delta).max() > channel_threshold)
        
        groups_changed = sum(group_details.values())
        total_groups = len(group_details)
        return {
            'imu_groups_changed': groups_changed,
            'imu_total_groups': total_groups,
            'imu_group_details': group_details
        }
    
    def analyze_group_modifications(self, imu_delta, group_level='sensor'):
        """Analyze which sensor/modality groups were modified.
        
        Args:
            group_level: 'sensor' (8 groups) or 'modality' (16 groups).
        """
        group_changes = {}
        if group_level == 'sensor':
            for i, sensor in enumerate(SENSOR_NAMES):
                base_idx = i * 6
                sensor_delta = imu_delta[base_idx:base_idx + 6]
                group_changes[sensor] = np.abs(sensor_delta).mean()
        else:  # modality
            for i, sensor in enumerate(SENSOR_NAMES):
                base_idx = i * 6
                acc_delta = imu_delta[base_idx:base_idx + 3]
                gyr_delta = imu_delta[base_idx + 3:base_idx + 6]
                group_changes[f'{sensor}_acc'] = np.abs(acc_delta).mean()
                group_changes[f'{sensor}_gyr'] = np.abs(gyr_delta).mean()
        return group_changes


# ============================================================================
# Visualization
# ============================================================================

def visualize_imu_counterfactual(
    imu_orig, imu_cf, imu_delta,
    original_class, target_class, info,
    save_path='results/evaluation/imu_counterfactual_example.png'
):
    """Visualize IMU counterfactual changes on an 8×2 sensor grid.
    
    Args:
        imu_orig: Original IMU data [48, T] (denormalized).
        imu_cf: Counterfactual IMU data [48, T] (denormalized).
        imu_delta: Change in IMU [48, T].
        original_class: Original exercise class index.
        target_class: Target exercise class index.
        info: Dict with keys like 'success'/'valid', 'confidence'/'final_confidence',
              'iterations', 'method', 'use_shapley'.
        save_path: Path to save the visualization.
    """
    fig = plt.figure(figsize=(16, 18))
    colors = ['tab:blue', 'tab:green', 'tab:purple']
    
    for sensor_idx in range(8):
        # Column 1: Accelerometer (3 axes)
        ax = plt.subplot(8, 2, sensor_idx * 2 + 1)
        time_imu = np.arange(imu_orig.shape[1]) / SAMPLING_RATE_HZ
        
        accel_start = sensor_idx * 6
        for axis in range(3):
            ch_idx = accel_start + axis
            alpha_val = 0.6 if axis > 0 else 0.9
            ax.plot(time_imu, imu_orig[ch_idx], color=colors[axis],
                    linewidth=1.5, alpha=alpha_val, label=f'Orig {"XYZ"[axis]}')
            ax.plot(time_imu, imu_cf[ch_idx], color=colors[axis], linestyle='--',
                    linewidth=1.5, alpha=alpha_val, label=f'CF {"XYZ"[axis]}')
        
        ax.set_ylabel(f'{SENSOR_NAMES[sensor_idx]}\nAccel (g)', fontsize=9)
        ax.grid(True, alpha=0.3)
        if sensor_idx == 0:
            ax.set_title('Accelerometer (X, Y, Z)', fontweight='bold', fontsize=11)
            ax.legend(fontsize=7, ncol=3, loc='upper right')
        if sensor_idx == 7:
            ax.set_xlabel('Time (s)', fontsize=9)
        
        # Column 2: Gyroscope (3 axes)
        ax = plt.subplot(8, 2, sensor_idx * 2 + 2)
        
        gyro_start = sensor_idx * 6 + 3
        for axis in range(3):
            ch_idx = gyro_start + axis
            alpha_val = 0.6 if axis > 0 else 0.9
            ax.plot(time_imu, imu_orig[ch_idx], color=colors[axis],
                    linewidth=1.5, alpha=alpha_val, label=f'Orig {"XYZ"[axis]}')
            ax.plot(time_imu, imu_cf[ch_idx], color=colors[axis], linestyle='--',
                    linewidth=1.5, alpha=alpha_val, label=f'CF {"XYZ"[axis]}')
        
        ax.set_ylabel(f'{SENSOR_NAMES[sensor_idx]}\nGyro (deg/s)', fontsize=9)
        ax.grid(True, alpha=0.3)
        if sensor_idx == 0:
            ax.set_title('Gyroscope (X, Y, Z)', fontweight='bold', fontsize=11)
            ax.legend(fontsize=7, ncol=3, loc='upper right')
        if sensor_idx == 7:
            ax.set_xlabel('Time (s)', fontsize=9)
    
    # Build title from info dict (supports both M-CELS and Shapley-Adaptive keys)
    method = info.get('method', 'Counterfactual')
    confidence = info.get('confidence', info.get('final_confidence', 0))
    iterations = info.get('iterations', 0)
    success = info.get('success', info.get('valid', False))
    
    title_parts = [f'{method}: {EXERCISE_NAMES[original_class]} → {EXERCISE_NAMES[target_class]}']
    status_parts = []
    if 'success' in info or 'valid' in info:
        status_parts.append(f'{"✓" if success else "✗"} Success')
    status_parts.append(f'Confidence: {confidence:.1%}')
    status_parts.append(f'Iterations: {iterations}')
    if 'use_shapley' in info:
        status_parts.append(f'SHAP: {"✓" if info["use_shapley"] else "✗"}')
    title_parts.append(' | '.join(status_parts))
    
    fig.suptitle('\n'.join(title_parts), fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Visualization saved to {save_path}")
    plt.close()


# ============================================================================
# Data Loading & Preprocessing
# ============================================================================

def load_model(model_path='models/best_fcn_imu_trial_split.pth', device=None):
    """Load the trained FCN model.
    
    Returns:
        tuple: (model, device)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FCN(n_channels=48, n_classes=9, dropout=0.2).to(device)
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please train the model first using: python src/models/train_fcn_trial_split.py"
        )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✓ Loaded model from {model_path}")
    return model, device


def load_imu_data(norm_stats_path='models/fcn_imu_normalization.npz'):
    """Load IMU data, extract labels and subject IDs, preprocess, and normalize.
    
    Returns:
        tuple: (imu_processed, labels, subject_ids, imu_mean, imu_std)
            subject_ids: int array of subject IDs per sample.
            imu_mean/imu_std are None if normalization stats not found.
    """
    # Load raw arrays
    imu_raw = np.load('src/data/imu_all.npy', allow_pickle=True)
    sessions = np.load('src/data/sessions_all.npy', allow_pickle=True)
    
    # Extract labels and subject IDs from session paths
    # Format: 'dataset/Subject_X/Y/Trial_Z'
    labels = []
    subject_ids = []
    for session in sessions:
        parts = str(session).split('/')
        if len(parts) >= 3:
            try:
                label = int(parts[2])
                labels.append(label)
            except (ValueError, IndexError):
                labels.append(-1)
            # Extract subject ID
            try:
                subject_id = int(parts[1].split('_')[1])
                subject_ids.append(subject_id)
            except (ValueError, IndexError):
                subject_ids.append(-1)
        else:
            labels.append(-1)
            subject_ids.append(-1)
    labels = np.array(labels)
    subject_ids = np.array(subject_ids)
    
    # Filter valid samples
    valid_mask = labels >= 0
    imu_raw = imu_raw[valid_mask]
    labels = labels[valid_mask]
    subject_ids = subject_ids[valid_mask]
    
    # Preprocess
    imu_processed = preprocess_imu(imu_raw)
    
    # Normalize
    imu_mean, imu_std = None, None
    norm_path = Path(norm_stats_path)
    if norm_path.exists():
        norm_stats = np.load(norm_path)
        imu_mean = norm_stats['mean']
        imu_std = norm_stats['std']
        imu_processed = normalize_data(imu_processed, imu_mean, imu_std)
        print(f"✓ Applied normalization using saved statistics")
    else:
        print("⚠ Warning: Normalization stats not found, using unnormalized data")
    
    print(f"✓ Loaded {len(labels)} samples, IMU shape: {imu_processed.shape}")
    return imu_processed, labels, subject_ids, imu_mean, imu_std


def create_train_test_split(imu_processed, labels, subject_ids=None, seed=42):
    """Create reproducible 70/15/15 train/val/test split.
    
    Returns:
        dict with keys: imu_train, imu_val, imu_test, y_train, y_val, y_test,
                        and optionally subject_ids_train/val/test.
    """
    np.random.seed(seed)
    n_samples = len(labels)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    n_train = int(0.70 * n_samples)
    n_val = int(0.15 * n_samples)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    split = {
        'imu_train': imu_processed[train_idx],
        'imu_val': imu_processed[val_idx],
        'imu_test': imu_processed[test_idx],
        'y_train': labels[train_idx],
        'y_val': labels[val_idx],
        'y_test': labels[test_idx],
    }
    
    if subject_ids is not None:
        split['subject_ids_train'] = subject_ids[train_idx]
        split['subject_ids_val'] = subject_ids[val_idx]
        split['subject_ids_test'] = subject_ids[test_idx]
    
    print(f"✓ Train: {len(split['y_train'])}, Val: {len(split['y_val'])}, "
          f"Test: {len(split['y_test'])} samples")
    return split


def create_subject_split(imu_processed, labels, subject_ids,
                         train_ratio=0.70, val_ratio=0.15, seed=42):
    """Create a subject-disjoint train/val/test split.
    
    Subjects are allocated to splits using greedy largest-first bin-packing
    so that no subject appears in more than one split and sample counts
    approximate the requested ratios.
    
    Returns:
        dict: same keys as create_train_test_split, plus
              train_subjects / val_subjects / test_subjects (sets).
    """
    np.random.seed(seed)
    unique_subjects = np.unique(subject_ids)
    n_total = len(labels)
    target_train = int(train_ratio * n_total)
    target_val = int(val_ratio * n_total)

    subj_counts = {s: int((subject_ids == s).sum()) for s in unique_subjects}
    subj_list = list(unique_subjects)
    np.random.shuffle(subj_list)
    subj_list.sort(key=lambda s: subj_counts[s], reverse=True)

    train_subj, val_subj, test_subj = set(), set(), set()
    n_train, n_val, n_test = 0, 0, 0

    for s in subj_list:
        cnt = subj_counts[s]
        gap_train = target_train - n_train
        gap_val = target_val - n_val
        gap_test = (n_total - target_train - target_val) - n_test
        best = max(('train', gap_train), ('val', gap_val), ('test', gap_test),
                   key=lambda x: x[1])
        if best[0] == 'train':
            train_subj.add(s); n_train += cnt
        elif best[0] == 'val':
            val_subj.add(s); n_val += cnt
        else:
            test_subj.add(s); n_test += cnt

    train_idx = np.where(np.isin(subject_ids, list(train_subj)))[0]
    val_idx   = np.where(np.isin(subject_ids, list(val_subj)))[0]
    test_idx  = np.where(np.isin(subject_ids, list(test_subj)))[0]

    split = {
        'imu_train': imu_processed[train_idx],
        'imu_val':   imu_processed[val_idx],
        'imu_test':  imu_processed[test_idx],
        'y_train': labels[train_idx],
        'y_val':   labels[val_idx],
        'y_test':  labels[test_idx],
        'subject_ids_train': subject_ids[train_idx],
        'subject_ids_val':   subject_ids[val_idx],
        'subject_ids_test':  subject_ids[test_idx],
        'train_subjects': train_subj,
        'val_subjects':   val_subj,
        'test_subjects':  test_subj,
    }

    print(f"✓ Subject-split — Train: {len(split['y_train'])} ({len(train_subj)} subj), "
          f"Val: {len(split['y_val'])} ({len(val_subj)} subj), "
          f"Test: {len(split['y_test'])} ({len(test_subj)} subj)")
    return split


def select_test_samples(model, device, imu_test, y_test, n_samples=50, seed=42):
    """Select correctly predicted test samples from incorrect exercise classes.
    
    Args:
        n_samples: Maximum number of samples to select.
        
    Returns:
        np.array of indices into imu_test/y_test.
    """
    with torch.no_grad():
        imu_tensor = torch.FloatTensor(imu_test).to(device)
        logits = model(imu_tensor)
        probs = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        confidences = probs.cpu().numpy()
    
    incorrect_mask = np.isin(y_test, INCORRECT_CLASSES)
    correctly_predicted = (predictions == y_test)
    valid_samples = incorrect_mask & correctly_predicted
    valid_indices = np.where(valid_samples)[0]
    
    print(f"  Test samples from incorrect classes: {incorrect_mask.sum()}")
    print(f"  Correctly predicted: {valid_samples.sum()}")
    
    if len(valid_indices) > 0:
        valid_confs = np.array([confidences[i, y_test[i]] for i in valid_indices])
        print(f"  Confidence: min={np.min(valid_confs):.1%}, "
              f"mean={np.mean(valid_confs):.1%}, max={np.max(valid_confs):.1%}")
    
    np.random.seed(seed)
    n = min(n_samples, len(valid_indices))
    selected = np.random.choice(valid_indices, size=n, replace=False)
    print(f"✓ Selected {n} test samples for evaluation")
    return selected


# ============================================================================
# Predict Function Factory
# ============================================================================

def create_predict_fn(model, device):
    """Create a predict_fn wrapper compatible with both explainers.
    
    NOTE: No torch.no_grad() — gradients are needed for mask optimization.
    """
    def predict_fn(x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = x.to(device)
        logits = model(x)
        return logits
    return predict_fn


def get_target_class(original_label):
    """Map an incorrect exercise label to its correct version."""
    return TARGET_CLASS_MAP[original_label]


# ============================================================================
# Evaluation Loop
# ============================================================================

def run_evaluation_loop(
    explainer, evaluator, imu_test, y_test, selected_indices,
    imu_mean=None, imu_std=None,
    vis_dir=None, vis_count=3,
    change_threshold=0.0001, channel_threshold=1e-6,
    verbose=True
):
    """Run the counterfactual generation and evaluation loop.
    
    Args:
        explainer: Initialized counterfactual explainer (M-CELS or Shapley-Adaptive).
        evaluator: IMUExplainerEvaluator instance.
        imu_test: Test IMU data [N, 48, T].
        y_test: Test labels [N].
        selected_indices: Indices into imu_test to evaluate.
        imu_mean, imu_std: Normalization stats for denormalization in visualizations.
        vis_dir: Directory to save visualizations. None to skip.
        vis_count: Number of successful examples to visualize.
        change_threshold: Threshold for proximity changed_pct.
        channel_threshold: Threshold for channel sparsity.
        verbose: Print per-sample details.
    
    Returns:
        dict: Results dictionary with lists of per-sample metrics.
    """
    test_size = len(selected_indices)
    
    results = {
        'validity': [], 'confidence': [], 'original_class': [],
        'target_class': [], 'predicted_class': [], 'proximity': [],
        'sparsity': [], 'group_sparsity_sensor': [], 'group_sparsity_modality': [],
        'continuity': [], 'iterations': [], 'time_seconds': []
    }
    
    # Store counterfactuals for visualization
    cf_cache = {}
    
    for idx, sample_idx in enumerate(selected_indices):
        original_label = y_test[sample_idx]
        imu_orig = imu_test[sample_idx]
        target_class = get_target_class(original_label)
        
        if verbose:
            print(f"\nSample {idx+1}/{test_size}:")
            print(f"  Original: {EXERCISE_NAMES[original_label]}")
            print(f"  Target: {EXERCISE_NAMES[target_class]}")
        
        start_time = time.time()
        try:
            result = explainer.generate_counterfactual(
                data=imu_orig, label=original_label, target_class=target_class
            )
            elapsed_time = time.time() - start_time
            
            imu_cf = result['counterfactual']
            info = result['info']
            
            valid, confidence, pred = evaluator.evaluate_validity(imu_cf, target_class)
            imu_delta = imu_cf - imu_orig
            proximity = evaluator.evaluate_proximity(imu_orig, imu_cf, change_threshold)
            continuity = evaluator.evaluate_temporal_continuity(imu_delta)
            sparsity = evaluator.evaluate_channel_sparsity(imu_delta, channel_threshold)
            group_spars_sensor = evaluator.evaluate_group_sparsity(imu_delta, group_level='sensor', channel_threshold=channel_threshold)
            group_spars_modality = evaluator.evaluate_group_sparsity(imu_delta, group_level='modality', channel_threshold=channel_threshold)
            
            results['validity'].append(valid)
            results['confidence'].append(confidence)
            results['original_class'].append(original_label)
            results['target_class'].append(target_class)
            results['predicted_class'].append(pred)
            results['proximity'].append(proximity)
            results['sparsity'].append(sparsity)
            results['group_sparsity_sensor'].append(group_spars_sensor)
            results['group_sparsity_modality'].append(group_spars_modality)
            results['continuity'].append(continuity)
            results['iterations'].append(info.get('iterations', 0))
            results['time_seconds'].append(elapsed_time)
            
            # Cache for visualization
            if valid:
                cf_cache[idx] = (imu_orig, imu_cf, info, original_label, target_class, confidence)
            
            if verbose:
                print(f"  ✓ Valid: {valid}, Confidence: {confidence:.2%}, "
                      f"Iterations: {info.get('iterations', 0)}, Time: {elapsed_time:.2f}s")
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            if verbose:
                print(f"  ✗ Error: {e}")
            results['validity'].append(False)
            results['confidence'].append(0.0)
            results['original_class'].append(original_label)
            results['target_class'].append(target_class)
            results['predicted_class'].append(original_label)
            results['proximity'].append({'imu_l2': 0, 'imu_l1': 0, 'imu_changed_pct': 0})
            results['sparsity'].append({'imu_channels_changed': 0, 'imu_channel_magnitudes': np.zeros(48)})
            results['group_sparsity_sensor'].append({'imu_groups_changed': 0, 'imu_total_groups': 8, 'imu_group_details': {}})
            results['group_sparsity_modality'].append({'imu_groups_changed': 0, 'imu_total_groups': 16, 'imu_group_details': {}})
            results['continuity'].append({'imu_temporal_grad': 0})
            results['iterations'].append(0)
            results['time_seconds'].append(elapsed_time)
    
    print(f"\n✓ Completed evaluation on {test_size} test samples")
    
    # Generate visualizations for successful examples
    if vis_dir and cf_cache:
        _generate_visualizations(
            cf_cache, vis_dir, vis_count, imu_mean, imu_std
        )
    
    return results


def _generate_visualizations(cf_cache, vis_dir, vis_count, imu_mean, imu_std):
    """Generate visualizations for successful counterfactuals."""
    vis_dir = Path(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    successful_keys = list(cf_cache.keys())[:vis_count]
    print(f"\n  Creating visualizations for {len(successful_keys)} successful examples...")
    
    for vis_idx, key in enumerate(successful_keys):
        imu_orig, imu_cf, info, original_label, target_class, confidence = cf_cache[key]
        
        # Populate info for title
        info['confidence'] = confidence
        info['valid'] = True
        if 'method' not in info:
            info['method'] = 'Counterfactual'
        
        # Denormalize for visualization
        if imu_mean is not None and imu_std is not None:
            imu_orig_vis = denormalize_data(imu_orig, imu_mean, imu_std)
            imu_cf_vis = denormalize_data(imu_cf, imu_mean, imu_std)
        else:
            imu_orig_vis = imu_orig
            imu_cf_vis = imu_cf
        imu_delta_vis = imu_cf_vis - imu_orig_vis
        
        save_path = vis_dir / f'example_{vis_idx+1}_class{original_label}_to_{target_class}.png'
        visualize_imu_counterfactual(
            imu_orig_vis, imu_cf_vis, imu_delta_vis,
            original_label, target_class, info,
            save_path=str(save_path)
        )
    
    print(f"  ✓ Visualizations saved to {vis_dir}")


# ============================================================================
# Results Aggregation & Saving
# ============================================================================

def print_results_summary(results):
    """Print a formatted summary of evaluation results."""
    success_rate = np.mean(results['validity']) * 100
    avg_confidence = np.mean(results['confidence']) * 100
    avg_iterations = np.mean(results['iterations'])
    avg_time = np.mean(results['time_seconds'])
    total_time = np.sum(results['time_seconds'])
    
    avg_imu_l2 = np.mean([p['imu_l2'] for p in results['proximity']])
    avg_imu_changed = np.mean([p['imu_changed_pct'] for p in results['proximity']])
    avg_imu_channels = np.mean([s['imu_channels_changed'] for s in results['sparsity']])
    
    print(f"\n📊 Success Metrics:")
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Average Confidence: {avg_confidence:.1f}%")
    print(f"  Average Iterations: {avg_iterations:.1f}")
    print(f"  Average Time: {avg_time:.2f}s")
    print(f"  Total Time: {total_time:.2f}s")
    
    print(f"\n📏 Proximity Metrics:")
    print(f"  Average IMU L2 Distance: {avg_imu_l2:.4f}")
    print(f"  Average IMU Changed: {avg_imu_changed:.2f}%")
    
    avg_sensor_groups = np.mean([g['imu_groups_changed'] for g in results['group_sparsity_sensor']])
    avg_modality_groups = np.mean([g['imu_groups_changed'] for g in results['group_sparsity_modality']])
    
    print(f"\n🎯 Sparsity Metrics:")
    print(f"  Average IMU Channels Changed: {avg_imu_channels:.1f} / 48")
    print(f"  Average Sensor Groups Changed: {avg_sensor_groups:.1f} / 8")
    print(f"  Average Modality Groups Changed: {avg_modality_groups:.1f} / 16")
    
    return {
        'success_rate': success_rate,
        'avg_confidence': avg_confidence,
        'avg_iterations': avg_iterations,
        'avg_time': avg_time,
        'avg_imu_l2': avg_imu_l2,
        'avg_imu_changed': avg_imu_changed,
        'avg_imu_channels': avg_imu_channels,
        'avg_sensor_groups': avg_sensor_groups,
        'avg_modality_groups': avg_modality_groups,
    }


def save_detailed_results(results, selected_indices, csv_path,
                          subject_ids_test=None, extra_summary=None):
    """Save per-sample results to CSV and a summary CSV.
    
    Args:
        results: Results dict from run_evaluation_loop.
        selected_indices: Array of test indices evaluated.
        csv_path: Path for the detailed CSV (summary will be <stem>_summary.csv).
        subject_ids_test: Optional array of subject IDs for the test set.
        extra_summary: Optional dict of extra rows for the summary CSV.
    """
    results_dir = Path(csv_path).parent
    results_dir.mkdir(exist_ok=True)
    
    detailed_rows = []
    for i in range(len(results['validity'])):
        row = {
            'sample_idx': selected_indices[i],
            'original_class': results['original_class'][i],
            'original_class_name': EXERCISE_NAMES[results['original_class'][i]],
            'target_class': results['target_class'][i],
            'target_class_name': EXERCISE_NAMES[results['target_class'][i]],
            'predicted_class': results['predicted_class'][i],
            'predicted_class_name': EXERCISE_NAMES[results['predicted_class'][i]],
        }
        
        # Subject / injured leg info
        if subject_ids_test is not None:
            subj_id = int(subject_ids_test[selected_indices[i]])
            row['subject_id'] = subj_id
            row['injured_leg'] = SUBJECT_INJURED_LEG.get(subj_id, 'unknown')
        
        row.update({
            'valid': results['validity'][i],
            'confidence': results['confidence'][i],
            'iterations': results['iterations'][i],
            'time_seconds': results['time_seconds'][i],
            'imu_l2': results['proximity'][i]['imu_l2'],
            'imu_l1': results['proximity'][i]['imu_l1'],
            'imu_changed_pct': results['proximity'][i]['imu_changed_pct'],
            'imu_channels_changed': results['sparsity'][i]['imu_channels_changed'],
            'sensor_groups_changed': results['group_sparsity_sensor'][i]['imu_groups_changed'],
            'modality_groups_changed': results['group_sparsity_modality'][i]['imu_groups_changed'],
        })
        
        # List which sensor groups were changed
        sensor_details = results['group_sparsity_sensor'][i].get('imu_group_details', {})
        changed_sensors = [name for name, changed in sensor_details.items() if changed]
        row['changed_sensor_groups'] = '; '.join(changed_sensors) if changed_sensors else 'none'
        
        # List which modality groups were changed
        modality_details = results['group_sparsity_modality'][i].get('imu_group_details', {})
        changed_modalities = [name for name, changed in modality_details.items() if changed]
        row['changed_modality_groups'] = '; '.join(changed_modalities) if changed_modalities else 'none'
        
        row['imu_temporal_grad'] = results['continuity'][i]['imu_temporal_grad']
        detailed_rows.append(row)
    
    df_detailed = pd.DataFrame(detailed_rows)
    df_detailed.to_csv(csv_path, index=False)
    print(f"✓ Detailed results saved to: {csv_path}")
    
    # Summary
    metrics = print_results_summary(results)
    
    summary_data = {
        'Metric': [
            'Success Rate (%)', 'Average Confidence (%)',
            'Average Iterations', 'Average Time (s)',
            'Average IMU L2', 'Average IMU Changed (%)',
            'Average IMU Channels Changed',
            'Average Sensor Groups Changed (/8)',
            'Average Modality Groups Changed (/16)',
        ],
        'Value': [
            metrics['success_rate'], metrics['avg_confidence'],
            metrics['avg_iterations'], metrics['avg_time'],
            metrics['avg_imu_l2'], metrics['avg_imu_changed'],
            metrics['avg_imu_channels'],
            metrics['avg_sensor_groups'],
            metrics['avg_modality_groups'],
        ]
    }
    
    if extra_summary:
        for key, value in extra_summary.items():
            summary_data['Metric'].append(key)
            summary_data['Value'].append(value)
    
    df_summary = pd.DataFrame(summary_data)
    summary_path = Path(csv_path).with_name(Path(csv_path).stem + '_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"✓ Summary saved to: {summary_path}")
    
    return metrics
