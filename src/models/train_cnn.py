"""
Training Pipeline - KneE-PAD Real Data with Late Fusion

This script demonstrates the complete training pipeline with late fusion:
1. Load real data (EMG + IMU from 8 sensors)
2. Preprocess signals separately (filtering, rectification)
3. NO resampling - keep native sampling rates
4. Late fusion: separate branches for EMG and IMU, fuse before classification
5. Split train/val/test by subject
6. Train late fusion model for 9-class classification
7. Evaluate and save model
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.processor import TimeSeriesProcessor


class LateFusionModel(nn.Module):
    """
    Late Fusion Model for EMG + IMU
    
    Separate branches for EMG and IMU, fuse features before classification.
    Each modality keeps its native sampling rate and temporal resolution.
    """
    
    def __init__(
        self,
        emg_channels=8,
        imu_channels=48,
        n_classes=9,
        hidden_dim=64,
        dropout=0.2
    ):
        super().__init__()
        
        # EMG branch (8 channels)
        self.emg_branch = nn.Sequential(
            nn.Conv1d(emg_channels, hidden_dim, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1)  # Global pooling
        )
        
        # IMU branch (48 channels)
        self.imu_branch = nn.Sequential(
            nn.Conv1d(imu_channels, hidden_dim, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1)  # Global pooling
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, emg, imu):
        """
        Args:
            emg: [batch, 8, T_emg]
            imu: [batch, 48, T_imu]
            
        Returns:
            logits: [batch, n_classes]
        """
        # Process each modality
        emg_features = self.emg_branch(emg).squeeze(-1)  # [batch, hidden_dim]
        imu_features = self.imu_branch(imu).squeeze(-1)  # [batch, hidden_dim]
        
        # Concatenate features
        fused = torch.cat([emg_features, imu_features], dim=1)  # [batch, hidden_dim*2]
        
        # Classification
        logits = self.fusion(fused)
        
        return logits


class KneePADDataset(Dataset):
    """PyTorch dataset for KneE-PAD data with late fusion"""
    
    def __init__(self, emg, imu, y):
        """
        Args:
            emg: EMG data [N, 8, T_emg]
            imu: IMU data [N, 48, T_imu]
            y: Labels [N]
        """
        self.emg = torch.FloatTensor(emg)
        self.imu = torch.FloatTensor(imu)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.emg[idx], self.imu[idx], self.y[idx]


def preprocess_emg(emg_data, sampling_rate=1259.26):
    """
    Preprocess EMG signals: rectify + envelope
    
    Args:
        emg_data: [N, 8, T] EMG data
        sampling_rate: EMG sampling rate
        
    Returns:
        Preprocessed EMG [N, 8, T]
    """
    print("  Preprocessing EMG: rectify + envelope...")
    N, C, T = emg_data.shape
    processed = np.zeros_like(emg_data)
    
    # Design envelope filter (6 Hz lowpass)
    nyquist = sampling_rate / 2
    b, a = signal.butter(4, 6.0 / nyquist, btype='low')
    
    for i in range(N):
        for ch in range(C):
            # Rectify
            rectified = np.abs(emg_data[i, ch, :])
            # Envelope
            processed[i, ch, :] = signal.filtfilt(b, a, rectified)
    
    return processed


def preprocess_imu(imu_data, sampling_rate=148.15):
    """
    Preprocess IMU signals: lowpass filter
    
    Args:
        imu_data: [N, 48, T] IMU data
        sampling_rate: IMU sampling rate
        
    Returns:
        Preprocessed IMU [N, 48, T]
    """
    print("  Preprocessing IMU: lowpass filter...")
    N, C, T = imu_data.shape
    processed = np.zeros_like(imu_data)
    
    # Design lowpass filter (10 Hz cutoff)
    nyquist = sampling_rate / 2
    b, a = signal.butter(4, 10.0 / nyquist, btype='low')
    
    for i in range(N):
        for ch in range(C):
            processed[i, ch, :] = signal.filtfilt(b, a, imu_data[i, ch, :])
    
    return processed


def resample_to_common_length(emg_data, imu_data, target_length=600):
    """
    Resample EMG and IMU to common temporal length
    
    Args:
        emg_data: [N, 8, T_emg] EMG data
        imu_data: [N, 48, T_imu] IMU data
        target_length: Target time points
        
    Returns:
        emg_resampled: [N, 8, target_length]
        imu_resampled: [N, 48, target_length]
    """
    print(f"  Resampling to {target_length} time points...")
    N = emg_data.shape[0]
    
    emg_resampled = np.zeros((N, 8, target_length))
    imu_resampled = np.zeros((N, 48, target_length))
    
    for i in range(N):
        # EMG resampling
        for ch in range(8):
            T_orig = emg_data.shape[2]
            time_orig = np.linspace(0, 1, T_orig)
            time_new = np.linspace(0, 1, target_length)
            f = interp1d(time_orig, emg_data[i, ch, :], kind='cubic')
            emg_resampled[i, ch, :] = f(time_new)
        
        # IMU resampling
        for ch in range(48):
            T_orig = imu_data.shape[2]
            time_orig = np.linspace(0, 1, T_orig)
            time_new = np.linspace(0, 1, target_length)
            f = interp1d(time_orig, imu_data[i, ch, :], kind='cubic')
            imu_resampled[i, ch, :] = f(time_new)
    
    return emg_resampled, imu_resampled


def combine_emg_imu(emg_data, imu_data):
    """
    Combine EMG and IMU into unified 56-channel format
    
    Channel layout per sensor:
    [EMG, Accel_X, Accel_Y, Accel_Z, Gyro_X, Gyro_Y, Gyro_Z]
    
    Args:
        emg_data: [N, 8, T] EMG data
        imu_data: [N, 48, T] IMU data
        
    Returns:
        combined: [N, 56, T] Combined data
    """
    print("  Combining EMG + IMU into 56-channel format...")
    N, _, T = emg_data.shape
    combined = np.zeros((N, 56, T))
    
    for sensor_id in range(8):
        base_idx = sensor_id * 7
        
        # EMG channel
        combined[:, base_idx, :] = emg_data[:, sensor_id, :]
        
        # IMU channels (6 per sensor)
        imu_start = sensor_id * 6
        combined[:, base_idx+1:base_idx+7, :] = imu_data[:, imu_start:imu_start+6, :]
    
    return combined


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch with late fusion"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for emg_batch, imu_batch, y_batch in dataloader:
        emg_batch = emg_batch.to(device)
        imu_batch = imu_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(emg_batch, imu_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate model with late fusion"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for emg_batch, imu_batch, y_batch in dataloader:
            emg_batch = emg_batch.to(device)
            imu_batch = imu_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(emg_batch, imu_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total


def main():
    print("=" * 70)
    print("KneE-PAD Training Pipeline")
    print("=" * 70)
    
    # ========================================================================
    # 1. Load Data
    # ========================================================================
    print("\n[1/7] Loading real data...")
    processor = TimeSeriesProcessor(
        emg_sampling_rate=1259.26,
        imu_sampling_rate=148.15
    )
    
    data_dict = processor.load_numpy_data()
    emg_raw = data_dict['emg']      # [N, 8, T_emg]
    imu_raw = data_dict['imu']      # [N, 48, T_imu]
    labels = data_dict['labels']     # [N]
    subjects = data_dict['subjects'] # [N]
    
    print(f"✓ Loaded {len(labels)} samples from {len(np.unique(subjects))} subjects")
    
    # Filter out invalid labels
    valid_mask = labels >= 0
    emg_raw = emg_raw[valid_mask]
    imu_raw = imu_raw[valid_mask]
    labels = labels[valid_mask]
    subjects = subjects[valid_mask]
    
    print(f"✓ Filtered to {len(labels)} valid samples (labels 0-8)")
    
    # ========================================================================
    # 2. Preprocess Signals (keep native sampling rates)
    # ========================================================================
    print("\n[2/5] Preprocessing signals...")
    emg_processed = preprocess_emg(emg_raw, sampling_rate=1259.26)
    imu_processed = preprocess_imu(imu_raw, sampling_rate=148.15)
    print("✓ Preprocessing complete")
    print(f"  EMG shape: {emg_processed.shape} (native rate: 1259.26 Hz)")
    print(f"  IMU shape: {imu_processed.shape} (native rate: 148.15 Hz)")
    print("  → No resampling: using late fusion architecture")
    
    # ========================================================================
    # 3. Train/Val/Test Split by Subject
    # ========================================================================
    print("\n[3/5] Splitting train/val/test by subject...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)
    
    # Randomly shuffle subjects
    shuffled_subjects = unique_subjects.copy()
    np.random.shuffle(shuffled_subjects)
    
    # Split: 70% train, 15% validation, 15% test
    n_train = int(0.70 * n_subjects)
    n_val = int(0.15 * n_subjects)
    
    train_subjects = shuffled_subjects[:n_train]
    val_subjects = shuffled_subjects[n_train:n_train+n_val]
    test_subjects = shuffled_subjects[n_train+n_val:]
    
    # Create masks
    train_mask = np.isin(subjects, train_subjects)
    val_mask = np.isin(subjects, val_subjects)
    test_mask = np.isin(subjects, test_subjects)
    
    # Split data
    emg_train = emg_processed[train_mask]
    y_train = labels[train_mask]
    emg_val = emg_processed[val_mask]
    y_val = labels[val_mask]
    emg_test = emg_processed[test_mask]
    y_test = labels[test_mask]
    
    imu_train = imu_processed[train_mask]
    imu_val = imu_processed[val_mask]
    imu_test = imu_processed[test_mask]
    
    print(f"✓ Train: {len(y_train)} samples from {len(train_subjects)} subjects")
    print(f"✓ Val:   {len(y_val)} samples from {len(val_subjects)} subjects")
    print(f"✓ Test:  {len(y_test)} samples from {len(test_subjects)} subjects")
    print(f"  Train subjects: {sorted(train_subjects.tolist())}")
    print(f"  Val subjects:   {sorted(val_subjects.tolist())}")
    print(f"  Test subjects:  {sorted(test_subjects.tolist())}")
    
    # ========================================================================
    # 4. Create DataLoaders
    # ========================================================================
    print("\n[4/5] Creating data loaders...")
    train_dataset = KneePADDataset(emg_train, imu_train, y_train)
    val_dataset = KneePADDataset(emg_val, imu_val, y_val)
    test_dataset = KneePADDataset(emg_test, imu_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches:   {len(val_loader)}")
    print(f"✓ Test batches:  {len(test_loader)}")
    
    # ========================================================================
    # 5. Train Late Fusion Model
    # ========================================================================
    print("\n[5/5] Training late fusion model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    print(f"  Architecture: Late Fusion (EMG branch + IMU branch)")
    
    model = LateFusionModel(
        emg_channels=8,
        imu_channels=48,
        n_classes=9,
        hidden_dim=64,
        dropout=0.2
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Training loop
    n_epochs = 50
    best_acc = 0
    train_losses = []
    val_losses = []
    test_losses = []
    train_accs = []
    val_accs = []
    test_accs = []
    
    print(f"\nTraining for {n_epochs} epochs...")
    print("-" * 70)
    
    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        
        # Save best model based on validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            model_path = project_root / 'models' / 'best_cnn.pth'
            torch.save(model.state_dict(), model_path)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}/{n_epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
                  f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}%")
    
    print("-" * 70)
    print(f"\n✓ Training complete!")
    print(f"✓ Best validation accuracy: {best_acc:.2f}%")
    print(f"✓ Final test accuracy: {test_accs[-1]:.2f}%")
    print(f"✓ Model saved as 'models/best_cnn.pth'")
    
    # ========================================================================
    # Visualize Training Progress
    # ========================================================================
    print("\nGenerating training plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.plot(test_losses, label='Test Loss', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training, Validation and Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Train Acc', linewidth=2)
    ax2.plot(val_accs, label='Val Acc', linewidth=2)
    ax2.plot(test_accs, label='Test Acc', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training, Validation and Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = project_root / 'results' / 'training' / 'cnn' / 'visualizations' / 'cnn_training_progress.png'
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training plot saved to '{plot_path}'")
    
    # ========================================================================
    # Per-Class Performance
    # ========================================================================
    print("\nEvaluating per-class performance...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for emg_batch, imu_batch, y_batch in test_loader:
            emg_batch = emg_batch.to(device)
            imu_batch = imu_batch.to(device)
            outputs = model(emg_batch, imu_batch)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    exercise_names = [
        "Squat - Correct",
        "Squat - Weight transfer",
        "Squat - Injured leg forward",
        "Leg Extension - Correct",
        "Leg Extension - Limited ROM",
        "Leg Extension - Lifting limb",
        "Walking - Correct",
        "Walking - No full extension",
        "Walking - Hip abduction"
    ]
    
    print("\nPer-class accuracy:")
    for i in range(9):
        mask = all_labels == i
        if mask.sum() > 0:
            acc = (all_preds[mask] == all_labels[mask]).mean() * 100
            print(f"  {exercise_names[i]:<45} {acc:5.1f}%")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print("\nArchitecture: Late Fusion")
    print("  - EMG branch: 8 channels @ 1259.26 Hz (native rate)")
    print("  - IMU branch: 48 channels @ 148.15 Hz (native rate)")
    print("  - Fusion: Concatenate features → Classification")
    print("\nNext steps:")
    print("  1. Review training_progress.png")
    print("  2. Load model: model.load_state_dict(torch.load('best_model.pth'))")
    print("  3. Generate counterfactuals for incorrect executions")
    print("  4. Visualize explanations with patient_view or clinician_view")
    print("=" * 70)


if __name__ == "__main__":
    main()
