"""
Late Fusion TCN Training - KneE-PAD Real Data

This script trains a TCN with late fusion:
1. Load real data (EMG + IMU from 8 sensors)
2. Preprocess signals separately (NO resampling)
3. Late fusion TCN: separate branches for EMG and IMU
4. Split train/val/test by subject
5. Train and evaluate
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.processor import TimeSeriesProcessor
from src.architectures.tcn_model import LateFusionTCN
from src.data.preprocessing import preprocess_emg, preprocess_imu


class KneePADDataset(Dataset):
    """PyTorch dataset for KneE-PAD data with late fusion"""
    
    def __init__(self, emg, imu, y):
        self.emg = torch.FloatTensor(emg)
        self.imu = torch.FloatTensor(imu)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.emg[idx], self.imu[idx], self.y[idx]


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
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
    """Evaluate model"""
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
    print("KneE-PAD Late Fusion TCN Training")
    print("=" * 70)
    
    # ========================================================================
    # 1. Load Data
    # ========================================================================
    print("\n[1/5] Loading real data...")
    processor = TimeSeriesProcessor(
        emg_sampling_rate=1259.26,
        imu_sampling_rate=148.15
    )
    
    data_dict = processor.load_numpy_data()
    emg_raw = data_dict['emg']
    imu_raw = data_dict['imu']
    labels = data_dict['labels']
    subjects = data_dict['subjects']
    
    print(f"✓ Loaded {len(labels)} samples from {len(np.unique(subjects))} subjects")
    
    # Filter valid labels
    valid_mask = labels >= 0
    emg_raw = emg_raw[valid_mask]
    imu_raw = imu_raw[valid_mask]
    labels = labels[valid_mask]
    subjects = subjects[valid_mask]
    
    print(f"✓ Filtered to {len(labels)} valid samples")
    
    # ========================================================================
    # 2. Preprocess
    # ========================================================================
    print("\n[2/5] Preprocessing signals...")
    emg_processed = preprocess_emg(emg_raw, sampling_rate=1259.26)
    imu_processed = preprocess_imu(imu_raw, sampling_rate=148.15)
    print("✓ Preprocessing complete")
    print(f"  EMG: {emg_processed.shape} @ 1259.26 Hz")
    print(f"  IMU: {imu_processed.shape} @ 148.15 Hz")
    print("  → No resampling: using late fusion architecture")
    
    # ========================================================================
    # 3. Train/Val/Test Split
    # ========================================================================
    print("\n[3/5] Splitting by subject...")
    np.random.seed(42)
    
    unique_subjects = np.unique(subjects)
    shuffled_subjects = unique_subjects.copy()
    np.random.shuffle(shuffled_subjects)
    
    n_train = int(0.70 * len(unique_subjects))
    n_val = int(0.15 * len(unique_subjects))
    
    train_subjects = shuffled_subjects[:n_train]
    val_subjects = shuffled_subjects[n_train:n_train+n_val]
    test_subjects = shuffled_subjects[n_train+n_val:]
    
    train_mask = np.isin(subjects, train_subjects)
    val_mask = np.isin(subjects, val_subjects)
    test_mask = np.isin(subjects, test_subjects)
    
    emg_train, imu_train, y_train = emg_processed[train_mask], imu_processed[train_mask], labels[train_mask]
    emg_val, imu_val, y_val = emg_processed[val_mask], imu_processed[val_mask], labels[val_mask]
    emg_test, imu_test, y_test = emg_processed[test_mask], imu_processed[test_mask], labels[test_mask]
    
    print(f"✓ Train: {len(y_train)} samples, Val: {len(y_val)}, Test: {len(y_test)}")
    
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
    
    print(f"✓ Loaders ready: {len(train_loader)} train batches")
    
    # ========================================================================
    # 5. Train TCN
    # ========================================================================
    print("\n[5/5] Training Late Fusion TCN...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    print(f"  Architecture: Late Fusion TCN")
    print(f"    - EMG TCN: 8 channels → 4 dilated blocks → hidden_dim=64")
    print(f"    - IMU TCN: 48 channels → 4 dilated blocks → hidden_dim=64")
    print(f"    - Receptive field: {2**(4)-1} = 15 timesteps per block")
    
    model = LateFusionTCN(
        emg_channels=8,
        imu_channels=48,
        n_classes=9,
        num_levels=4,
        kernel_size=3,
        hidden_dim=64,
        dropout=0.2
    ).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    - Trainable parameters: {n_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    n_epochs = 20
    best_val_acc = 0
    train_losses, val_losses, test_losses = [], [], []
    train_accs, val_accs, test_accs = [], [], []
    
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
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = project_root / 'models' / 'best_tcn.pth'
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}/{n_epochs} | "
                  f"Train: {train_loss:.4f}/{train_acc:.1f}% | "
                  f"Val: {val_loss:.4f}/{val_acc:.1f}% | "
                  f"Test: {test_loss:.4f}/{test_acc:.1f}%")
    
    print("-" * 70)
    print(f"\n✓ Training complete!")
    print(f"✓ Best validation accuracy: {best_val_acc:.2f}%")
    print(f"✓ Final test accuracy: {test_accs[-1]:.2f}%")
    model_path = project_root / 'models' / 'best_tcn.pth'
    print(f"✓ Model saved as '{model_path}'")
    
    # ========================================================================
    # Visualize
    # ========================================================================
    print("\nGenerating plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(train_losses, label='Train', linewidth=2)
    ax1.plot(val_losses, label='Val', linewidth=2)
    ax1.plot(test_losses, label='Test', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('TCN Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(train_accs, label='Train', linewidth=2)
    ax2.plot(val_accs, label='Val', linewidth=2)
    ax2.plot(test_accs, label='Test', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('TCN Training Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = project_root / 'results' / 'training' / 'tcn' / 'visualizations' / 'tcn_training.png'
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to '{plot_path}'")
    
    # ========================================================================
    # Per-Class Performance
    # ========================================================================
    print("\nPer-class accuracy on test set:")
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
    
    for i in range(9):
        mask = all_labels == i
        if mask.sum() > 0:
            acc = (all_preds[mask] == all_labels[mask]).mean() * 100
            print(f"  {exercise_names[i]:<45} {acc:5.1f}%")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print("\nArchitecture: Late Fusion TCN")
    print(f"  - Parameters: {n_params:,}")
    print("  - EMG branch: 8 channels @ native 1259 Hz")
    print("  - IMU branch: 48 channels @ native 148 Hz")
    print("  - Dilated causal convolutions for temporal modeling")
    print("  - Stable gradients for counterfactual generation")
    print("=" * 70)


if __name__ == "__main__":
    main()
