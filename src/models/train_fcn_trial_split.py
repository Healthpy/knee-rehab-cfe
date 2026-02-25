"""
FCN Training - IMU-Only with Trial-Level Split

This script trains an FCN using ONLY IMU data with TRIAL-LEVEL splitting:
- Only uses IMU data (48 channels from 8 sensors)
- Each trial is treated independently
- Same subject can appear in train/val/test sets
- Useful for comparison with EMG+IMU fusion models
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.processor import TimeSeriesProcessor
from src.architectures.fcn_model import FCN
from src.data.preprocessing import preprocess_imu, compute_normalization_stats, normalize_data


class IMUDataset(Dataset):
    """PyTorch dataset for IMU-only data"""
    
    def __init__(self, imu, y):
        self.imu = torch.FloatTensor(imu)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.imu[idx], self.y[idx]


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for imu_batch, y_batch in dataloader:
        imu_batch = imu_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(imu_batch)
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
        for imu_batch, y_batch in dataloader:
            imu_batch = imu_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(imu_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total


def main():
    print("=" * 70)
    print("KneE-PAD FCN Training - IMU-ONLY with TRIAL-LEVEL SPLIT")
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
    imu_raw = data_dict['imu']
    labels = data_dict['labels']
    subjects = data_dict['subjects']
    
    print(f"✓ Loaded {len(labels)} samples from {len(np.unique(subjects))} subjects")
    
    # Filter valid labels
    valid_mask = labels >= 0
    imu_raw = imu_raw[valid_mask]
    labels = labels[valid_mask]
    subjects = subjects[valid_mask]
    
    print(f"✓ Filtered to {len(labels)} valid samples")
    
    # ========================================================================
    # 2. Preprocess IMU Only
    # ========================================================================
    print("\n[2/5] Preprocessing IMU signals...")
    imu_processed = preprocess_imu(imu_raw, sampling_rate=148.15)
    print("✓ Preprocessing complete")
    print(f"  IMU: {imu_processed.shape} @ 148.15 Hz")
    print(f"  Using ONLY IMU data (no EMG)")
    
    # ========================================================================
    # 3. Train/Val/Test Split - TRIAL LEVEL
    # ========================================================================
    print("\n[3/5] Splitting by trial (random split)...")
    print("  ⚠️  Note: Same subject can appear in train/val/test sets")
    
    np.random.seed(42)
    n_samples = len(labels)
    
    # Random permutation of trial indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Split 70/15/15
    n_train = int(0.70 * n_samples)
    n_val = int(0.15 * n_samples)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    imu_train = imu_processed[train_idx]
    imu_val = imu_processed[val_idx]
    imu_test = imu_processed[test_idx]
    
    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]
    
    # ========================================================================
    # 3.5. Normalize IMU Data
    # ========================================================================
    print("\n[3.5/5] Computing normalization statistics...")
    # Compute mean and std from TRAINING set only
    imu_mean, imu_std = compute_normalization_stats(imu_train, axis=(0, 2))
    print(f"  IMU mean range: [{imu_mean.min():.4f}, {imu_mean.max():.4f}]")
    print(f"  IMU std range: [{imu_std.min():.4f}, {imu_std.max():.4f}]")
    
    # Normalize all splits using training statistics
    imu_train = normalize_data(imu_train, imu_mean, imu_std)
    imu_val = normalize_data(imu_val, imu_mean, imu_std)
    imu_test = normalize_data(imu_test, imu_mean, imu_std)
    
    print("✓ Data normalized using training set statistics")
    
    # Save normalization statistics
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    np.savez(models_dir / 'fcn_imu_normalization.npz', 
             mean=imu_mean, std=imu_std)
    print(f"✓ Normalization stats saved to: models/fcn_imu_normalization.npz")
    
    # Check subject overlap
    train_subjects = set(subjects[train_idx])
    val_subjects = set(subjects[val_idx])
    test_subjects = set(subjects[test_idx])
    
    print(f"✓ Train: {len(y_train)} samples from {len(train_subjects)} subjects")
    print(f"✓ Val: {len(y_val)} samples from {len(val_subjects)} subjects")
    print(f"✓ Test: {len(y_test)} samples from {len(test_subjects)} subjects")
    print(f"  Subject overlap train/val: {len(train_subjects & val_subjects)}")
    print(f"  Subject overlap train/test: {len(train_subjects & test_subjects)}")
    print(f"  Subject overlap val/test: {len(val_subjects & test_subjects)}")
    
    # ========================================================================
    # 4. Create DataLoaders
    # ========================================================================
    print("\n[4/5] Creating data loaders...")
    train_dataset = IMUDataset(imu_train, y_train)
    val_dataset = IMUDataset(imu_val, y_val)
    test_dataset = IMUDataset(imu_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"✓ Loaders ready: {len(train_loader)} train batches")
    
    # ========================================================================
    # 5. Train FCN (IMU-Only)
    # ========================================================================
    print("\n[5/5] Training FCN (IMU-Only)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    print(f"  Architecture: FCN for IMU-Only")
    
    model = FCN(
        n_channels=48,  # 8 sensors × 6 IMU channels
        n_classes=9,
        dropout=0.2
    ).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    - Trainable parameters: {n_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    n_epochs = 50
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
            # Ensure models directory exists
            models_dir = Path('models')
            models_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), models_dir / 'best_fcn_imu_trial_split.pth')
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{n_epochs} | "
                  f"Train: {train_acc:5.2f}% (loss={train_loss:.3f}) | "
                  f"Val: {val_acc:5.2f}% (loss={val_loss:.3f}) | "
                  f"Test: {test_acc:5.2f}% (loss={test_loss:.3f})")
    
    print("-" * 70)
    print(f"\n✓ Training complete!")
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")
    print(f"  Model saved to: models/best_fcn_imu_trial_split.pth")
    
    # ========================================================================
    # 6. Plot Training History
    # ========================================================================
    print("\n[6/6] Plotting training history...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Train', linewidth=2)
    ax1.plot(val_losses, label='Val', linewidth=2)
    ax1.plot(test_losses, label='Test', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('FCN Training Loss (IMU-Only, Trial-Split)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Train', linewidth=2)
    ax2.plot(val_accs, label='Val', linewidth=2)
    ax2.plot(test_accs, label='Test', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('FCN Training Accuracy (IMU-Only, Trial-Split)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    results_dir = Path('results') / 'training' / 'fcn'
    vis_dir = results_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(vis_dir / 'fcn_imu_trial_split_training.png', dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {vis_dir / 'fcn_imu_trial_split_training.png'}")
    
    # ========================================================================
    # 7. Generate Confusion Matrix
    # ========================================================================
    print("\n[7/7] Generating confusion matrix...")
    
    # Load best model
    model.load_state_dict(torch.load(models_dir / 'best_fcn_imu_trial_split.pth'))
    model.eval()
    
    # Get predictions on test set
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imu_batch, y_batch in test_loader:
            imu_batch = imu_batch.to(device)
            outputs = model(imu_batch)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Exercise class names
    class_names = [
        "Squat - Correct",
        "Squat - Weight transfer",
        "Squat - Injured leg forward",
        "Leg Ext - Correct",
        "Leg Ext - Limited ROM",
        "Leg Ext - Lifting limb",
        "Walking - Correct",
        "Walking - No extension",
        "Walking - Hip abduction"
    ]
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - FCN (IMU-Only, Trial-Split)', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save confusion matrix
    plt.savefig(vis_dir / 'fcn_imu_trial_split_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {vis_dir / 'fcn_imu_trial_split_confusion_matrix.png'}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=3))
    
    # Save classification report to file
    with open(results_dir / 'fcn_imu_trial_split_classification_report.txt', 'w') as f:
        f.write("FCN (IMU-Only, Trial-Split) - Classification Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(classification_report(all_labels, all_preds, target_names=class_names, digits=3))
    print(f"✓ Classification report saved to: {results_dir / 'fcn_imu_trial_split_classification_report.txt'}")
    
    # ========================================================================
    # Final Results
    # ========================================================================
    print("\n" + "=" * 70)
    print("FINAL RESULTS - FCN (IMU-Only, Trial-Split)")
    print("=" * 70)
    print(f"Best Val Accuracy:  {best_val_acc:.2f}%")
    print(f"Final Test Accuracy: {test_accs[-1]:.2f}%")
    print(f"Architecture: FCN with {n_params:,} parameters")
    print(f"Data: IMU-only (48 channels)")
    print(f"Split: Trial-level (70/15/15)")
    print("=" * 70)


if __name__ == "__main__":
    main()
