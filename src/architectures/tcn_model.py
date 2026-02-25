"""
Temporal Convolutional Network (TCN) for Movement Classification

TCN advantages for KneE-PAD:
- Preserves temporal structure
- Stable gradients (good for counterfactuals)
- Receptive field covers full movement cycle
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class TemporalBlock(nn.Module):
    """
    Building block for TCN with dilated convolutions
    """
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Remove extra padding from causal convolution"""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
        
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TCNClassifier(nn.Module):
    """
    Temporal Convolutional Network for Exercise Classification
    
    Architecture:
    - Multiple dilated conv blocks (exponentially increasing receptive field)
    - Global average pooling
    - Classification head
    
    Key properties:
    - Maintains temporal resolution throughout network
    - Suitable for gradient-based counterfactual generation
    """
    
    def __init__(
        self,
        n_channels: int = 56,  # 8 sensors × 7 channels (1 EMG + 6 IMU)
        n_classes: int = 9,    # 9 exercise variations
        num_levels: int = 4,
        kernel_size: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.2
    ):
        """
        Args:
            n_channels: Number of input channels (56: 8 Delsys sensors × 7 channels)
            n_classes: Number of movement classes (9: 3 exercises × 3 variations)
            num_levels: Number of TCN blocks
            kernel_size: Convolutional kernel size
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # TCN layers
        layers = []
        num_channels = [hidden_dim] * num_levels
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = n_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                TemporalBlock(
                    in_channels, out_channels,
                    kernel_size, dilation, dropout
                )
            )
        
        self.tcn = nn.Sequential(*layers)
        
        # Classification head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, n_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Forward pass
        
        Args:
            x: Input time series [batch, channels, time]
            return_features: Whether to return intermediate features
            
        Returns:
            logits: Class logits [batch, n_classes]
            features: (optional) Temporal features [batch, hidden_dim, time]
        """
        # TCN encoding
        features = self.tcn(x)  # [batch, hidden_dim, time]
        
        # Global pooling
        pooled = self.global_pool(features).squeeze(-1)  # [batch, hidden_dim]
        
        # Classification
        h = F.relu(self.fc1(pooled))
        h = self.dropout(h)
        logits = self.fc2(h)
        
        if return_features:
            return logits, features
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities"""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions"""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


class LateFusionTCN(nn.Module):
    """
    Late Fusion TCN for EMG + IMU data
    
    Processes EMG and IMU with separate TCN branches at native sampling rates,
    then fuses features for classification.
    """
    
    def __init__(
        self,
        emg_channels: int = 8,
        imu_channels: int = 48,
        n_classes: int = 9,
        num_levels: int = 4,
        kernel_size: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.2
    ):
        """
        Args:
            emg_channels: Number of EMG channels (8)
            imu_channels: Number of IMU channels (48)
            n_classes: Number of classes (9)
            num_levels: Number of TCN blocks
            kernel_size: Convolutional kernel size
            hidden_dim: Hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.emg_channels = emg_channels
        self.imu_channels = imu_channels
        self.n_classes = n_classes
        
        # EMG branch
        emg_layers = []
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = emg_channels if i == 0 else hidden_dim
            emg_layers.append(
                TemporalBlock(
                    in_channels, hidden_dim,
                    kernel_size, dilation, dropout
                )
            )
        self.emg_tcn = nn.Sequential(*emg_layers)
        
        # IMU branch
        imu_layers = []
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = imu_channels if i == 0 else hidden_dim
            imu_layers.append(
                TemporalBlock(
                    in_channels, hidden_dim,
                    kernel_size, dilation, dropout
                )
            )
        self.imu_tcn = nn.Sequential(*imu_layers)
        
        # Global pooling
        self.emg_pool = nn.AdaptiveAvgPool1d(1)
        self.imu_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, emg: torch.Tensor, imu: torch.Tensor, return_features: bool = False):
        """
        Forward pass
        
        Args:
            emg: EMG data [batch, 8, T_emg]
            imu: IMU data [batch, 48, T_imu]
            return_features: Whether to return intermediate features
            
        Returns:
            logits: Class logits [batch, n_classes]
            features: (optional) Tuple of (emg_features, imu_features)
        """
        # Process each modality
        emg_features = self.emg_tcn(emg)  # [batch, hidden_dim, T_emg]
        imu_features = self.imu_tcn(imu)  # [batch, hidden_dim, T_imu]
        
        # Global pooling
        emg_pooled = self.emg_pool(emg_features).squeeze(-1)  # [batch, hidden_dim]
        imu_pooled = self.imu_pool(imu_features).squeeze(-1)  # [batch, hidden_dim]
        
        # Fusion
        fused = torch.cat([emg_pooled, imu_pooled], dim=1)  # [batch, hidden_dim*2]
        logits = self.fusion(fused)
        
        if return_features:
            return logits, (emg_features, imu_features)
        return logits
    
    def predict_proba(self, emg: torch.Tensor, imu: torch.Tensor) -> torch.Tensor:
        """Get class probabilities"""
        logits = self.forward(emg, imu)
        return F.softmax(logits, dim=1)
    
    def predict(self, emg: torch.Tensor, imu: torch.Tensor) -> torch.Tensor:
        """Get class predictions"""
        logits = self.forward(emg, imu)
        return torch.argmax(logits, dim=1)


class TCNTrainer:
    """
    Training wrapper for TCN model
    """
    
    def __init__(
        self,
        model: TCNClassifier,
        lr: float = 0.001,
        weight_decay: float = 1e-5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: bool = True
    ):
        """
        Train the model
        
        Args:
            X_train: Training data [N, T, C]
            y_train: Training labels [N]
            X_val: Validation data [N_val, T, C]
            y_val: Validation labels [N_val]
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Print training progress
        """
        # Convert to PyTorch format [N, C, T]
        X_train = torch.FloatTensor(X_train).permute(0, 2, 1).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        
        if X_val is not None:
            X_val = torch.FloatTensor(X_val).permute(0, 2, 1).to(self.device)
            y_val = torch.LongTensor(y_val).to(self.device)
        
        # Training loop
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            self.model.train()
            
            # Shuffle data
            perm = torch.randperm(len(X_train))
            X_train = X_train[perm]
            y_train = y_train[perm]
            
            # Batch training
            epoch_loss = 0
            epoch_correct = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                # Forward
                self.optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = self.criterion(logits, batch_y)
                
                # Backward
                loss.backward()
                self.optimizer.step()
                
                # Metrics
                epoch_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                epoch_correct += (preds == batch_y).sum().item()
            
            # Epoch metrics
            train_loss = epoch_loss / (len(X_train) / batch_size)
            train_acc = epoch_correct / len(X_train)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation
            if X_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}: "
                          f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                          f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            else:
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")
        
        return history
    
    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
        """Evaluate model on data"""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            loss = self.criterion(logits, y).item()
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean().item()
        return loss, acc
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# if __name__ == "__main__":
#     # Test single-input TCN
#     print("Testing single-input TCN...")
#     model = TCNClassifier(n_channels=20, n_classes=3)
#     x = torch.randn(4, 20, 100)  # [batch, channels, time]
#     logits = model(x)
#     print(f"  Input shape: {x.shape}")
#     print(f"  Output shape: {logits.shape}")
#     print("  ✓ Single-input TCN works\n")
    
#     # Test late fusion TCN
#     print("Testing Late Fusion TCN...")
#     late_fusion_model = LateFusionTCN(
#         emg_channels=8,
#         imu_channels=48,
#         n_classes=9,
#         num_levels=4,
#         hidden_dim=64
#     )
#     emg = torch.randn(4, 8, 5037)   # [batch, 8, T_emg]
#     imu = torch.randn(4, 48, 593)   # [batch, 48, T_imu]
#     logits = late_fusion_model(emg, imu)
#     print(f"  EMG shape: {emg.shape}")
#     print(f"  IMU shape: {imu.shape}")
#     print(f"  Output shape: {logits.shape}")
    
#     # Count parameters
#     n_params = sum(p.numel() for p in late_fusion_model.parameters())
#     print(f"  Parameters: {n_params:,}")
#     print("  ✓ Late Fusion TCN works")
