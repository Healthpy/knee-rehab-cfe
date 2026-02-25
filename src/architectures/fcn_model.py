"""
Fully Convolutional Network (FCN) for Movement Classification

FCN for time series classification:
- Three convolutional blocks with batch normalization
- Global average pooling
- Simple and effective for time series classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCN(nn.Module):
    """
    Fully Convolutional Network for Time Series Classification
    
    Architecture:
    - Conv Block 1: Conv1d -> BatchNorm -> ReLU
    - Conv Block 2: Conv1d -> BatchNorm -> ReLU  
    - Conv Block 3: Conv1d -> BatchNorm -> ReLU
    - Global Average Pooling
    - Linear Classification Layer
    
    Reference: 
    Wang et al., "Time Series Classification from Scratch with Deep Neural Networks"
    """
    
    def __init__(
        self,
        n_channels: int = 48,  # 48 IMU channels (8 sensors × 6 IMU channels)
        n_classes: int = 9,    # 9 exercise variations
        dropout: float = 0.2
    ):
        """
        Args:
            n_channels: Number of input channels (48 for IMU-only: 8 sensors × 6 channels)
            n_classes: Number of movement classes (9: 3 exercises × 3 variations)
            dropout: Dropout probability
        """
        super().__init__()
        
        # Conv Block 1
        self.conv1 = nn.Conv1d(n_channels, 128, kernel_size=8, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        
        # Conv Block 2
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        
        # Conv Block 3
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classification layer (after global average pooling)
        self.fc = nn.Linear(128, n_classes)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, n_channels, seq_len]
            
        Returns:
            logits: Class logits [batch_size, n_classes]
        """
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global Average Pooling
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(-1)  # [batch_size, 128]
        
        # Dropout
        x = self.dropout(x)
        
        # Classification
        logits = self.fc(x)
        
        return logits
    
    def get_features(self, x):
        """
        Extract features before classification layer
        
        Args:
            x: Input tensor [batch_size, n_channels, seq_len]
            
        Returns:
            features: Feature tensor [batch_size, 128]
        """
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global Average Pooling
        features = F.adaptive_avg_pool1d(x, 1)
        features = features.squeeze(-1)  # [batch_size, 128]
        
        return features


if __name__ == "__main__":
    # Test the model
    print("=" * 70)
    print("FCN Model Test")
    print("=" * 70)
    
    # Create model
    model = FCN(n_channels=48, n_classes=9, dropout=0.2)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: FCN for IMU-only (48 channels)")
    print(f"Trainable parameters: {n_params:,}")
    
    # Test forward pass
    batch_size = 8
    seq_len = 100
    x = torch.randn(batch_size, 48, seq_len)
    
    print(f"\nInput shape: {x.shape}")
    
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    features = model.get_features(x)
    print(f"Features shape: {features.shape}")
    
    print("\n✓ Model test passed!")
