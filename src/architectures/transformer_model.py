"""
Transformer-based model for time-series classification

Uses temporal attention to capture dependencies across the movement cycle.
"""

import torch
import torch.nn as nn
import math
import numpy as np
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Add positional information to time series"""
    
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [seq_len, batch, d_model]
        """
        x = x + self.pe[:x.size(0)]
        return x


class TransformerClassifier(nn.Module):
    """
    Transformer encoder for movement classification
    
    Advantages:
    - Captures long-range temporal dependencies
    - Attention weights interpretable (which phases matter)
    - Good for variable-length sequences
    """
    
    def __init__(
        self,
        n_channels: int = 56,  # 8 sensors × 7 channels (1 EMG + 6 IMU)
        n_classes: int = 9,    # 9 exercise variations
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        """
        Args:
            n_channels: Number of input channels (56: 8 Delsys sensors × 7 channels)
            n_classes: Number of output classes (9: 3 exercises × 3 variations)
            d_model: Transformer embedding dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: FFN hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(n_channels, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )
        
        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ):
        """
        Forward pass
        
        Args:
            x: Input [batch, channels, time] or [batch, time, channels]
            return_attention: Whether to return attention weights
            
        Returns:
            logits: Class logits [batch, n_classes]
            attention: (optional) Attention weights
        """
        # Handle input format
        if x.dim() == 3:
            if x.size(1) == self.n_channels:  # [batch, channels, time]
                x = x.permute(0, 2, 1)  # -> [batch, time, channels]
        
        batch_size = x.size(0)
        
        # Project to d_model
        x = self.input_proj(x)  # [batch, time, d_model]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(-1, batch_size, -1)  # [1, batch, d_model]
        x = x.permute(1, 0, 2)  # [time, batch, d_model]
        x = torch.cat([cls_tokens, x], dim=0)  # [time+1, batch, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        encoded = self.transformer(x)  # [time+1, batch, d_model]
        
        # Use CLS token for classification
        cls_output = encoded[0]  # [batch, d_model]
        
        # Classification
        logits = self.classifier(cls_output)
        
        if return_attention:
            # Extract attention weights (simplified - would need custom layer)
            return logits, None
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities"""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions"""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


class AttentionVisualizer:
    """
    Extract and visualize attention patterns
    
    Useful for understanding which temporal phases the model focuses on
    """
    
    @staticmethod
    def get_attention_map(
        model: TransformerClassifier,
        x: torch.Tensor
    ) -> np.ndarray:
        """
        Extract attention weights from model
        
        Args:
            model: Trained transformer model
            x: Input time series [batch, channels, time]
            
        Returns:
            Attention map [batch, time, time]
        """
        # This is a placeholder - actual implementation would require
        # custom transformer with attention output
        pass
    
    @staticmethod
    def plot_temporal_attention(
        attention: np.ndarray,
        time_labels: Optional[np.ndarray] = None
    ):
        """
        Visualize which time points the model attends to
        
        Args:
            attention: Attention weights [time, time]
            time_labels: Optional time labels (% of movement)
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 8))
        plt.imshow(attention, cmap='viridis', aspect='auto')
        plt.colorbar(label='Attention Weight')
        plt.xlabel('Key Position (Time)')
        plt.ylabel('Query Position (Time)')
        plt.title('Temporal Attention Pattern')
        
        if time_labels is not None:
            positions = np.linspace(0, len(attention)-1, 5).astype(int)
            plt.xticks(positions, [f"{time_labels[i]:.0f}%" for i in positions])
            plt.yticks(positions, [f"{time_labels[i]:.0f}%" for i in positions])
        
        plt.tight_layout()
        return plt.gcf()


if __name__ == "__main__":
    # Test Transformer
    model = TransformerClassifier(n_channels=20, n_classes=3)
    
    # Dummy data
    x = torch.randn(4, 20, 100)  # [batch, channels, time]
    logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print("Transformer model initialized successfully")
