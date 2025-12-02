"""
Fully Connected Network (FCN) implementation for time series classification.

This module implements the FCN architecture using 1D convolutions for movement 
classification in the XAI counterfactual analysis framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class FCN(nn.Module):
    """
    Fully Connected Network for time series classification using 1D convolutions.
    
    This implementation uses convolutional layers followed by global average pooling
    which is more suitable for time series data than traditional fully connected layers.
    """
    
    def __init__(self, input_size: int, num_classes: int):
        """
        Initialize the FCN model.
        
        Args:
            input_size: Number of input channels (e.g., 48 for IMU channels)
            num_classes: Number of output classes
        """
        super(FCN, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # First convolutional block
        self.conv1 = nn.Conv1d(input_size, 128, kernel_size=8, stride=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        
        # Second convolutional block
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        
        # Third convolutional block
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        
        # Final classification layer
        self.fc = nn.Linear(128, num_classes)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, time_steps)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Ensure input is 3D (batch_size, channels, time_steps)
        if x.dim() == 2:
            # If 2D, assume it's (batch_size, features) and needs reshaping
            # This shouldn't happen with proper time series data, but handle it
            x = x.unsqueeze(-1)
        
        # Apply convolutional blocks
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        
        # Global average pooling over time dimension
        x = torch.mean(x, dim=2)
        
        # Final classification
        x = self.fc(x)
        
        return x
    
    
    def get_features(self, x: torch.Tensor, layer_name: str = 'conv3') -> torch.Tensor:
        """
        Extract features from a specific layer.
        
        Args:
            x: Input tensor of shape (batch_size, channels, time_steps)
            layer_name: Name of layer to extract features from ('conv1', 'conv2', 'conv3', 'gap')
            
        Returns:
            Feature tensor from the specified layer
        """
        # Ensure input is 3D
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        # Forward through layers up to the specified layer
        if layer_name == 'conv1':
            x = self.relu1(self.bn1(self.conv1(x)))
            return x
        elif layer_name == 'conv2':
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            return x
        elif layer_name == 'conv3':
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            return x
        elif layer_name == 'gap':
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = torch.mean(x, dim=2)  # Global average pooling
            return x
        else:
            raise ValueError(f"Unknown layer name: {layer_name}")
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            Probability tensor of shape (batch_size, num_classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions.
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted class indices
        """
        probabilities = self.predict_proba(x)
        predictions = torch.argmax(probabilities, dim=1)
        return predictions
    
    def get_model_info(self) -> dict:
        """
        Get information about the model architecture.
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'architecture': 'Convolutional FCN',
            'conv_layers': 3,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
    
    def save_model(self, filepath: str, include_optimizer: bool = False, **kwargs):
        """
        Save model state to file.
        
        Args:
            filepath: Path to save the model
            include_optimizer: Whether to include optimizer state
            **kwargs: Additional information to save
        """
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_size': self.input_size,
                'num_classes': self.num_classes,
            },
            **kwargs
        }
        
        torch.save(save_dict, filepath)
    
    @classmethod
    def load_model(cls, filepath: str, device: str = 'cpu'):
        """
        Load model from file.
        
        Args:
            filepath: Path to model file
            device: Device to load model to
            
        Returns:
            Loaded FCN model
        """
        checkpoint = torch.load(filepath, map_location=device)
        
        # Extract model configuration
        config = checkpoint['model_config']
        
        # Create model
        model = cls(
            input_size=config['input_size'],
            num_classes=config['num_classes']
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model


class FCNTrainer:
    """Trainer class for FCN models with advanced features."""
    
    def __init__(
        self,
        model: FCN,
        device: str = 'auto',
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001
    ):
        """
        Initialize the trainer.
        
        Args:
            model: FCN model to train
            device: Computing device
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        self.model = model
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
