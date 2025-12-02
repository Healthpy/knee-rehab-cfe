import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))
from src.models.fcn_pytorch_model import FCN

def load_movement_data(data_dir, movement_type):
    """Load pre-processed movement data from npz files."""
    train_path = os.path.join(data_dir, f"{movement_type}_train.npz")
    test_path = os.path.join(data_dir, f"{movement_type}_test.npz")
    
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError(f"Data files not found for {movement_type}")
    
    print(f"Loading {movement_type} data from {data_dir}")
    
    train_data = np.load(train_path)
    test_data = np.load(test_path)
    
    return {
        'X_train_imu': train_data['X_train_imu'],
        'X_train_emg': train_data.get('X_train_emg', None),
        'y_train': train_data['y_train'],
        'subjects_train': train_data['subjects'],
        'X_test_imu': test_data['X_test_imu'], 
        'X_test_emg': test_data.get('X_test_emg', None),
        'y_test': test_data['y_test'],
        'subjects_test': test_data['subjects']
    }

def prepare_data_loaders(X_train_imu, y_train, X_test_imu, y_test, val_ratio=0.2, batch_size=16):
    """Prepare data loaders for training, validation and testing."""
    # Convert to PyTorch tensors
    X_train_imu_t = torch.tensor(X_train_imu, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_imu_t = torch.tensor(X_test_imu, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_imu_t, y_train_t)
    
    # Split training data into training and validation
    val_size = int(val_ratio * len(train_dataset))
    train_size = len(train_dataset) - val_size
    
    train_data, val_data = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, X_test_imu_t, y_test_t

def train_model(model, train_loader, val_loader, model_name, num_epochs=50, lr=0.0001):
    """Train the model and return the best model path."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)
        
        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dir = "models/"
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"{model_name}_best_model.pth")
            torch.save(model.state_dict(), model_path)
    
    return model_path

def evaluate_model(model_path, input_size, num_classes, test_data, test_targets):
    """Evaluate the model on test data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FCN(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    test_data = test_data.to(device)
    test_targets = test_targets.to(device)
    
    with torch.no_grad():
        outputs = model(test_data)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == test_targets).sum().item()
        total = test_targets.size(0)
        accuracy = correct / total
    
    return accuracy

def train_movement_model(movement_type, data_dir):
    """Train model for a specific movement type using pre-processed data."""
    print(f"\nProcessing {movement_type} movement...")
    
    # Load pre-processed data
    data = load_movement_data(data_dir, movement_type)
    
    X_train_imu = data['X_train_imu']
    y_train = data['y_train']
    X_test_imu = data['X_test_imu']
    y_test = data['y_test']
    
    print(f"Train data shape: {X_train_imu.shape}")
    print(f"Test data shape: {X_test_imu.shape}")
    print(f"Train labels: {np.unique(y_train, return_counts=True)}")
    print(f"Test labels: {np.unique(y_test, return_counts=True)}")
    
    # Prepare data loaders
    train_loader, val_loader, X_test_t, y_test_t = prepare_data_loaders(
        X_train_imu, y_train, X_test_imu, y_test
    )
    
    # Create and train model
    input_size = X_train_imu.shape[1]
    num_classes = len(np.unique(y_train))
    model = FCN(input_size, num_classes)
    
    print(f"Training {movement_type} model with input size: {input_size}, num classes: {num_classes}")
    model_path = train_model(model, train_loader, val_loader, model_name=movement_type)
    
    # Evaluate model
    accuracy = evaluate_model(model_path, input_size, num_classes, X_test_t, y_test_t)
    print(f"{movement_type.capitalize()} Model Accuracy: {accuracy:.4f}")
    
    return accuracy

# Main execution
if __name__ == "__main__":
    # Data paths - use the existing processed data
    data_dir = "src/data/norm_movement_data"
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        print("Please ensure the movement data files are available.")
        exit(1)
    
    # Train models for each movement type
    movement_types = ["squat", "extension", "gait"]
    accuracies = {}
    
    print("Starting model training for all movement types...")
    print(f"Using data from: {data_dir}")
    
    for movement in movement_types:
        try:
            accuracies[movement] = train_movement_model(movement, data_dir)
        except Exception as e:
            print(f"Error training {movement} model: {e}")
            accuracies[movement] = None
    
    # Print summary of results
    print("\n===== Results Summary =====")
    for movement, acc in accuracies.items():
        if acc is not None:
            print(f"{movement.capitalize()} Model Accuracy: {acc:.4f}")
        else:
            print(f"{movement.capitalize()} Model: Training failed")
    
    print("\nTraining completed!")