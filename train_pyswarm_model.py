"""
Training script for PySwarm-optimized CNN model

This script trains the PySwarm-optimized CNN model using the optimized parameters.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import get_data_loaders
from cnn_model_pyswarm import PySwarmOptimizedCNN
from pyswarm_optimize import load_params
from train import calculate_accuracy

def train_pyswarm_model(params, num_epochs=30, data_dir='D:\\venv\\Ghanam CNN\\data_set'):

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create data loaders with the optimized batch size
    batch_size = params.get('batch_size', 32)
    train_loader, val_loader = get_data_loaders(data_dir, batch_size=batch_size)

    # Create model
    model = PySwarmOptimizedCNN(params=params).to(device)
    print(f"Created PySwarm-optimized CNN model with parameters: {params}")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=params.get('label_smoothing', 0.1))
    optimizer = optim.AdamW(
        model.parameters(),
        lr=params.get('learning_rate', 0.0005),
        weight_decay=params.get('weight_decay', 1e-4)
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        verbose=True
    )

    # Initialize lists to store metrics
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    # Best validation accuracy for model saving
    best_val_acc = 0.0

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Track statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            train_pbar.set_postfix({"loss": loss.item(), "acc": 100 * correct / total})

        # Calculate training metrics
        train_loss = running_loss / total
        train_acc = 100 * correct / total

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        # Progress bar for validation
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")

        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Track statistics
                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # Update progress bar
                val_pbar.set_postfix({"loss": loss.item(), "acc": 100 * val_correct / val_total})

        # Calculate validation metrics
        val_loss = val_running_loss / val_total
        val_acc = 100 * val_correct / val_total

        # Update scheduler
        scheduler.step(val_acc)

        # Store metrics
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, 'models/pyswarm_dog_cat_cnn_best.pth')
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")

    # Save final model
    save_model(model, 'models/pyswarm_dog_cat_cnn_final.pth')

    # Plot training history
    plot_training_history(train_loss_history, val_loss_history, train_acc_history, val_acc_history)

    print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")

    return model, {
        'train_loss': train_loss_history,
        'train_acc': train_acc_history,
        'val_loss': val_loss_history,
        'val_acc': val_acc_history
    }

def save_model(model, path):

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save model
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def plot_training_history(train_loss, val_loss, train_acc, val_acc):

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    ax1.plot(train_loss, label='Training Loss')
    ax1.plot(val_loss, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs. Epoch')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracy
    ax2.plot(train_acc, label='Training Accuracy')
    ax2.plot(val_acc, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy vs. Epoch')
    ax2.legend()
    ax2.grid(True)

    # Save figure
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/pyswarm_training_history.png')
    plt.close()

    print("Training history plot saved to plots/pyswarm_training_history.png")

def main():
    """Main function"""
    # Load parameters
    params = load_params()

    if params is None:
        print("No optimized parameters found. Using default parameters instead.")
        # Default parameters
        params = {
            'num_layers': 5,
            'conv_filters': [32, 64, 128, 256, 512],
            'kernel_sizes': [3, 3, 3, 3, 3],
            'fc_neurons': [256, 128, 64],
            'dropout_rate': 0.5,
            'learning_rate': 0.0005,
            'weight_decay': 1e-4,
            'label_smoothing': 0.1,
            'batch_size': 32
        }

    # Train model
    train_pyswarm_model(params, num_epochs=30)

if __name__ == "__main__":
    main()
