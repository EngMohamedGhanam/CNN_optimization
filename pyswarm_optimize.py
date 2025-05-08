import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pyswarm import pso
from tqdm import tqdm

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import get_data_loaders
from cnn_model_pyswarm import PySwarmOptimizedCNN

# Global variables for data loaders
TRAIN_LOADER = None
VAL_LOADER = None

def setup_data(data_dir='D:\\venv\\Ghanam CNN\\data_set'):
    """Set up data loaders for optimization"""
    global TRAIN_LOADER, VAL_LOADER
    try:
        train_loader, val_loader = get_data_loaders(data_dir, batch_size=32)
        TRAIN_LOADER = train_loader
        VAL_LOADER = val_loader
        return True
    except Exception as e:
        print(f"Error setting up data: {str(e)}")
        return False

def fitness_function(params, data_dir='D:\\venv\\Ghanam CNN\\data_set'):
    """
    Fitness function for PySwarm optimization

    Args:
        params: List of parameters to optimize
            [num_layers, fc_neurons_1, fc_neurons_2, fc_neurons_3, batch_size, learning_rate, dropout_rate, weight_decay]
        data_dir: Directory containing the data

    Returns:
        float: Negative validation accuracy (since PSO minimizes)
    """
    global TRAIN_LOADER, VAL_LOADER

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Extract parameters
    num_layers = int(params[0])
    fc_neurons_1 = int(params[1])
    fc_neurons_2 = int(params[2])
    fc_neurons_3 = int(params[3])
    batch_size = int(params[4])
    learning_rate = params[5]
    dropout_rate = params[6]
    weight_decay = params[7]

    # Create parameter dictionary
    param_dict = {
        'num_layers': num_layers,
        'conv_filters': [32, 64, 128, 256, 512][:num_layers],
        'kernel_sizes': [3, 3, 3, 3, 3][:num_layers],
        'fc_neurons': [fc_neurons_1, fc_neurons_2, fc_neurons_3],
        'dropout_rate': dropout_rate,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'label_smoothing': 0.1,
        'batch_size': batch_size
    }

    print(f"\nTrying parameters: {param_dict}")

    try:
        # Initialize model with these parameters
        model = PySwarmOptimizedCNN(params=param_dict).to(device)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss(label_smoothing=param_dict['label_smoothing'])
        optimizer = optim.AdamW(
            model.parameters(),
            lr=param_dict['learning_rate'],
            weight_decay=param_dict['weight_decay']
        )

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.3,
            patience=1,
            verbose=True
        )

        # Create new data loaders with the optimized batch size
        train_loader, val_loader = get_data_loaders(data_dir, batch_size=param_dict['batch_size'])

        # Train the model for multiple epochs with early stopping
        best_val_acc = 0.0
        patience = 3
        epochs_without_improvement = 0
        num_epochs = 5  # Increased epochs for better training

        for epoch in range(num_epochs):
            val_acc = train_single_epoch(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                device,
                scheduler
            )

            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break

        print(f"Best validation accuracy: {best_val_acc:.2f}%")

        # Return negative accuracy because PSO minimizes
        return -best_val_acc

    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        return -0.0

def train_single_epoch(model, train_loader, val_loader, criterion, optimizer, device, scheduler=None):
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Progress bar for training
    train_pbar = tqdm(train_loader, desc="Training")

    for inputs, labels in train_pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        train_pbar.set_postfix({"loss": loss.item(), "acc": 100 * correct / total})

    train_loss = running_loss / total
    train_acc = 100 * correct / total

    # Validation phase
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_running_loss / val_total
    val_acc = 100 * val_correct / val_total

    if scheduler is not None:
        scheduler.step(val_acc)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    return val_acc

def optimize_model(swarmsize=20, maxiter=10, data_dir='D:\\venv\\Ghanam CNN\\data_set'):
    if not setup_data(data_dir):
        print("Cannot proceed with optimization due to missing data.")
        return None

    # Expanded parameter bounds
    # [num_layers, fc_neurons_1, fc_neurons_2, fc_neurons_3, batch_size, learning_rate, dropout_rate, weight_decay]
    lb = [3, 32, 16, 8, 8, 1e-5, 0.1, 1e-5]
    ub = [10, 256, 128, 64, 64, 1e-3, 0.7, 1e-3]

    print(f"Starting PSO optimization with {swarmsize} particles and {maxiter} iterations...")

    def fitness_wrapper(params):
        return fitness_function(params, data_dir=data_dir)

    best_params, best_score = pso(
        fitness_wrapper,
        lb,
        ub,
        swarmsize=swarmsize,
        maxiter=maxiter,
        debug=True,
        phip=0.5,
        phig=0.5
    )

    # Convert best parameters to dictionary
    num_layers = int(best_params[0])
    fc_neurons_1 = int(best_params[1])
    fc_neurons_2 = int(best_params[2])
    fc_neurons_3 = int(best_params[3])
    batch_size = int(best_params[4])
    learning_rate = best_params[5]
    dropout_rate = best_params[6]
    weight_decay = best_params[7]

    best_params_dict = {
        'num_layers': num_layers,
        'conv_filters': [32, 64, 128, 256, 512][:num_layers],
        'kernel_sizes': [3, 3, 3, 3, 3][:num_layers],
        'fc_neurons': [fc_neurons_1, fc_neurons_2, fc_neurons_3],
        'dropout_rate': dropout_rate,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'label_smoothing': 0.1,
        'batch_size': batch_size
    }

    save_params(best_params_dict)

    print(f"\nOptimization completed!")
    print(f"Best parameters: {best_params_dict}")
    print(f"Best validation accuracy: {-best_score:.2f}%")

    return best_params_dict

def save_params(params, filename='models/pyswarm_params.json'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"Parameters saved to {filename}")

def load_params(filename='models/pyswarm_params.json'):
    if not os.path.exists(filename):
        print(f"Parameter file {filename} not found")
        return None
    with open(filename, 'r') as f:
        params = json.load(f)
    print(f"Parameters loaded from {filename}")
    return params

if __name__ == "__main__":
    optimize_model(swarmsize=10, maxiter=5)