import os
import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import create_folders, get_data_loaders
from cnn_model import DogCatCNN
from train import train_model
from gui import launch_gui

def main():
    # Create necessary folders
    create_folders()

    # Set device (fixed 'gpu' to 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters - improved for better accuracy
    batch_size = 16  # Smaller batch size for better generalization
    learning_rate = 0.0005  # Lower learning rate for more stable training
    num_epochs = 20  # Increased epochs for better learning
    dropout_rate = 0.4  # Slightly reduced dropout

    # Use the correct data directory path
    data_dir = 'D:\\venv\\project_optimization\\data_set'
    train_dir = os.path.join(data_dir, 'training_set', 'training_set')

    # Check if data exists
    if not os.path.exists(train_dir) or not os.path.exists(os.path.join(train_dir, 'dogs')) or not os.path.exists(os.path.join(train_dir, 'cats')):
        print(f"Could not find data in {train_dir}")
        print("Please make sure the data is in the correct location.")
        print("You can use the GUI to test a pre-trained model.")

        # Check if a pre-trained model exists
        model_path = 'models/dog_cat_cnn_best.pth'
        if os.path.exists(model_path):
            print(f"Found pre-trained model at {model_path}")
            launch_gui(model_path, device)
        return

    # Create data loaders
    train_loader, val_loader = get_data_loaders(data_dir, batch_size)

    # Check if data loaders were created successfully
    if train_loader is None or val_loader is None:
        print("\nCould not create data loaders. Checking for pre-trained model...")
        model_path = 'models/dog_cat_cnn_best.pth'

        if os.path.exists(model_path):
            print(f"Found pre-trained model at {model_path}")
            print("Launching GUI with pre-trained model...")
            launch_gui(model_path, device)
        else:
            print("No pre-trained model found.")
            print("Please either:")
            print("1. Add training data to the data/train and data/test directories")
            print("2. Download a pre-trained model and place it in the models directory")
        return

    # Print dataset information
    print("\nDataset Information:")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # Initialize model
    model = DogCatCNN(dropout_rate=dropout_rate).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Added label smoothing for better generalization
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # AdamW with weight decay

    # Learning rate scheduler - more sophisticated scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # Monitor validation accuracy
        factor=0.5,  # Reduce LR by half when plateau
        patience=3,  # Wait 3 epochs before reducing
        verbose=True
    )

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, scheduler)

    # Save the model
    model_path = 'models/dog_cat_cnn_best.pth'
    print(f"Saving model to {model_path}")

    # Launch GUI with trained model
    print("Launching GUI with trained model...")
    launch_gui(model_path, device)

if __name__ == "__main__":
    main()
