# Dog vs Cat CNN Classifier with PySwarm Optimization

This project implements a Convolutional Neural Network (CNN) for classifying images of dogs and cats using PyTorch, with PySwarm optimization to find the best hyperparameters for the model architecture.

## Project Overview

The project consists of two main components:
1. A standard CNN model for dog and cat classification
2. A PySwarm-optimized version of the CNN model with improved performance

Both models are accessible through a user-friendly GUI that allows for easy image classification.

## Folders Structure

```
.
├── data_set/
│   ├── train/
│   │   ├── dogs/
│   │   └── cats/
│   └── test/
│       ├── dogs/
│       └── cats/
├── models/
│   ├── dog_cat_cnn_best.pth             # Original model weights
│   ├── pyswarm_dog_cat_cnn_best.pth     # PySwarm-optimized model weights
│   ├── pyswarm_params.json              # PySwarm optimization parameters
├── data_loader.py                       # Data loading and preprocessing
├── cnn_model.py                         # Original CNN model architecture
├── cnn_model_pyswarm.py                 # PySwarm-optimized CNN model
├── train.py                             # Training functions
├── train_pyswarm_model.py               # Training for PySwarm model
├── pyswarm_optimize.py                  # PySwarm optimization implementation
├── gui.py                               # Original GUI implementation
├── pyswarm_config_gui.py                # PySwarm configuration GUI
├── main.py                              # Main execution for original model
├── run.py                               # Command-line interface
├── run_pyswarm.py                       # Command-line for PySwarm
└── README.md
```

## Features

### CNN Model Architecture
- Multiple Conv2D layers with ReLU activation
- MaxPooling2D layers for downsampling
- Dropout for regularization
- Batch normalization for training stability
- Fully connected layers for classification

### PySwarm Optimization
- Particle Swarm Optimization for hyperparameter tuning
- Optimizes number of convolutional layers (3-10)
- Optimizes fully connected layer neurons (32-256, 16-128, 8-64)
- Optimizes batch size following 2^n pattern (8-64)
- Optimizes learning rate following 0.001*n pattern (0.0001-0.002)
- Optimizes dropout rate (0.1-0.7)
- Optimizes weight decay (1e-5 to 1e-3)
- Automatically saves optimized parameters to JSON file

### Training Features
- Adam optimizer with configurable learning rate
- Cross Entropy Loss with label smoothing
- Learning rate scheduling
- Early stopping to prevent overfitting
- Comprehensive metrics tracking
- Model checkpointing

### Data Processing
- Data augmentation (random flips, rotations, color jitter)
- Normalization using ImageNet statistics
- Train/validation split
- Error handling for corrupted images

### GUI Features
- Simple, clean interface with solid color backgrounds
- Upload and predict functionality
- Confidence visualization with progress bar
- Dog/cat classification with percentage confidence
- PySwarm configuration interface
- Model comparison interface


## Requirements

- Python 3.6+
- PyTorch
- torchvision
- PySwarm
- PIL (Pillow)
- matplotlib
- numpy
- tkinter
- tqdm
- json

## Usage

### Preparing the Data

1. Place dog images in the `data/train/dogs/` directory
2. Place cat images in the `data/train/cats/` directory
3. Place test dog images in the `data/test/dogs/` directory
4. Place test cat images in the `data/test/cats/` directory

### Training the Original Model

```bash
python main.py
```

This will:
1. Create necessary folders if they don't exist
2. Train the original CNN model if data is available
3. Save the trained model to the `models/` directory
4. Launch the GUI for testing the model

### Running PySwarm Optimization

```bash
python run_pyswarm.py --optimize
```

This will:
1. Run PySwarm optimization to find the best hyperparameters
2. Save the optimized parameters to `models/pyswarm_params.json`

You can customize the optimization process with these options:
- `--swarmsize <number>`: Set the number of particles in the swarm (default: 20)
- `--iterations <number>`: Set the number of iterations (default: 10)

For a faster optimization, you can use:
```bash
python run_pyswarm.py --optimize --swarmsize 5 --iterations 3
```

### Training the PySwarm-Optimized Model

```bash
python run_pyswarm.py --train
```

This will:
1. Load the optimized parameters from `models/pyswarm_params.json`
2. Train the PySwarm-optimized CNN model
3. Save the trained model to `models/pyswarm_dog_cat_cnn_best.pth`

You can customize the training process with this option:
- `--epochs <number>`: Set the number of epochs for training (default: 30)

### Comparing Models

```bash
python run_pyswarm.py --compare
```

This will launch the comparison GUI that allows you to:
1. Upload an image
2. View predictions from both the original and PySwarm-optimized models
3. Compare confidence levels and accuracy

### Using the PySwarm Configuration GUI

```bash
python pyswarm_config_gui.py
```

This will launch a GUI that allows you to:
1. Configure PySwarm optimization parameters
2. Configure CNN model parameters
3. Run PySwarm optimization
4. Upload and classify images using the trained model

### Using the GUI

1. Click "Upload Image" to select an image of a dog or cat
2. Click "Predict" to classify the image
3. View the prediction result and confidence level

### Using the Comparison GUI

1. Click "Upload Image" to select an image of a dog or cat
2. View predictions from both the original and optimized models
3. Compare confidence levels and accuracy statistics

## Model Details

### Original Model

- **Input Shape**: 224x224x3 (RGB images)
- **Output**: 2 classes (Dog, Cat)
- **Layers**:
  - 4 CNN blocks with batch normalization and max pooling
  - 3 Fully connected layers with dropout

### PySwarm-Optimized Model

- **Input Shape**: 224x224x3 (RGB images)
- **Output**: 2 classes (Dog, Cat)
- **Layers**:
  - Variable number of CNN blocks (3-10, optimized by PySwarm)
  - Each block includes Conv2D, BatchNorm2D, ReLU, and MaxPool2D
  - Optimized filter counts for each layer
  - Optimized number of neurons in fully connected layers
  - Optimized dropout rate
  - Optimized learning rate and weight decay

## PySwarm Optimization Parameters

- **Batch Size**: Follows 2^n pattern (e.g., 8, 16, 32, 64)
- **Learning Rate**: Follows 0.001*n pattern (e.g., 0.0001, 0.0005, 0.001, 0.002)
- **Swarm Size**: 20 particles (default)
- **Iterations**: 10 (default)
- **Parameter Bounds**:
  - Number of layers: 3-10
  - FC neurons 1: 32-256
  - FC neurons 2: 16-128
  - FC neurons 3: 8-64
  - Batch size exponent: 3-6 (resulting in batch sizes 8-64)
  - Learning rate multiplier: 0.1-2.0 (resulting in learning rates 0.0001-0.002)
  - Dropout rate: 0.1-0.7
  - Weight decay: 1e-5 to 1e-3

## Training Metrics

After training, the models will generate plots showing:
- Training and validation loss
- Training and validation accuracy

These plots are saved as:
- `models/training_metrics.png` (original model)
- `models/pyswarm_training_metrics.png` (PySwarm model)

## Pre-trained Models

If you don't have data for training, the system will check for pre-trained models:
- Original model: `models/dog_cat_cnn_best.pth`
- PySwarm model: `models/pyswarm_dog_cat_cnn_best.pth`

## Performance Comparison

The comparison GUI allows you to visually compare the performance of the original and PySwarm-optimized models. It displays:
- Predictions from both models
- Confidence levels for both models
- Accuracy statistics

The PySwarm-optimized model typically achieves:
- Higher accuracy
- Better generalization
- More efficient architecture (depending on the optimization results)

## Conclusion

This project demonstrates the power of PySwarm optimization for finding optimal parameters for CNN architectures. By using Particle Swarm Optimization, we can automatically discover model configurations that outperform manually designed architectures.

