# Dog vs Cat CNN Classifier with PySwarm Optimization

This project implements a Convolutional Neural Network (CNN) for classifying images of dogs and cats using PyTorch, with PySwarm optimization to find the best hyperparameters for the model architecture.

## Folders Structure

```
.
├── models/
│   ├── dog_cat_cnn_best.pth             # Original model weights
│   ├── optimized_dog_cat_cnn_best.pth   # Optimized model weights
│   ├── optimized_params.json            # Optimized parameters
│   └── training_metrics.png             # Training metrics plot
├── data_loader.py                       # Data loading and preprocessing
├── cnn_model.py                         # Original CNN model architecture
├── cnn_model_optimized.py               # Optimized CNN model architecture                          # Training functions
├── gui.py                               # GUI implementation
├── comparison_gui.py                    # Comparison GUI for original vs optimized models
├── optimize_model.py                    # PySwarm optimization implementation
├── run_optimization.py                  # Script to run optimization and comparison
├── main.py                              # Main execution
└── README.md
```

## Features

- **CNN Model Architecture**:
  - Multiple Conv2D layers with ReLU activation
  - MaxPooling2D layers
  - Dropout for regularization
  - Dynamic convolution capabilities
  - Batch normalization
  - Fully connected layers

- **PySwarm Optimization**:
  - Particle Swarm Optimization for hyperparameter tuning
  - Optimizes number of layers (3-20)
  - Optimizes convolutional filter counts (3-256)
  - Optimizes kernel sizes (3×3 to 7×7)
  - Optimizes FC layer neuron counts (1-300)
  - Optimizes learning rate, weight decay, and label smoothing
  - Automatically limits layers based on input size
  - Saves optimized parameters to JSON file

- **Training**:
  - Adam optimizer
  - Cross Entropy Loss
  - Learning rate scheduling
  - Metrics tracking (accuracy, loss, validation accuracy, validation loss)
  - Model saving functionality

- **Data Processing**:
  - Data augmentation (random flips, rotations, color jitter)
  - Normalization
  - Train/validation split

- **GUI**:
  - Simple interface for image classification
  - Upload and predict functionality
  - Confidence visualization

- **Comparison GUI**:
  - Side-by-side comparison of original and optimized models
  - Shows predictions from both models on the same image
  - Displays confidence levels for both models
  - Tracks accuracy statistics

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

### Running the Optimization

```bash
python run_optimization.py --optimize
```

This will:
1. Run PySwarm optimization to find the best hyperparameters
2. Save the optimized parameters to `models/optimized_params.json`

You can customize the optimization process with these options:
- `--swarmsize <number>`: Set the number of particles in the swarm (default: 20)
- `--iterations <number>`: Set the number of iterations (default: 10)

For a faster optimization, you can use:
```bash
python run_optimization.py --optimize --swarmsize 5 --iterations 3
```

### Training the Optimized Model

```bash
python run_optimization.py --train
```

This will:
1. Load the optimized parameters from `models/optimized_params.json`
2. Train the optimized CNN model
3. Save the trained model to `models/optimized_dog_cat_cnn_best.pth`

You can customize the training process with this option:
- `--epochs <number>`: Set the number of epochs for training (default: 100)

### Comparing the Models

```bash
python run_optimization.py --compare
```

This will launch the comparison GUI that allows you to:
1. Upload an image
2. View predictions from both the original and optimized models
3. Compare confidence levels and accuracy

### All-in-One Command

```bash
python run_optimization.py --optimize --train --compare
```

This will run the optimization, train the optimized model, and launch the comparison GUI.

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
  - 4 Convolutional blocks with batch normalization and max pooling
  - Dynamic convolution in the third block
  - 3 Fully connected layers with dropout

### Optimized Model

- **Input Shape**: 224x224x3 (RGB images)
- **Output**: 2 classes (Dog, Cat)
- **Layers**:
  - Variable number of convolutional blocks (3-20, optimized by PySwarm)
  - Each block includes Conv2D, BatchNorm2D, ReLU, and MaxPool2D
  - Optimized filter counts for each convolutional layer (3-256)
  - Optimized kernel sizes for each convolutional layer (3×3 to 7×7)
  - Optimized number of neurons in fully connected layers (1-300)
  - Fixed dropout rate of 0.5
  - Optimized learning rate, weight decay, and label smoothing

### Optimization Parameters

- **Swarm Size**: 20 particles (default)
- **Iterations**: 10 (default)
- **Cognitive Parameter (phip)**: 0.5
- **Social Parameter (phig)**: 0.5
- **Constraints**:
  - Number of layers: 3-20
  - Convolutional filter counts: 3-256
  - Kernel sizes: 3×3 to 7×7
  - FC layer neuron counts: 1-300
  - Learning rate: 0.0001-0.001
  - Weight decay: 1e-5 to 1e-3
  - Label smoothing: 0.0-0.2

## Training Metrics

After training, the models will generate plots showing:
- Training and validation loss
- Training and validation accuracy

These plots are saved as:
- `models/training_metrics.png` (original model)
- `models/optimized_training_metrics.png` (optimized model)

## Pre-trained Models

If you don't have data for training, the system will check for pre-trained models:
- Original model: `models/dog_cat_cnn_best.pth`
- Optimized model: `models/optimized_dog_cat_cnn_best.pth`

## Performance Comparison

The comparison GUI allows you to visually compare the performance of the original and optimized models. It displays:
- Predictions from both models
- Confidence levels for both models
- Accuracy statistics

The optimized model typically achieves:
- Higher accuracy
- Better generalization
- Faster inference time (depending on the optimized architecture)

## Conclusion

This project demonstrates the power of PySwarm optimization for finding optimal hyperparameters for CNN architectures. By using Particle Swarm Optimization, we can automatically discover model configurations that outperform manually designed architectures.
