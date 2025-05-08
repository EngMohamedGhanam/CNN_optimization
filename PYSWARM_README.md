# PySwarm Optimization for CNN Model

This extension adds PySwarm optimization to the CNN model for dog and cat classification. The PySwarm optimization algorithm is used to find the optimal values for:

- Cognitive Coefficient (phip)
- Social Coefficient (phig)
- Swarm Size
- Number of layers
- Number of neurons per layer

## Files

- `cnn_model_pyswarm.py`: PySwarm-optimized CNN model
- `pyswarm_optimize.py`: PySwarm optimization script
- `train_pyswarm_model.py`: Training script for PySwarm-optimized model
- `pyswarm_config_gui.py`: GUI for configuring and running PySwarm optimization with image classification

## Usage

### Using the Configuration GUI

```bash
python pyswarm_config_gui.py
```

This will launch the PySwarm Configuration GUI, which allows you to:
1. Configure PySwarm optimization parameters
2. Configure CNN model parameters
3. Run PySwarm optimization
4. Upload and classify images using the trained model

The optimized parameters will be saved to `models/pyswarm_params.json`, and the trained model will be saved to `models/pyswarm_dog_cat_cnn_best.pth`.

## Parameters Optimized by PySwarm

- `num_layers`: Number of convolutional layers (3-10)
- `fc_neurons_1`: Number of neurons in the first fully connected layer (32-256)
- `fc_neurons_2`: Number of neurons in the second fully connected layer (16-128)
- `fc_neurons_3`: Number of neurons in the third fully connected layer (8-64)
- `batch_size`: Batch size for training (8-64)

## How PySwarm Optimization Works

1. The PySwarm algorithm initializes a swarm of particles with random positions in the parameter space.
2. Each particle represents a set of parameters for the CNN model.
3. The fitness function evaluates the performance of each particle by training the CNN model with the corresponding parameters for a single epoch.
4. The particles move in the parameter space based on their own best position (cognitive component) and the swarm's best position (social component).
5. The algorithm iterates until the maximum number of iterations is reached or the fitness function converges.
6. The best parameters found by the algorithm are saved and used to train the final model.

## Requirements

- PyTorch
- PySwarm
- Matplotlib
- Pillow
- tqdm

## Note

Make sure to train the model before using the image classification functionality.

## Image Classification GUI

The PySwarm Configuration GUI now includes an image classification tab that allows you to:

1. Upload an image from your computer
2. Classify the image as either a dog or a cat using the PySwarm-optimized model
3. View the classification result with a confidence percentage

### Using the Image Classification Tab

1. Launch the GUI by running `python pyswarm_config_gui.py`
2. Click on the "Image Classification" tab
3. Click the "Upload Image" button to select an image from your computer
4. Click the "Classify Image" button to run the classification
5. View the result and confidence percentage in the "Classification Result" section

The GUI will automatically try to load the best available model in the following order:
- PySwarm-optimized model (`pyswarm_dog_cat_cnn_best.pth`)
- PySwarm-optimized final model (`pyswarm_dog_cat_cnn_final.pth`)
- Regular CNN model (`dog_cat_cnn_best.pth`)
- Regular CNN final model (`dog_cat_cnn_final.pth`)

If none of these models are found, an error message will be displayed.
