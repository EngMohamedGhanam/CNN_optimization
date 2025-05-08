import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DynamicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super(DynamicConv2d, self).__init__()
        # Calculate padding if not provided (to maintain spatial dimensions)
        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(in_channels // 8, 1), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(in_channels // 8, 1), out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        out = self.conv(x)
        return out * attention_weights

class PySwarmOptimizedCNN(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), params=None):
        super(PySwarmOptimizedCNN, self).__init__()

        # Default parameters if none provided
        if params is None:
            params = {
                'num_layers': 5,                # Number of layers (3-20)
                'conv_filters': [32, 64, 128, 256, 512],  # Filters per layer (3-256)
                'kernel_sizes': [3, 3, 3, 3, 3],  # Kernel sizes (3-7)
                'fc_neurons': [1024, 256, 64],  # FC layer neurons (1-300)
                'dropout_rate': 0.5,            # Dropout rate
                'learning_rate': 0.0005,        # Learning rate
                'weight_decay': 1e-4,           # Weight decay
                'label_smoothing': 0.1,         # Label smoothing
                'batch_size': 32                # Batch size
            }

        self.params = params
        self.input_shape = input_shape

        # Extract parameters with constraints
        dropout_rate = params.get('dropout_rate', 0.5)
        num_layers = params.get('num_layers', 5)

        # Calculate maximum possible layers based on input size
        # Each pooling layer reduces size by factor of 2
        # We need at least 2x2 size before the final pooling
        max_possible_layers = min(
            int(math.log2(input_shape[1])) - 1,  # Based on width
            int(math.log2(input_shape[2])) - 1   # Based on height
        )

        # Ensure num_layers is within constraints (3-20) and doesn't exceed possible layers
        num_layers = max(3, min(20, min(max_possible_layers, num_layers)))
        print(f"Using {num_layers} layers (max possible: {max_possible_layers})")

        # Get convolutional filters with constraints (3-256)
        conv_filters = params.get('conv_filters', [32, 64, 128, 256, 512])
        # Ensure each filter count is within constraints
        conv_filters = [max(3, min(256, f)) for f in conv_filters[:num_layers]]
        # Pad or truncate to match num_layers
        if len(conv_filters) < num_layers:
            conv_filters.extend([64] * (num_layers - len(conv_filters)))
        conv_filters = conv_filters[:num_layers]

        # Get kernel sizes with constraints (3x3 to 7x7)
        kernel_sizes = params.get('kernel_sizes', [3] * num_layers)
        # Ensure each kernel size is within constraints
        kernel_sizes = [max(3, min(7, k)) for k in kernel_sizes[:num_layers]]
        # Pad or truncate to match num_layers
        if len(kernel_sizes) < num_layers:
            kernel_sizes.extend([3] * (num_layers - len(kernel_sizes)))
        kernel_sizes = kernel_sizes[:num_layers]

        # Get FC neurons with constraints (1-300)
        fc_neurons = params.get('fc_neurons', [1024, 256, 64])
        # Ensure each FC neuron count is within constraints
        fc_neurons = [max(1, min(300, n)) for n in fc_neurons]

        # Create convolutional layers dynamically
        self.features = nn.ModuleList()
        in_channels = 3  # RGB input

        for i in range(num_layers):
            # Add convolutional block
            layer_block = nn.Sequential()

            # First conv in the block
            layer_block.add_module(
                f'conv{i+1}_1',
                nn.Conv2d(
                    in_channels,
                    conv_filters[i],
                    kernel_size=kernel_sizes[i],
                    padding=kernel_sizes[i]//2
                )
            )
            layer_block.add_module(f'bn{i+1}_1', nn.BatchNorm2d(conv_filters[i]))
            layer_block.add_module(f'relu{i+1}_1', nn.ReLU(inplace=True))

            # Second conv in the block (for deeper layers)
            if i >= 1:  # Add second conv for deeper layers
                layer_block.add_module(
                    f'conv{i+1}_2',
                    nn.Conv2d(
                        conv_filters[i],
                        conv_filters[i],
                        kernel_size=kernel_sizes[i],
                        padding=kernel_sizes[i]//2
                    )
                )
                layer_block.add_module(f'bn{i+1}_2', nn.BatchNorm2d(conv_filters[i]))
                layer_block.add_module(f'relu{i+1}_2', nn.ReLU(inplace=True))

            # Add pooling
            layer_block.add_module(f'pool{i+1}', nn.MaxPool2d(kernel_size=2, stride=2))

            # Add to features
            self.features.append(layer_block)

            # Update in_channels for next layer
            in_channels = conv_filters[i]

        # Calculate output size after all convolutional layers
        output_size = input_shape[1] // (2**num_layers)  # Width
        output_size *= input_shape[2] // (2**num_layers)  # Height
        output_size *= conv_filters[-1]  # Channels

        # Create classifier (FC layers)
        self.classifier = nn.ModuleList()

        # Input to first FC layer
        in_features = output_size

        # Add FC layers
        for i, neurons in enumerate(fc_neurons):
            fc_block = nn.Sequential()

            # Linear layer
            fc_block.add_module(f'fc{i+1}', nn.Linear(in_features, neurons))
            fc_block.add_module(f'bn_fc{i+1}', nn.BatchNorm1d(neurons))
            fc_block.add_module(f'relu_fc{i+1}', nn.ReLU(inplace=True))

            # Add dropout (except for last layer)
            if i < len(fc_neurons) - 1:
                fc_block.add_module(f'dropout{i+1}', nn.Dropout(dropout_rate))
            else:
                fc_block.add_module(f'dropout{i+1}', nn.Dropout(dropout_rate/2))  # Less dropout in final layer

            # Add to classifier
            self.classifier.append(fc_block)

            # Update in_features for next layer
            in_features = neurons

        # Final classification layer
        self.fc_final = nn.Linear(fc_neurons[-1], 2)  # 2 classes: dogs and cats

        # Flatten layer
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Set model to eval mode during inference to avoid batch norm issues with batch size 1
        training = self.training

        # Pass through all feature layers
        for layer in self.features:
            x = layer(x)

        # Flatten
        x = self.flatten(x)

        # Handle batch size of 1 during inference
        if not training and x.size(0) == 1:
            # Temporarily set all batch norm layers to eval mode
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()

        # Pass through classifier layers
        for layer in self.classifier:
            x = layer(x)

        # Final classification
        x = self.fc_final(x)

        # Restore original training state if needed
        if not training and x.size(0) == 1:
            self.train(training)

        return x
