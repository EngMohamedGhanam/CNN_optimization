"""
PySwarm Configuration GUI
A simple interface to configure and run PySwarm optimization for CNN models
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import torch
import torch.nn.functional as F
import numpy as np
from pyswarm import pso
from PIL import Image, ImageTk
import torchvision.transforms as transforms

# Import project modules
from cnn_model_pyswarm import PySwarmOptimizedCNN
from train_pyswarm_model import train_pyswarm_model

class PySwarmConfigGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PySwarm Optimization Configuration")
        self.root.geometry("700x700")

        # Default parameters
        self.default_params = {
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

        # PySwarm parameters
        self.swarm_size = tk.IntVar(value=10)
        self.max_iterations = tk.IntVar(value=20)
        self.min_epochs = tk.IntVar(value=5)
        self.max_epochs = tk.IntVar(value=30)

        # CNN parameters
        self.num_layers = tk.IntVar(value=self.default_params['num_layers'])
        self.dropout_rate = tk.DoubleVar(value=self.default_params['dropout_rate'])
        self.learning_rate = tk.DoubleVar(value=self.default_params['learning_rate'])
        self.weight_decay = tk.DoubleVar(value=self.default_params['weight_decay'])
        self.label_smoothing = tk.DoubleVar(value=self.default_params['label_smoothing'])
        self.batch_size = tk.IntVar(value=self.default_params['batch_size'])

        # Image classification variables
        self.image_path = None
        self.image_tk = None
        self.model = None
        self.transform = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create the GUI
        self.create_widgets()

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="PySwarm CNN Optimization", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        # Create notebook (tabs)
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10)

        # PySwarm parameters tab
        swarm_frame = ttk.Frame(notebook, padding=10)
        notebook.add(swarm_frame, text="PySwarm Parameters")

        # CNN parameters tab
        cnn_frame = ttk.Frame(notebook, padding=10)
        notebook.add(cnn_frame, text="CNN Parameters")

        # Image Classification tab
        classify_frame = ttk.Frame(notebook, padding=10)
        notebook.add(classify_frame, text="Image Classification")

        # Fill PySwarm parameters tab
        self.create_swarm_parameters(swarm_frame)

        # Fill CNN parameters tab
        self.create_cnn_parameters(cnn_frame)

        # Fill Image Classification tab
        self.create_classification_tab(classify_frame)

        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        # Reset button
        reset_button = ttk.Button(button_frame, text="Reset to Defaults", command=self.reset_to_defaults)
        reset_button.pack(side=tk.LEFT, padx=5)

        # Save config button
        save_button = ttk.Button(button_frame, text="Save Configuration", command=self.save_configuration)
        save_button.pack(side=tk.LEFT, padx=5)

        # Load config button
        load_button = ttk.Button(button_frame, text="Load Configuration", command=self.load_configuration)
        load_button.pack(side=tk.LEFT, padx=5)

        # Run optimization button
        run_button = ttk.Button(button_frame, text="Run Optimization", command=self.run_optimization)
        run_button.pack(side=tk.RIGHT, padx=5)

        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding=10)
        status_frame.pack(fill=tk.X, pady=10)

        # Status label
        self.status_var = tk.StringVar(value="Ready to start optimization")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, wraplength=550)
        status_label.pack(fill=tk.X)

    def create_swarm_parameters(self, parent):
        # Swarm size
        swarm_size_frame = ttk.Frame(parent)
        swarm_size_frame.pack(fill=tk.X, pady=5)

        swarm_size_label = ttk.Label(swarm_size_frame, text="Swarm Size:", width=20)
        swarm_size_label.pack(side=tk.LEFT)

        swarm_size_entry = ttk.Entry(swarm_size_frame, textvariable=self.swarm_size, width=10)
        swarm_size_entry.pack(side=tk.LEFT, padx=5)

        swarm_size_info = ttk.Label(swarm_size_frame, text="Number of particles in the swarm")
        swarm_size_info.pack(side=tk.LEFT, padx=5)

        # Max iterations
        max_iter_frame = ttk.Frame(parent)
        max_iter_frame.pack(fill=tk.X, pady=5)

        max_iter_label = ttk.Label(max_iter_frame, text="Max Iterations:", width=20)
        max_iter_label.pack(side=tk.LEFT)

        max_iter_entry = ttk.Entry(max_iter_frame, textvariable=self.max_iterations, width=10)
        max_iter_entry.pack(side=tk.LEFT, padx=5)

        max_iter_info = ttk.Label(max_iter_frame, text="Maximum number of iterations")
        max_iter_info.pack(side=tk.LEFT, padx=5)

        # Min epochs
        min_epochs_frame = ttk.Frame(parent)
        min_epochs_frame.pack(fill=tk.X, pady=5)

        min_epochs_label = ttk.Label(min_epochs_frame, text="Min Epochs:", width=20)
        min_epochs_label.pack(side=tk.LEFT)

        min_epochs_entry = ttk.Entry(min_epochs_frame, textvariable=self.min_epochs, width=10)
        min_epochs_entry.pack(side=tk.LEFT, padx=5)

        min_epochs_info = ttk.Label(min_epochs_frame, text="Minimum epochs for each model evaluation")
        min_epochs_info.pack(side=tk.LEFT, padx=5)

        # Max epochs
        max_epochs_frame = ttk.Frame(parent)
        max_epochs_frame.pack(fill=tk.X, pady=5)

        max_epochs_label = ttk.Label(max_epochs_frame, text="Max Epochs:", width=20)
        max_epochs_label.pack(side=tk.LEFT)

        max_epochs_entry = ttk.Entry(max_epochs_frame, textvariable=self.max_epochs, width=10)
        max_epochs_entry.pack(side=tk.LEFT, padx=5)

        max_epochs_info = ttk.Label(max_epochs_frame, text="Maximum epochs for final model training")
        max_epochs_info.pack(side=tk.LEFT, padx=5)

    def create_cnn_parameters(self, parent):
        # Number of layers
        num_layers_frame = ttk.Frame(parent)
        num_layers_frame.pack(fill=tk.X, pady=5)

        num_layers_label = ttk.Label(num_layers_frame, text="Number of Layers:", width=20)
        num_layers_label.pack(side=tk.LEFT)

        num_layers_slider = ttk.Scale(num_layers_frame, from_=3, to=10,
                                      variable=self.num_layers, orient=tk.HORIZONTAL)
        num_layers_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        num_layers_value = ttk.Label(num_layers_frame, textvariable=self.num_layers, width=5)
        num_layers_value.pack(side=tk.LEFT)

        # Dropout rate
        dropout_frame = ttk.Frame(parent)
        dropout_frame.pack(fill=tk.X, pady=5)

        dropout_label = ttk.Label(dropout_frame, text="Dropout Rate:", width=20)
        dropout_label.pack(side=tk.LEFT)

        dropout_slider = ttk.Scale(dropout_frame, from_=0.1, to=0.9,
                                  variable=self.dropout_rate, orient=tk.HORIZONTAL)
        dropout_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        dropout_value = ttk.Label(dropout_frame, textvariable=self.dropout_rate, width=5)
        dropout_value.pack(side=tk.LEFT)

        # Learning rate
        lr_frame = ttk.Frame(parent)
        lr_frame.pack(fill=tk.X, pady=5)

        lr_label = ttk.Label(lr_frame, text="Learning Rate:", width=20)
        lr_label.pack(side=tk.LEFT)

        lr_slider = ttk.Scale(lr_frame, from_=0.0001, to=0.01,
                             variable=self.learning_rate, orient=tk.HORIZONTAL)
        lr_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        lr_value = ttk.Label(lr_frame, textvariable=self.learning_rate, width=5)
        lr_value.pack(side=tk.LEFT)

        # Weight decay
        wd_frame = ttk.Frame(parent)
        wd_frame.pack(fill=tk.X, pady=5)

        wd_label = ttk.Label(wd_frame, text="Weight Decay:", width=20)
        wd_label.pack(side=tk.LEFT)

        wd_slider = ttk.Scale(wd_frame, from_=1e-6, to=1e-3,
                             variable=self.weight_decay, orient=tk.HORIZONTAL)
        wd_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        wd_value = ttk.Label(wd_frame, textvariable=self.weight_decay, width=5)
        wd_value.pack(side=tk.LEFT)

        # Label smoothing
        ls_frame = ttk.Frame(parent)
        ls_frame.pack(fill=tk.X, pady=5)

        ls_label = ttk.Label(ls_frame, text="Label Smoothing:", width=20)
        ls_label.pack(side=tk.LEFT)

        ls_slider = ttk.Scale(ls_frame, from_=0.0, to=0.2,
                             variable=self.label_smoothing, orient=tk.HORIZONTAL)
        ls_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        ls_value = ttk.Label(ls_frame, textvariable=self.label_smoothing, width=5)
        ls_value.pack(side=tk.LEFT)

        # Batch size
        bs_frame = ttk.Frame(parent)
        bs_frame.pack(fill=tk.X, pady=5)

        bs_label = ttk.Label(bs_frame, text="Batch Size:", width=20)
        bs_label.pack(side=tk.LEFT)

        bs_slider = ttk.Scale(bs_frame, from_=8, to=128,
                             variable=self.batch_size, orient=tk.HORIZONTAL)
        bs_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        bs_value = ttk.Label(bs_frame, textvariable=self.batch_size, width=5)
        bs_value.pack(side=tk.LEFT)

    def reset_to_defaults(self):
        """Reset all parameters to default values"""
        self.swarm_size.set(10)
        self.max_iterations.set(20)
        self.min_epochs.set(5)
        self.max_epochs.set(30)

        self.num_layers.set(self.default_params['num_layers'])
        self.dropout_rate.set(self.default_params['dropout_rate'])
        self.learning_rate.set(self.default_params['learning_rate'])
        self.weight_decay.set(self.default_params['weight_decay'])
        self.label_smoothing.set(self.default_params['label_smoothing'])
        self.batch_size.set(self.default_params['batch_size'])

        self.status_var.set("Parameters reset to defaults")

    def save_configuration(self):
        """Save current configuration to a JSON file"""
        config = {
            'swarm_size': self.swarm_size.get(),
            'max_iterations': self.max_iterations.get(),
            'min_epochs': self.min_epochs.get(),
            'max_epochs': self.max_epochs.get(),
            'num_layers': self.num_layers.get(),
            'dropout_rate': self.dropout_rate.get(),
            'learning_rate': self.learning_rate.get(),
            'weight_decay': self.weight_decay.get(),
            'label_smoothing': self.label_smoothing.get(),
            'batch_size': self.batch_size.get()
        }

        try:
            os.makedirs('configs', exist_ok=True)
            with open('configs/pyswarm_config.json', 'w') as f:
                json.dump(config, f, indent=4)
            self.status_var.set("Configuration saved to configs/pyswarm_config.json")
        except Exception as e:
            self.status_var.set(f"Error saving configuration: {str(e)}")

    def load_configuration(self):
        """Load configuration from a JSON file"""
        try:
            with open('configs/pyswarm_config.json', 'r') as f:
                config = json.load(f)

            self.swarm_size.set(config.get('swarm_size', 10))
            self.max_iterations.set(config.get('max_iterations', 20))
            self.min_epochs.set(config.get('min_epochs', 5))
            self.max_epochs.set(config.get('max_epochs', 30))

            self.num_layers.set(config.get('num_layers', self.default_params['num_layers']))
            self.dropout_rate.set(config.get('dropout_rate', self.default_params['dropout_rate']))
            self.learning_rate.set(config.get('learning_rate', self.default_params['learning_rate']))
            self.weight_decay.set(config.get('weight_decay', self.default_params['weight_decay']))
            self.label_smoothing.set(config.get('label_smoothing', self.default_params['label_smoothing']))
            self.batch_size.set(config.get('batch_size', self.default_params['batch_size']))

            self.status_var.set("Configuration loaded from configs/pyswarm_config.json")
        except FileNotFoundError:
            self.status_var.set("Configuration file not found. Using defaults.")
        except Exception as e:
            self.status_var.set(f"Error loading configuration: {str(e)}")

    def run_optimization(self):
        """Run PySwarm optimization with current parameters"""
        # Confirm with user
        result = messagebox.askyesno(
            "Confirm Optimization",
            "Running PySwarm optimization may take a long time. Continue?"
        )

        if not result:
            return

        # Get current parameters
        params = {
            'num_layers': self.num_layers.get(),
            'dropout_rate': self.dropout_rate.get(),
            'learning_rate': self.learning_rate.get(),
            'weight_decay': self.weight_decay.get(),
            'label_smoothing': self.label_smoothing.get(),
            'batch_size': self.batch_size.get(),
            'swarm_size': self.swarm_size.get(),
            'max_iterations': self.max_iterations.get(),
            'min_epochs': self.min_epochs.get(),
            'max_epochs': self.max_epochs.get()
        }

        # Save parameters
        os.makedirs('models', exist_ok=True)
        with open('models/pyswarm_params.json', 'w') as f:
            json.dump(params, f, indent=4)

        # Update status
        self.status_var.set("Starting PySwarm optimization. This may take a while...")
        self.root.update()

        # Run optimization in a separate thread to avoid freezing the GUI
        import threading
        thread = threading.Thread(target=self._run_optimization_thread, args=(params,))
        thread.daemon = True
        thread.start()

    def _run_optimization_thread(self, params):
        """Run optimization in a separate thread"""
        try:
            # Import here to avoid circular imports
            from pyswarm_optimize import optimize_model

            # Run optimization
            optimize_model(
                swarm_size=params['swarm_size'],
                max_iter=params['max_iterations'],
                min_epochs=params['min_epochs'],
                max_epochs=params['max_epochs']
            )

            # Update status when done
            self.status_var.set("PySwarm optimization completed successfully!")
        except Exception as e:
            self.status_var.set(f"Error during optimization: {str(e)}")

    def create_classification_tab(self, parent):
        """Create the image classification tab"""
        # Create frames
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill=tk.X, pady=10)

        image_frame = ttk.LabelFrame(parent, text="Image", padding=10)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        result_frame = ttk.LabelFrame(parent, text="Classification Result", padding=10)
        result_frame.pack(fill=tk.X, pady=10)

        # Upload button
        upload_button = ttk.Button(top_frame, text="Upload Image", command=self.upload_image)
        upload_button.pack(side=tk.LEFT, padx=5)

        # Classify button
        self.classify_button = ttk.Button(top_frame, text="Classify Image", command=self.classify_image, state='disabled')
        self.classify_button.pack(side=tk.LEFT, padx=5)

        # Image display
        self.image_label = ttk.Label(image_frame, text="No image uploaded")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Result labels
        result_label_frame = ttk.Frame(result_frame)
        result_label_frame.pack(fill=tk.X, pady=5)

        ttk.Label(result_label_frame, text="Result:", width=15).pack(side=tk.LEFT)
        self.result_var = tk.StringVar(value="None")
        ttk.Label(result_label_frame, textvariable=self.result_var).pack(side=tk.LEFT)

        # Confidence progress bar
        confidence_frame = ttk.Frame(result_frame)
        confidence_frame.pack(fill=tk.X, pady=5)

        ttk.Label(confidence_frame, text="Confidence:", width=15).pack(side=tk.LEFT)
        self.progress_bar = ttk.Progressbar(confidence_frame, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.confidence_var = tk.StringVar(value="0%")
        ttk.Label(confidence_frame, textvariable=self.confidence_var, width=8).pack(side=tk.LEFT)

    def upload_image(self):
        """Upload an image for classification"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )

        if not file_path:
            return

        try:
            # Save the image path
            self.image_path = file_path

            # Load and display the image
            image = Image.open(file_path)
            image = image.resize((300, 300))  # Use default resampling method
            self.image_tk = ImageTk.PhotoImage(image)

            self.image_label.config(image=self.image_tk, text="")

            # Enable classify button
            self.classify_button.state(['!disabled'])

            # Reset result
            self.result_var.set("None")
            self.progress_bar['value'] = 0
            self.confidence_var.set("0%")

            self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
        except Exception as e:
            self.status_var.set(f"Error loading image: {str(e)}")

    def classify_image(self):
        """Classify the uploaded image as dog or cat"""
        if not self.image_path:
            self.status_var.set("Please upload an image first")
            return

        try:
            # Load model if not already loaded
            if self.model is None:
                self.load_model()

            # Preprocess image
            image = Image.open(self.image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Make prediction
            with torch.no_grad():
                try:
                    outputs = self.model(image_tensor)

                    # Check output shape to determine how to handle it
                    if outputs.shape[1] == 2:  # Binary classification (cat/dog)
                        probabilities = F.softmax(outputs, dim=1)

                        # Get probabilities for both classes
                        cat_prob = probabilities[0][0].item() * 100
                        dog_prob = probabilities[0][1].item() * 100

                        # Determine predicted class
                        if cat_prob > dog_prob:
                            predicted_class = "Cat"
                            confidence_value = cat_prob
                        else:
                            predicted_class = "Dog"
                            confidence_value = dog_prob
                    else:
                        # Handle multi-class output (more than 2 classes)
                        probabilities = F.softmax(outputs, dim=1)
                        confidence, class_idx = torch.max(probabilities, 1)
                        confidence_value = confidence.item() * 100

                        # Map class index to label (assuming 0=cat, 1=dog for simplicity)
                        class_idx = class_idx.item()
                        predicted_class = "Cat" if class_idx == 0 else "Dog"

                        # Print debug info
                        print(f"Multi-class output detected. Shape: {outputs.shape}")
                        print(f"Class index: {class_idx}, Confidence: {confidence_value:.2f}%")

                except RuntimeError as e:
                    self.status_var.set(f"Error during model inference: {str(e)}")
                    print(f"Model inference error: {str(e)}")
                    return

            # Update GUI with prediction
            self.result_var.set(predicted_class)
            self.progress_bar['value'] = confidence_value
            self.confidence_var.set(f"{confidence_value:.1f}%")

            self.status_var.set(f"Classification complete: {predicted_class} with {confidence_value:.1f}% confidence")
        except Exception as e:
            self.status_var.set(f"Error during classification: {str(e)}")
            print(f"Classification error: {str(e)}")

            # Reset result display
            self.result_var.set("Error")
            self.progress_bar['value'] = 0
            self.confidence_var.set("0%")

    def load_model(self):
        """Load the PySwarm optimized model"""
        try:
            # Define image transformation
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            # Try different model paths in order of preference
            model_paths = [
                'D:/venv/models/pyswarm_dog_cat_cnn_best.pth',
                'D:/venv/models/pyswarm_dog_cat_cnn_final.pth',
                'D:/venv/models/dog_cat_cnn_best.pth',
                'D:/venv/models/dog_cat_cnn_final.pth'
            ]

            # Find the first available model
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    self.status_var.set(f"Found model at {path}")
                    break

            # Check if any model exists
            if model_path is None:
                self.status_var.set(f"No model found in any of the expected locations")
                raise FileNotFoundError(f"No model found in any of the expected locations")

            # Load the model parameters
            try:
                # First try to load parameters from JSON if available
                params_path = 'D:/venv/models/pyswarm_params.json'
                if os.path.exists(params_path):
                    with open(params_path, 'r') as f:
                        params = json.load(f)
                    self.model = PySwarmOptimizedCNN(params=params)
                else:
                    # Use default parameters if JSON not found
                    self.model = PySwarmOptimizedCNN()

                # Load the model state dict
                state_dict = torch.load(model_path, map_location=self.device)

                # Try to load with strict=False to ignore missing keys
                self.model.load_state_dict(state_dict, strict=False)
                self.model.to(self.device)
                self.model.eval()

                self.status_var.set("Model loaded successfully")
            except Exception as e:
                # If that fails, try loading as a regular CNN model
                from cnn_model import DogCatCNN
                self.model = DogCatCNN()
                self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
                self.model.to(self.device)
                self.model.eval()
                self.status_var.set("Loaded alternative model successfully")
        except Exception as e:
            self.status_var.set(f"Error loading model: {str(e)}")
            print(f"Model loading error: {str(e)}")
            raise

def main():
    root = tk.Tk()
    app = PySwarmConfigGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()



