import os
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F
from torchvision import transforms



from cnn_model import DogCatCNN

class DogCatClassifierGUI:
    """GUI for dog and cat classification using the trained model"""
    def __init__(self, root, model_path, device):
        self.root = root
        self.root.title("Dogs VS Cats Classifier")
        self.root.geometry("800x600")  # Larger window size

        # Configure the style
        self.configure_style()

        # Load model
        self.device = device
        self.model = DogCatCNN()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Class labels
        self.class_labels = ['Cat', 'Dog']

        # Variables
        self.image_path = None
        self.photo = None
        self.background_image = None

        # Load background image
        self.load_background()

        # GUI elements
        self.create_widgets()

    def configure_style(self):
        """Configure the ttk style for the GUI"""
        style = ttk.Style()

        # Configure frame style with a slightly lighter background for contrast
        style.configure(
            "TFrame",
            background="#34495e"  # Slightly lighter blue than the canvas
        )

        # Configure button style
        style.configure(
            "Accent.TButton",
            font=("Arial", 11, "bold"),
            background="#2980b9",
            foreground="#ffffff"
        )

        # Configure label style - larger fonts for better visibility
        style.configure(
            "Title.TLabel",
            font=("Arial", 24, "bold"),  # Large font
            foreground="#ffffff",
            background="#34495e"  # Match the frame background
        )

        style.configure(
            "Subtitle.TLabel",
            font=("Arial", 16),  # Medium font
            foreground="#ffffff",
            background="#34495e"  # Match the frame background
        )

        style.configure(
            "Result.TLabel",
            font=("Arial", 20, "bold"),  # Large font
            foreground="#ffffff",
            background="#34495e"  # Match the frame background
        )

    def load_background(self):
        """Set a simple color background instead of an image"""
        # Don't use image background - just use a simple color
        self.background_image = None
        # Set the root background color to a nice blue gradient
        self.root.configure(background="#2c3e50")
        print("Using simple color background instead of image")

    def create_widgets(self):
        """Create GUI widgets"""
        # Large frame with generous padding
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=40, pady=40)

        # Large title
        self.label_title = ttk.Label(
            self.main_frame,
            text="Dogs VS Cats Classifier",
            style="Title.TLabel"
        )
        self.label_title.pack(pady=20)

        # Clear instructions
        self.label_instructions = ttk.Label(
            self.main_frame,
            text="Upload an image of a dog or cat and click classify",
            style="Subtitle.TLabel"
        )
        self.label_instructions.pack(pady=15)

        # Large image display area
        self.image_frame = tk.Frame(
            self.main_frame,
            bg="#ffffff",  # White background
            highlightbackground="#3498db",  # Blue border
            highlightthickness=2,
            borderwidth=2,
            relief="groove"
        )
        self.image_frame.pack(pady=20)

        # Large image display
        self.label_image = tk.Label(
            self.image_frame,
            bg="#f0f0f0",  # Light gray background
            width=40,  # Large width
            height=20,  # Large height
            text="No image selected",
            font=("Arial", 14)  # Large font
        )
        self.label_image.pack(padx=10, pady=10)

        # Large button layout
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(pady=20)

        # Large upload button
        self.btn_upload = ttk.Button(
            self.button_frame,
            text="Upload Image",
            command=self.upload_image,
            width=20,  # Large button
            style="Accent.TButton"
        )
        self.btn_upload.pack(side="left", padx=20)

        # Large classify button
        self.btn_predict = ttk.Button(
            self.button_frame,
            text="Classify",
            command=self.predict_image,
            width=20,  # Large button
            style="Accent.TButton"
        )
        self.btn_predict.pack(side="right", padx=20)
        self.btn_predict.state(['disabled'])  # Initially disabled

        # Large result display
        self.label_result = ttk.Label(
            self.main_frame,
            text="Result: None",
            style="Result.TLabel"
        )
        self.label_result.pack(pady=20)

        # Confidence display
        self.label_confidence = ttk.Label(
            self.main_frame,
            text="Confidence: 0%",
            style="Subtitle.TLabel"
        )
        self.label_confidence.pack(pady=10)

        # Large progress bar
        self.progress_bar = ttk.Progressbar(
            self.main_frame,
            orient='horizontal',
            length=400,  # Large bar
            mode='determinate',
            style="Horizontal.TProgressbar"
        )
        self.progress_bar.pack(pady=20, fill='x', padx=40)

        # No model info - save space

    # No need for complex resize handling with a simple layout

    def upload_image(self):
        """Upload an image for prediction"""
        self.image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )

        if self.image_path:
            try:
                # Display the image
                image = Image.open(self.image_path)

                # Get original dimensions
                width, height = image.size

                # Calculate new dimensions while maintaining aspect ratio
                max_size = 400  # Large display size
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))

                # Simple resize - faster performance
                image = image.resize((new_width, new_height))

                # Create and display the image
                self.photo = ImageTk.PhotoImage(image)
                self.label_image.configure(image=self.photo)

                # Show image filename
                filename = os.path.basename(self.image_path)
                if len(filename) > 30:
                    filename = filename[:27] + "..."

                # Reset prediction - clear text
                self.label_result.configure(text="Result: None")
                self.progress_bar['value'] = 0
                self.label_confidence.configure(text="Confidence: 0%")

                # Enable predict button
                self.btn_predict.state(['!disabled'])

            except Exception as e:
                self.label_result.configure(text=f"Error loading image: {str(e)}")
                self.label_confidence.configure(text="Please try another image")

    def predict_image(self):
        """Predict the class of the uploaded image"""
        if not self.image_path:
            self.label_result.configure(text="Please upload an image first!")
            return

        try:
            # Load and preprocess the image
            image = Image.open(self.image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
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

            # Update GUI with detailed prediction
            self.label_result.configure(text=f"Result: {predicted_class}")
            self.progress_bar['value'] = confidence_value
            self.label_confidence.configure(
                text=f"Confidence: {confidence_value:.1f}%"
            )

        except Exception as e:
            self.label_result.configure(text="Error during classification")
            self.label_confidence.configure(text="Please try another image")

def launch_gui(model_path, device):
    """Launch the GUI with the trained model"""
    root = tk.Tk()
    # Create the app and store it in root to prevent garbage collection
    root.app = DogCatClassifierGUI(root, model_path, device)
    root.mainloop()

if __name__ == "__main__":
    import sys
    import torch

    # Default model path
    model_path = "models/dog_cat_cnn_best.pth"

    # Check if model path is provided as argument
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    # Check if model exists
    import os
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please provide a valid model path or train a model first.")
        sys.exit(1)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Launch GUI
    print(f"Launching GUI with model: {model_path}")
    launch_gui(model_path, device)

