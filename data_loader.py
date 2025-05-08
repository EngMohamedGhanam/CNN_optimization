import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    """Custom dataset for loading dog and cat images"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = ['cats', 'dogs']
        self.class_to_idx = {'cats': 0, 'dogs': 1}

        self.image_paths = []
        self.labels = []

        # Load images from both classes
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)

            return image, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder in case of error
            placeholder = torch.zeros((3, 224, 224))
            return placeholder, torch.tensor(label, dtype=torch.long)

def get_data_loaders(data_dir, batch_size=32, img_size=224, use_test_set=True):
    
# Define transformations with enhanced augmentation
    train_transform = transforms.Compose([
        transforms.Resize((img_size + 30, img_size + 30)),  # Resize larger then crop
        transforms.RandomCrop((img_size, img_size)),  # Random crop for more variation
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),  # Add vertical flips
        transforms.RandomRotation(20),  # Increased rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Add affine transforms
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Add perspective changes
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)  # Random erasing for robustness
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Use the correct folder structure
    train_dir = os.path.join(data_dir, 'training_set', 'training_set')
    test_dir = os.path.join(data_dir, 'test_set', 'test_set')

    # Check if data directories exist and contain images
    if not os.path.exists(train_dir) or not os.path.exists(os.path.join(train_dir, 'dogs')) or not os.path.exists(os.path.join(train_dir, 'cats')):
        print(f"Warning: Training data not found in {train_dir}")
        print("Please make sure the data is in the correct location.")
        return None, None

    # Create training dataset
    try:
        train_dataset = CustomDataset(train_dir, transform=train_transform)

        # Check if dataset is empty
        if len(train_dataset) == 0:
            print("Warning: No images found in training directory.")
            return None, None

        if use_test_set and os.path.exists(test_dir) and os.path.exists(os.path.join(test_dir, 'dogs')) and os.path.exists(os.path.join(test_dir, 'cats')):
            # Use the provided test set
            val_dataset = CustomDataset(test_dir, transform=val_transform)
            if len(val_dataset) == 0:
                print("Warning: No images found in test directory. Splitting training set instead.")
                # Fall back to splitting training set
                train_size = int(0.8 * len(train_dataset))
                val_size = len(train_dataset) - train_size
                train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
                val_dataset.dataset.transform = val_transform
        else:
            # Split training set if test set is not available
            print("Test set not found or incomplete. Splitting training set for validation.")
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
            val_dataset.dataset.transform = val_transform

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        print(f"Successfully loaded {len(train_dataset)} training images and {len(val_dataset)} validation images.")
        return train_loader, val_loader

    except Exception as e:
        print(f"Error creating data loaders: {str(e)}")
        return None, None

def create_folders():
    """Create necessary folders for the project"""
    folders = [
        'data/train/dogs',
        'data/train/cats',
        'data/test/dogs',
        'data/test/cats',
        'models'
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")
