import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicConv2d(nn.Module):
    """Dynamic convolution layer that adapts to input"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DynamicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        out = self.conv(x)
        return out * attention_weights

class DogCatCNN(nn.Module):
    """Improved CNN model for dog and cat classification"""
    def __init__(self, input_shape=(3, 224, 224), dropout_rate=0.5):
        super(DogCatCNN, self).__init__()

        self.input_shape = input_shape

        # First convolutional block - increased filters
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional block - double conv
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third convolutional block - double conv
        self.dyn_conv = DynamicConv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fourth convolutional block - increased filters
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fifth convolutional block - new
        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate size after convolutions and pooling
        # Input: 224x224 -> After 5 pooling layers: 7x7
        conv_output_size = input_shape[1] // (2**5) * input_shape[2] // (2**5) * 512

        # Fully connected layers - deeper
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(conv_output_size, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, 64)
        self.bn_fc3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(dropout_rate/2)  # Less dropout in final layers
        self.fc4 = nn.Linear(64, 2)  # 2 classes: dogs and cats

    def forward(self, x):
        # First block - double conv
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = self.pool1(F.relu(self.bn1_2(self.conv1_2(x))))

        # Second block - double conv
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = self.pool2(F.relu(self.bn2_2(self.conv2_2(x))))

        # Third block with dynamic convolution
        x = F.relu(self.bn3_1(self.dyn_conv(x)))
        x = self.pool3(F.relu(self.bn3_2(self.conv3_2(x))))

        # Fourth block - double conv
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = self.pool4(F.relu(self.bn4_2(self.conv4_2(x))))

        # Fifth block - double conv
        x = F.relu(self.bn5_1(self.conv5_1(x)))
        x = self.pool5(F.relu(self.bn5_2(self.conv5_2(x))))

        # Flatten and fully connected layers
        x = self.flatten(x)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn_fc3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)

        return x
