"""
MEMBER 1: CNN from Scratch Implementation

This module contains the custom CNN architecture for pneumonia classification.

Author: Member 1
Date: 2026-11-20
Description:
    Implements a 5-block convolutional neural network from scratch using PyTorch.
    Includes batch normalization, dropout regularization, and adaptive pooling.
    Performance depends on training and evaluation; no score is guaranteed.

Key Learning Points:
    1. Convolutional layers extract spatial features
    2. Batch normalization stabilizes training
    3. Dropout prevents overfitting
    4. Early stopping prevents unnecessary training
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class CustomCNN(nn.Module):
    """
    Custom CNN architecture built from scratch for binary image classification.

    Architecture:
        - 5 convolutional blocks with increasing filter sizes (32→64→128→256→512)
        - Each block: 2 Conv → BatchNorm → ReLU → MaxPool → Dropout
        - Global average pooling before classification head
        - Classification head: FC layers with dropout

    Parameters:
        num_blocks (int): Number of conv blocks (default: 5)
        dropout_rate (float): Dropout probability (default: 0.3)

    Example:
        >>> model = CustomCNN(num_blocks=5, dropout_rate=0.3)
        >>> x = torch.randn(32, 3, 256, 256)  # batch of 32 images
        >>> output = model(x)  # (32, 2) - logits for binary classification
    """

    def __init__(self, num_blocks=5, dropout_rate=0.3):
        super(CustomCNN, self).__init__()

        # Convolutional feature extraction blocks
        self.features = nn.Sequential(
            # Block 1: 3 → 32 filters
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 256 → 128
            nn.Dropout(dropout_rate),
            # Block 2: 32 → 64 filters
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128 → 64
            nn.Dropout(dropout_rate),
            # Block 3: 64 → 128 filters
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64 → 32
            nn.Dropout(dropout_rate),
            # Block 4: 128 → 256 filters
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 → 16
            nn.Dropout(dropout_rate),
            # Block 5: 256 → 512 filters
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 → 8
            nn.Dropout(dropout_rate),
        )

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 2),  # Binary classification: NORMAL (0) or PNEUMONIA (1)
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 256, 256)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, 2)
        """
        # Feature extraction
        x = self.features(x)

        # Global average pooling
        x = self.avgpool(x)

        # Flatten for fully connected layers
        x = torch.flatten(x, 1)

        # Classification
        x = self.classifier(x)

        return x


def create_cnn_model(device, num_blocks=5, dropout_rate=0.3):
    """
    Create and initialize CNN model.

    Args:
        device (torch.device): Device to place model on (cpu or cuda)
        num_blocks (int): Number of convolutional blocks
        dropout_rate (float): Dropout probability

    Returns:
        nn.Module: CNN model on specified device
    """
    model = CustomCNN(num_blocks=num_blocks, dropout_rate=dropout_rate)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✓ CNN Model Created")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")

    return model


def get_cnn_optimizer(model, learning_rate=0.001):
    """
    Create optimizer for CNN training.

    Args:
        model (nn.Module): CNN model
        learning_rate (float): Learning rate

    Returns:
        torch.optim.Optimizer: Adam optimizer with weight decay
    """
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    return optimizer


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_cnn_model(device)

    # Test forward pass
    test_input = torch.randn(2, 3, 256, 256).to(device)
    output = model(test_input)
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"✓ Model test passed!")
