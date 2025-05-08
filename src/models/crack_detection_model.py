"""
Crack detection model for structural health monitoring.

This module defines CNN architectures for crack detection in structural images.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models
from typing import Dict, Tuple, Optional, List, Union

class CrackDetectionCNN(nn.Module):
    """Custom CNN architecture for crack detection."""
    
    def __init__(self, num_classes=2, pretrained=True, input_channels=3):
        """
        Initialize the model.
        
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
            input_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
        """
        super(CrackDetectionCNN, self).__init__()
        
        # Load pretrained ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Modify the first layer to accept the specified number of input channels
        if input_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Replace the final fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        return self.backbone(x)

class DroneAwareCNN(nn.Module):
    """CNN architecture designed for both standard and drone imagery."""
    
    def __init__(self, num_classes=2, pretrained=True):
        """
        Initialize the model.
        
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
        """
        super(DroneAwareCNN, self).__init__()
        
        # Use a pretrained ResNet as backbone
        self.backbone = models.resnet34(pretrained=pretrained)
        
        # No need to modify first layer - we'll convert all images to RGB
        
        # Replace the final fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Additional layers for attention to focus on drone-specific features
        self.attention = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Forward pass with attention mechanism.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Get features from early layers
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        # Process through backbone layers with residual connections
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        features = self.backbone.layer4(x)
        
        # Apply attention (optional branch)
        # attention_map = self.attention(features)
        # attended_features = features * attention_map
        
        # Global average pooling
        x = self.backbone.avgpool(features)
        x = torch.flatten(x, 1)
        
        # Classification head
        x = self.backbone.fc(x)
        
        return x

class MobileNetCrackDetector(nn.Module):
    """
    MobileNet-based model for crack detection in structural images.
    
    This model is designed to be lightweight for possible deployment on mobile devices.
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        """
        Initialize the model with a pre-trained MobileNet backbone.
        
        Args:
            num_classes: Number of output classes (default: 2 for binary classification)
            pretrained: Whether to use pre-trained weights for the backbone
        """
        super(MobileNetCrackDetector, self).__init__()
        
        # Use MobileNetV3 as the backbone
        self.backbone = models.mobilenet_v3_small(weights='DEFAULT' if pretrained else None)
        
        # Replace the final classifier
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        """Forward pass through the network."""
        return self.backbone(x)

class EfficientNetCrackDetector(nn.Module):
    """EfficientNet-based model for crack detection."""
    
    def __init__(self, num_classes=2, pretrained=True):
        """
        Initialize the model.
        
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
        """
        super(EfficientNetCrackDetector, self).__init__()
        
        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # Modify the first layer to accept single-channel images
        self.backbone.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Replace the classifier
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        return self.backbone(x)

def get_model(model_name="cnn", num_classes=2, pretrained=True, **kwargs):
    """
    Get a model for crack detection.
    
    Args:
        model_name (str): Name of the model architecture
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        **kwargs: Additional arguments for model initialization
        
    Returns:
        nn.Module: PyTorch model
    """
    # Check if we're using drone images
    input_channels = kwargs.get('input_channels', 3)
    
    if model_name.lower() == "cnn":
        return CrackDetectionCNN(num_classes=num_classes, pretrained=pretrained, input_channels=input_channels)
    elif model_name.lower() == "drone_cnn":
        return DroneAwareCNN(num_classes=num_classes, pretrained=pretrained)
    elif model_name.lower() == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        if input_channels != 3:
            model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Replace final fully connected layer
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        return model
    elif model_name.lower() == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        if input_channels != 3:
            model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Replace final fully connected layer
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        return model
    elif model_name.lower() == "efficientnet":
        model = models.efficientnet_b0(pretrained=pretrained)
        if input_channels != 3:
            model.features[0][0] = nn.Conv2d(
                input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
            )
        # Replace final classifier
        num_features = model.classifier[1].in_features
        model.classifier = nn.Linear(num_features, num_classes)
        return model
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
    return model

def load_model(model_path, model_name="cnn", num_classes=2, device="cuda"):
    """
    Load a trained model from a checkpoint.
    
    Args:
        model_path (str): Path to the model checkpoint
        model_name (str): Name of the model architecture
        num_classes (int): Number of output classes
        device (str): Device to load the model on
        
    Returns:
        nn.Module: Loaded model
    """
    # Create model instance
    model = get_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set model to evaluation mode
    model.eval()
    
    return model.to(device) 