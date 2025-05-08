"""
Defect detector model for structural health monitoring.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import torchvision.models as models

class SimpleDefectDetector(nn.Module):
    """A simple CNN model for defect detection."""
    
    def __init__(self, num_classes: int = 2):
        """
        Initialize the defect detector model.
        
        Args:
            num_classes: Number of defect classes to detect
        """
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tensor with class logits
        """
        # Convolutional layers with activation and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten and feed into fully connected layers
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class DefectDetectorResNet(nn.Module):
    """ResNet-based model for defect detection."""
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        """
        Initialize the ResNet-based defect detector.
        
        Args:
            num_classes: Number of defect classes to detect
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        # Load pre-trained ResNet model
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Replace the final fully connected layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tensor with class logits
        """
        return self.resnet(x)

def create_model(config: Dict[str, Any]) -> nn.Module:
    """
    Create a defect detection model based on configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized model
    """
    model_name = config.get('backbone', 'resnet50')
    num_classes = config.get('num_classes', 2)
    pretrained = config.get('pretrained', True)
    
    if model_name == 'simple':
        return SimpleDefectDetector(num_classes=num_classes)
    elif model_name == 'resnet50':
        return DefectDetectorResNet(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model backbone: {model_name}")

def load_model(model_path: str, config: Dict[str, Any]) -> nn.Module:
    """
    Load a model from a checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        config: Model configuration dictionary
        
    Returns:
        Loaded model
    """
    model = create_model(config)
    
    # Load model weights if available
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
    
    return model

def predict(model: nn.Module, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Make predictions with the model.
    
    Args:
        model: The defect detection model
        image: Input image tensor of shape (batch_size, channels, height, width)
        
    Returns:
        Tuple of (predicted class indices, class probabilities)
    """
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(probabilities, 1)
    
    return predicted, probabilities 