"""
Dataset module for structural health monitoring.

This module provides PyTorch dataset classes for loading and augmenting 
structural health monitoring image data.
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class CrackDetectionDataset(Dataset):
    """
    Dataset for crack detection in structural images.
    
    This dataset loads images from the processed data directory and applies 
    specified transforms.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = 'train',
        transforms_fn: Optional[Callable] = None,
        label_map: Optional[Dict] = None,
        include_drone: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing the processed data
            split: Data split to use ('train', 'val', or 'test')
            transforms_fn: Transform function to apply to the images
            label_map: Dictionary mapping class names to integers
            include_drone: Whether to include drone imagery
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transforms_fn = transforms_fn
        self.include_drone = include_drone
        
        # Set up logging
        self.logger = logging.getLogger(f"crack_detection_dataset_{split}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Set default class names and label map
        self.class_names = ['negative', 'positive']
        self.num_classes = 2
        self.label_map = label_map or {'negative': 0, 'positive': 1}
        
        # Load dataset info if available
        dataset_info_path = self.data_dir / "dataset_info.yaml"
        if dataset_info_path.exists():
            with open(dataset_info_path, 'r') as f:
                self.dataset_info = yaml.safe_load(f)
            
            # Update class names and counts
            if 'class_names' in self.dataset_info:
                self.class_names = self.dataset_info['class_names']
                self.num_classes = len(self.class_names)
                
                # Update label map if not provided
                if label_map is None:
                    self.label_map = {class_name: i for i, class_name in enumerate(self.class_names)}
        
        # Load the data
        self.data = self._load_data()
        self.logger.info(f"Loaded {len(self.data)} samples for {split} split")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """
        Load the dataset from the data directory.
        
        Returns:
            List of dictionaries containing image paths and labels
        """
        image_data = []
        
        # Standard image directories
        positive_dir = self.data_dir / "positive"
        negative_dir = self.data_dir / "negative"
        
        # Drone image directories
        drone_positive_dir = self.data_dir / "drone_positive"
        drone_negative_dir = self.data_dir / "drone_negative"
        
        # Process standard positive images
        if positive_dir.exists():
            for image_path in positive_dir.glob("*.*"):
                if image_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    image_data.append({
                        "path": str(image_path),
                        "label": 1,
                        "is_drone": False
                    })
        
        # Process standard negative images
        if negative_dir.exists():
            for image_path in negative_dir.glob("*.*"):
                if image_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    image_data.append({
                        "path": str(image_path),
                        "label": 0,
                        "is_drone": False
                    })
        
        # Process drone images if included
        if self.include_drone:
            # Process drone positive images
            if drone_positive_dir.exists():
                for image_path in drone_positive_dir.glob("*.*"):
                    if image_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        image_data.append({
                            "path": str(image_path),
                            "label": 1,
                            "is_drone": True
                        })
            
            # Process drone negative images
            if drone_negative_dir.exists():
                for image_path in drone_negative_dir.glob("*.*"):
                    if image_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        image_data.append({
                            "path": str(image_path),
                            "label": 0,
                            "is_drone": True
                        })
        
        return image_data
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing the image and label
        """
        item = self.data[idx]
        image_path = item["path"]
        label = item["label"]
        is_drone = item["is_drone"]
        
        try:
            # Load image (RGB for drone images, grayscale for standard)
            if is_drone:
                image = Image.open(image_path).convert('RGB')
            else:
                image = Image.open(image_path).convert('L')
                # Convert grayscale to RGB if needed for model consistency
                if any(d.exists() for d in [self.data_dir / "drone_positive", self.data_dir / "drone_negative"]):
                    image = Image.merge('RGB', (image, image, image))
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            # Return a blank image as fallback
            if is_drone:
                image = Image.new('RGB', (224, 224))
            else:
                image = Image.new('L', (224, 224))
                # Convert to RGB if needed
                if any(d.exists() for d in [self.data_dir / "drone_positive", self.data_dir / "drone_negative"]):
                    image = Image.merge('RGB', (image, image, image))
        
        # Apply transforms
        if self.transforms_fn:
            image = self.transforms_fn(image)
        
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long)
        }

def get_transforms(split: str = 'train') -> Callable:
    """
    Get data transforms for the specified split.
    
    Args:
        split: Data split ('train', 'val', or 'test')
        
    Returns:
        Transform function
    """
    # Base transforms for all splits
    if split == 'train':
        # More augmentations for training
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406] if any(Path("data/processed").glob("drone_*")) else [0.485],
                std=[0.229, 0.224, 0.225] if any(Path("data/processed").glob("drone_*")) else [0.229]
            )
        ])
    else:
        # Minimal transforms for validation and test
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406] if any(Path("data/processed").glob("drone_*")) else [0.485],
                std=[0.229, 0.224, 0.225] if any(Path("data/processed").glob("drone_*")) else [0.229]
            )
        ])

def create_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    config: Optional[Dict] = None,
    include_drone: bool = True
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for training, validation, and testing.
    
    Args:
        data_dir: Directory containing the processed data
        batch_size: Batch size for the dataloaders
        num_workers: Number of workers for data loading
        config: Optional configuration dictionary
        include_drone: Whether to include drone imagery
        
    Returns:
        Dictionary of dataloaders for each split
    """
    data_dir = Path(data_dir)
    
    # Check if we need to use train/val/test split or create our own
    if (data_dir / 'train').exists() and (data_dir / 'val').exists():
        # Use predefined splits
        train_dataset = CrackDetectionDataset(
            data_dir=data_dir / 'train',
            split='train',
            transforms_fn=get_transforms('train'),
            include_drone=include_drone
        )
        
        val_dataset = CrackDetectionDataset(
            data_dir=data_dir / 'val',
            split='val',
            transforms_fn=get_transforms('val'),
            include_drone=include_drone
        )
        
        test_split_exists = (data_dir / 'test').exists()
        test_dataset = CrackDetectionDataset(
            data_dir=data_dir / ('test' if test_split_exists else 'val'),
            split='test',
            transforms_fn=get_transforms('test'),
            include_drone=include_drone
        )
    else:
        # Create a dataset from the entire directory and split it
        full_dataset = CrackDetectionDataset(
            data_dir=data_dir,
            transforms_fn=None,  # We'll apply transforms after splitting
            include_drone=include_drone
        )
        
        # Extract indices for train/val/test splits
        indices = list(range(len(full_dataset)))
        
        # Get split ratios from config or use defaults
        if config and 'data' in config and 'train_val_test_split' in config['data']:
            train_ratio, val_ratio, test_ratio = config['data']['train_val_test_split']
        else:
            train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
            
        # Calculate split sizes
        train_size = int(train_ratio * len(indices))
        val_size = int(val_ratio * len(indices))
        test_size = len(indices) - train_size - val_size
        
        # Create split datasets
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create custom dataset splits with appropriate transforms
        def create_subset(idx_list, split_name):
            subset_data = [full_dataset.data[i] for i in idx_list]
            subset = CrackDetectionDataset(
                data_dir=data_dir,
                split=split_name,
                transforms_fn=get_transforms(split_name),
                include_drone=include_drone
            )
            subset.data = subset_data
            return subset
        
        train_dataset = create_subset(train_indices, 'train')
        val_dataset = create_subset(val_indices, 'val')
        test_dataset = create_subset(test_indices, 'test')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"Created dataloaders with {len(train_dataset)} training, "
               f"{len(val_dataset)} validation, and {len(test_dataset)} test samples")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

class CrackDataset(Dataset):
    """Dataset class for crack detection images."""
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Initialize the dataset.
        
        Args:
            image_paths (list): List of image file paths
            labels (list): List of corresponding labels
            transform (callable, optional): Optional transform to be applied on images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, label)
        """
        # Load image
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('L')  # Convert to grayscale
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            # Return a blank image as fallback
            image = Image.new('L', (224, 224))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.labels[idx]
        
        return image, label

def get_transforms(config):
    """
    Get data transforms for training and validation.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (train_transform, val_transform)
    """
    # Base transforms
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    # Training transforms with augmentation
    if config.get('data', {}).get('augmentation', {}).get('enabled', True):
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
    else:
        train_transform = base_transform
    
    return train_transform, base_transform

def create_dataloaders(data_dir, batch_size=32, num_workers=4, config=None):
    """
    Create data loaders for training and validation.
    
    Args:
        data_dir (str): Path to the data directory
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes for data loading
        config (dict, optional): Configuration dictionary
        
    Returns:
        dict: Dictionary containing train and validation data loaders
    """
    if config is None:
        config = {}
    
    # Get data transforms
    train_transform, val_transform = get_transforms(config)
    
    # Get image paths and labels
    image_paths = []
    labels = []
    
    # Walk through the data directory
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
                # Label is 1 if image is in a 'positive' directory, 0 otherwise
                labels.append(1 if 'positive' in root.lower() else 0)
    
    if not image_paths:
        raise ValueError(f"No images found in {data_dir}")
    
    # Split data into train and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    
    # Create datasets
    train_dataset = CrackDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = CrackDataset(val_paths, val_labels, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created data loaders with {len(train_dataset)} training and {len(val_dataset)} validation samples")
    
    return {
        'train': train_loader,
        'val': val_loader
    }

def preprocess_image(image_path, transform=None):
    """
    Preprocess a single image for inference.
    
    Args:
        image_path (str): Path to the image file
        transform (callable, optional): Optional transform to be applied
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
    
    try:
        image = Image.open(image_path).convert('L')
        image = transform(image)
        return image.unsqueeze(0)  # Add batch dimension
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {str(e)}")
        return None 