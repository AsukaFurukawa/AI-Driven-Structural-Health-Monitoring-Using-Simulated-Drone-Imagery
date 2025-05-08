"""
Data loader module for handling drone imagery datasets.
"""
import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import cv2
import numpy as np
from PIL import Image


class DroneImageDataset:
    """Dataset class for loading and preprocessing drone imagery."""
    
    def __init__(
        self,
        data_dir: str,
        img_size: Tuple[int, int] = (512, 512),
        transform=None
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing the image data
            img_size: Target size for image resizing (height, width)
            transform: Optional transform to be applied to images
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.transform = transform
        self.image_paths = self._get_image_paths()
        
    def _get_image_paths(self) -> List[Path]:
        """Get list of image paths from the data directory."""
        valid_extensions = ('.jpg', '.jpeg', '.png')
        return [
            f for f in self.data_dir.glob('**/*')
            if f.suffix.lower() in valid_extensions
        ]
    
    def __len__(self) -> int:
        """Return the total number of images."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get a single image item.
        
        Args:
            idx: Index of the image to retrieve
            
        Returns:
            Dictionary containing the image and its metadata
        """
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, self.img_size)
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'path': str(img_path)
        }
    
    def preprocess_image(
        self,
        image: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Preprocess a single image.
        
        Args:
            image: Input image array
            normalize: Whether to normalize pixel values to [0,1]
            
        Returns:
            Preprocessed image array
        """
        # Convert to float32
        image = image.astype(np.float32)
        
        # Normalize if requested
        if normalize:
            image /= 255.0
            
        return image
    
    def get_batch(
        self,
        batch_size: int,
        indices: Optional[List[int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get a batch of images.
        
        Args:
            batch_size: Number of images to retrieve
            indices: Optional list of specific indices to retrieve
            
        Returns:
            Dictionary containing batch of images and their metadata
        """
        if indices is None:
            indices = np.random.choice(
                len(self), batch_size, replace=False
            )
            
        batch_images = []
        batch_paths = []
        
        for idx in indices:
            item = self[idx]
            batch_images.append(item['image'])
            batch_paths.append(item['path'])
            
        return {
            'images': np.stack(batch_images),
            'paths': batch_paths
        } 