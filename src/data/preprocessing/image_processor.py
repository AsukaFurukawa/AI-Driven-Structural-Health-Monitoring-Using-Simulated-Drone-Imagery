import cv2
import numpy as np
import torch
from torchvision import transforms
from pathlib import Path
import yaml

class ImageProcessor:
    """
    Class for preprocessing drone imagery for structural defect detection.
    """
    def __init__(self, config_path: str = None):
        """
        Initialize the image processor.
        
        Args:
            config_path (str): Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.input_size = tuple(self.config['model']['input_size'])
        
        # Define standard transformations
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_config(self, config_path: str = None):
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent.parent / 'config' / 'config.yaml'
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single image for model input.
        
        Args:
            image (np.ndarray): Input image in BGR format (from cv2)
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        tensor = self.transform(image)
        
        # Add batch dimension
        return tensor.unsqueeze(0)
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality for better defect detection.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # Merge channels
        enhanced_lab = cv2.merge((cl, a, b))
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges in the image for potential crack detection.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Edge map
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        return edges
    
    def segment_defects(self, image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Segment potential defect regions using thresholding.
        
        Args:
            image (np.ndarray): Input image
            threshold (float): Threshold value for segmentation
            
        Returns:
            np.ndarray: Binary mask of potential defect regions
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # Apply morphological operations
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    @staticmethod
    def overlay_defect_map(image: np.ndarray, defect_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Overlay defect map on original image.
        
        Args:
            image (np.ndarray): Original image
            defect_map (np.ndarray): Defect probability map
            alpha (float): Transparency of overlay
            
        Returns:
            np.ndarray: Image with overlaid defect map
        """
        # Normalize defect map to 0-255
        heat_map = (defect_map * 255).astype(np.uint8)
        
        # Apply colormap
        colored_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
        
        # Overlay
        overlay = cv2.addWeighted(image, 1-alpha, colored_map, alpha, 0)
        
        return overlay 