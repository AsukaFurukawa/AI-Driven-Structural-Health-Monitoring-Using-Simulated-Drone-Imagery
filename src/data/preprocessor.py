"""
Image preprocessing module for structural health monitoring.

This module provides utilities for image preprocessing, normalization,
and augmentation for the structural health monitoring pipeline.
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from typing import Tuple, List, Dict, Optional, Union
import logging
import random
from tqdm import tqdm
import albumentations as A
import shutil
from PIL import Image, ImageEnhance, ImageFilter

class ImagePreprocessor:
    """
    Handles preprocessing, normalization, and augmentation of images for model training.
    """
    
    def __init__(self, data_dir: str, output_dir: str, target_size: List[int] = [224, 224],
                 normalize: bool = True, augmentation: bool = False):
        """
        Initialize the image preprocessor.
        
        Args:
            data_dir (str): Directory containing the dataset
            output_dir (str): Directory to save processed images
            target_size (List[int]): Target image dimensions [height, width]
            normalize (bool): Whether to normalize images
            augmentation (bool): Whether to apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.target_size = tuple(target_size)
        self.normalize = normalize
        self.augmentation = augmentation
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        
        # Setup augmentation pipeline if enabled
        if self.augmentation:
            self.aug_pipeline = A.Compose([
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
                A.GridDistortion(p=0.2),
                A.CLAHE(p=0.3),
            ])
    
    def preprocess_image(self, image_path: Path) -> np.ndarray:
        """
        Apply preprocessing to a single image.
        
        Args:
            image_path (Path): Path to the image
            
        Returns:
            np.ndarray: Preprocessed image
        """
        try:
            # Read image
            full_path = self.data_dir / image_path
            img = cv2.imread(str(full_path))
            if img is None:
                self.logger.warning(f"Failed to read image: {full_path}")
                return None
                
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to target size
            img = cv2.resize(img, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_AREA)
            
            # Normalize if enabled
            if self.normalize:
                img = img.astype(np.float32) / 255.0
            
            return img
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            return None
    
    def apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to an image.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Augmented image
        """
        if not self.augmentation:
            return image
            
        try:
            augmented = self.aug_pipeline(image=image)
            return augmented['image']
        except Exception as e:
            self.logger.error(f"Error applying augmentation: {str(e)}")
            return image
    
    def save_processed_image(self, image: np.ndarray, output_path: Path) -> str:
        """
        Save a processed image.
        
        Args:
            image (np.ndarray): Processed image
            output_path (Path): Output path
            
        Returns:
            str: Relative path to the saved image
        """
        try:
            # Create directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Denormalize if needed
            if self.normalize and image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            elif image.dtype != np.uint8:
                image = image.astype(np.uint8)
            
            # Convert to BGR for OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
            # Save image
            cv2.imwrite(str(output_path), image)
            
            # Return relative path
            return str(output_path.relative_to(self.output_dir.parent))
            
        except Exception as e:
            self.logger.error(f"Error saving image to {output_path}: {str(e)}")
            return None
    
    def preprocess_dataset(self, annotations_df: pd.DataFrame, augment_ratio: float = 0.5) -> pd.DataFrame:
        """
        Preprocess an entire dataset.
        
        Args:
            annotations_df (pd.DataFrame): DataFrame with annotations
            augment_ratio (float): Ratio of original data to augment (if augmentation is enabled)
            
        Returns:
            pd.DataFrame: Updated annotations with processed images
        """
        processed_annotations = []
        
        # Process each image in the dataset
        for idx, row in tqdm(annotations_df.iterrows(), desc="Preprocessing images", total=len(annotations_df)):
            # Get original image path
            original_path = row['image_path']
            
            # Process image
            processed_img = self.preprocess_image(original_path)
            if processed_img is None:
                continue
            
            # Create output path
            rel_path = Path(original_path)
            output_path = self.output_dir / rel_path.name
            
            # Save processed image
            saved_path = self.save_processed_image(processed_img, output_path)
            if saved_path is None:
                continue
                
            # Create annotation entry
            processed_row = row.copy()
            processed_row['image_path'] = saved_path
            processed_annotations.append(processed_row)
            
            # Apply augmentation if enabled
            if self.augmentation and random.random() < augment_ratio:
                # Apply augmentation
                augmented_img = self.apply_augmentation(processed_img)
                
                # Create output path for augmented image
                aug_name = f"aug_{rel_path.stem}_{random.randint(0, 999)}{rel_path.suffix}"
                aug_path = self.output_dir / aug_name
                
                # Save augmented image
                aug_saved_path = self.save_processed_image(augmented_img, aug_path)
                if aug_saved_path is None:
                    continue
                    
                # Create annotation entry for augmented image
                aug_row = row.copy()
                aug_row['image_path'] = aug_saved_path
                aug_row['augmented'] = True
                processed_annotations.append(aug_row)
        
        # Create new DataFrame
        processed_df = pd.DataFrame(processed_annotations)
        
        # Add information about preprocessing
        processed_df['preprocessed'] = True
        processed_df['normalized'] = self.normalize
        if 'augmented' not in processed_df.columns:
            processed_df['augmented'] = False
            
        return processed_df
    
    def apply_contrast_enhancement(self, image: np.ndarray, enhancement_factor: float = 1.5) -> np.ndarray:
        """
        Apply contrast enhancement to an image.
        
        Args:
            image (np.ndarray): Input image
            enhancement_factor (float): Contrast enhancement factor
            
        Returns:
            np.ndarray: Enhanced image
        """
        try:
            # Convert to PIL image
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
                
            pil_img = Image.fromarray(image)
            
            # Apply contrast enhancement
            enhancer = ImageEnhance.Contrast(pil_img)
            enhanced_img = enhancer.enhance(enhancement_factor)
            
            # Convert back to numpy
            enhanced = np.array(enhanced_img)
            
            # Normalize if needed
            if self.normalize:
                enhanced = enhanced.astype(np.float32) / 255.0
                
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Error applying contrast enhancement: {str(e)}")
            return image
    
    def apply_sharpening(self, image: np.ndarray, radius: float = 2.0, percent: int = 150) -> np.ndarray:
        """
        Apply sharpening to an image.
        
        Args:
            image (np.ndarray): Input image
            radius (float): Sharpening radius
            percent (int): Sharpening percentage
            
        Returns:
            np.ndarray: Sharpened image
        """
        try:
            # Convert to PIL image
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
                
            pil_img = Image.fromarray(image)
            
            # Apply unsharp mask filter for sharpening
            sharpened = pil_img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent))
            
            # Convert back to numpy
            sharpened_img = np.array(sharpened)
            
            # Normalize if needed
            if self.normalize:
                sharpened_img = sharpened_img.astype(np.float32) / 255.0
                
            return sharpened_img
            
        except Exception as e:
            self.logger.error(f"Error applying sharpening: {str(e)}")
            return image
    
    def enhance_structural_features(self, annotations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply structural feature enhancement to crack images.
        This is useful for improving visibility of cracks and defects.
        
        Args:
            annotations_df (pd.DataFrame): DataFrame with annotations
            
        Returns:
            pd.DataFrame: Updated annotations with enhanced images
        """
        enhanced_annotations = []
        
        # Process each image in the dataset
        for idx, row in tqdm(annotations_df.iterrows(), desc="Enhancing structural features", total=len(annotations_df)):
            # Get original image path
            img_path = row['image_path']
            output_path = self.output_dir / Path(img_path).name
            
            # Read and preprocess image
            full_path = self.output_dir.parent / img_path
            img = cv2.imread(str(full_path))
            
            if img is None:
                self.logger.warning(f"Failed to read image: {full_path}")
                continue
                
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Only apply enhancement to defect images
            if row.get('has_defect', False) or 'defect' in row.get('defect_class', '').lower():
                # Apply contrast enhancement
                enhanced = self.apply_contrast_enhancement(img, enhancement_factor=1.5)
                
                # Apply sharpening
                enhanced = self.apply_sharpening(enhanced, radius=1.5, percent=140)
                
                # Save enhanced image
                enhanced_path = self.output_dir / f"enhanced_{Path(img_path).name}"
                saved_path = self.save_processed_image(enhanced, enhanced_path)
                
                if saved_path:
                    # Create annotation entry
                    enhanced_row = row.copy()
                    enhanced_row['image_path'] = saved_path
                    enhanced_row['enhanced'] = True
                    enhanced_annotations.append(enhanced_row)
            
            # Always include the original
            enhanced_annotations.append(row)
                
        # Create new DataFrame
        enhanced_df = pd.DataFrame(enhanced_annotations)
        
        # Add information about enhancement
        if 'enhanced' not in enhanced_df.columns:
            enhanced_df['enhanced'] = False
            
        return enhanced_df

def apply_structural_feature_extraction(image: np.ndarray) -> np.ndarray:
    """
    Extract structural features from an image such as edges, corners, etc.
    
    Args:
        image (np.ndarray): Input image
        
    Returns:
        np.ndarray: Feature map
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Normalize to 8-bit
    if gray.max() <= 1.0:
        gray = (gray * 255).astype(np.uint8)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Apply corner detection
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    corners = cv2.dilate(corners, None)
    
    # Create feature map
    feature_map = np.zeros_like(gray)
    feature_map = np.maximum(feature_map, edges)
    feature_map[corners > 0.01 * corners.max()] = 255
    
    return feature_map

def analyze_crack_patterns(image: np.ndarray) -> Tuple[float, float, float]:
    """
    Analyze crack patterns in an image.
    
    Args:
        image (np.ndarray): Input image
        
    Returns:
        Tuple[float, float, float]: Crack length, width, and orientation
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Normalize to 8-bit
    if gray.max() <= 1.0:
        gray = (gray * 255).astype(np.uint8)
    
    # Threshold to separate cracks
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0, 0.0, 0.0
    
    # Find the largest contour (main crack)
    main_contour = max(contours, key=cv2.contourArea)
    
    # Calculate properties
    rect = cv2.minAreaRect(main_contour)
    (_, _), (width, height), angle = rect
    
    # Calculate crack length and width
    crack_length = max(width, height)
    crack_width = min(width, height)
    
    # Normalize orientation to 0-180 degrees
    orientation = angle % 180
    
    return crack_length, crack_width, orientation 