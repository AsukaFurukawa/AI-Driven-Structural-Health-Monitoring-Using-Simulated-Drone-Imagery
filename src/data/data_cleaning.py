import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm
from PIL import Image
import imagehash
import shutil

class DataCleaner:
    """Class for cleaning and validating the structural defect dataset."""
    
    def __init__(self, data_dir: str, min_image_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the data cleaner.
        
        Args:
            data_dir (str): Directory containing the dataset
            min_image_size (tuple): Minimum acceptable image dimensions
        """
        self.data_dir = Path(data_dir)
        self.min_image_size = min_image_size
        self.logger = logging.getLogger(__name__)
        
    def validate_image(self, image_path: Path) -> bool:
        """
        Check if an image is valid and meets minimum requirements.
        
        Args:
            image_path (Path): Path to the image file
            
        Returns:
            bool: True if image is valid, False otherwise
        """
        try:
            # Check if file exists
            if not image_path.exists():
                self.logger.warning(f"Image not found: {image_path}")
                return False
            
            # Try to open the image with PIL
            try:
                with Image.open(image_path) as img:
                    if img.format not in ['JPEG', 'PNG', 'BMP']:
                        self.logger.warning(f"Invalid image format: {image_path}")
                        return False
                    width, height = img.size
            except Exception as e:
                self.logger.warning(f"Could not open image with PIL: {image_path}")
                return False
            
            # Check dimensions
            if height < self.min_image_size[0] or width < self.min_image_size[1]:
                self.logger.warning(f"Image too small: {image_path} ({width}x{height})")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating image {image_path}: {str(e)}")
            return False
    
    def find_duplicate_images(self, image_paths: List[Path], threshold: int = 5) -> Dict[str, List[Path]]:
        """
        Find duplicate or near-duplicate images using perceptual hashing.
        
        Args:
            image_paths (List[Path]): List of image paths to check
            threshold (int): Hash difference threshold for considering images as duplicates
            
        Returns:
            Dict[str, List[Path]]: Groups of duplicate images
        """
        hash_dict = {}
        duplicates = {}
        
        for img_path in tqdm(image_paths, desc="Finding duplicates"):
            try:
                with Image.open(img_path) as img:
                    img_hash = str(imagehash.average_hash(img))
                    
                    # Check for near-duplicates
                    for existing_hash in hash_dict:
                        if imagehash.hex_to_hash(img_hash) - imagehash.hex_to_hash(existing_hash) < threshold:
                            if existing_hash not in duplicates:
                                duplicates[existing_hash] = [hash_dict[existing_hash]]
                            duplicates[existing_hash].append(img_path)
                            break
                    else:
                        hash_dict[img_hash] = img_path
                        
            except Exception as e:
                self.logger.error(f"Error processing {img_path}: {str(e)}")
                
        return duplicates
    
    def clean_annotations(self, annotations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate annotations.
        
        Args:
            annotations_df (pd.DataFrame): DataFrame containing annotations
            
        Returns:
            pd.DataFrame: Cleaned annotations
        """
        # Remove rows with missing values
        cleaned_df = annotations_df.dropna(subset=['image_path', 'defect_class'])
        
        # Validate image paths
        valid_images = []
        for idx, row in tqdm(cleaned_df.iterrows(), desc="Validating images", total=len(cleaned_df)):
            img_path = self.data_dir / row['image_path']
            if self.validate_image(img_path):
                valid_images.append(idx)
        
        cleaned_df = cleaned_df.loc[valid_images]
        
        # Validate bounding boxes if present
        if all(col in cleaned_df.columns for col in ['x_min', 'y_min', 'x_max', 'y_max']):
            # Remove invalid boxes
            cleaned_df = cleaned_df[
                (cleaned_df['x_min'] < cleaned_df['x_max']) &
                (cleaned_df['y_min'] < cleaned_df['y_max']) &
                (cleaned_df['x_min'] >= 0) &
                (cleaned_df['y_min'] >= 0)
            ]
        
        return cleaned_df
    
    def standardize_image_size(self, image_path: Path, target_size: Tuple[int, int]) -> None:
        """
        Resize image to standard size if needed.
        
        Args:
            image_path (Path): Path to the image
            target_size (tuple): Target image dimensions
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return
                
            height, width = img.shape[:2]
            if height != target_size[0] or width != target_size[1]:
                resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(image_path), resized)
                
        except Exception as e:
            self.logger.error(f"Error resizing {image_path}: {str(e)}")
    
    def clean_dataset(self, annotations_file: str, target_size: Tuple[int, int] = (224, 224)) -> pd.DataFrame:
        """
        Perform complete dataset cleaning.
        
        Args:
            annotations_file (str): Path to annotations CSV file
            target_size (tuple): Target image size for standardization
            
        Returns:
            pd.DataFrame: Cleaned annotations
        """
        # Load annotations
        df = pd.read_csv(annotations_file)
        
        # Clean annotations
        cleaned_df = self.clean_annotations(df)
        
        # Find and handle duplicates
        image_paths = [self.data_dir / path for path in cleaned_df['image_path']]
        duplicates = self.find_duplicate_images(image_paths)
        
        if duplicates:
            self.logger.info(f"Found {len(duplicates)} groups of duplicate images")
            # Keep only one image from each duplicate group
            for group in duplicates.values():
                for dup_path in group[1:]:
                    cleaned_df = cleaned_df[cleaned_df['image_path'] != str(dup_path.relative_to(self.data_dir))]
        
        # Standardize image sizes
        for img_path in tqdm(image_paths, desc="Standardizing image sizes"):
            self.standardize_image_size(img_path, target_size)
        
        return cleaned_df

def clean_combined_dataset(base_dir: str):
    """
    Clean the combined dataset.
    
    Args:
        base_dir (str): Base directory containing the combined dataset
    """
    logger = logging.getLogger(__name__)
    base_dir = Path(base_dir)
    combined_dir = base_dir / 'combined_dataset'
    
    cleaner = DataCleaner(str(combined_dir))
    
    # Clean each split
    for split in ['train', 'val', 'test']:
        annotations_file = combined_dir / f'{split}_annotations.csv'
        if annotations_file.exists():
            logger.info(f"Cleaning {split} split...")
            cleaned_df = cleaner.clean_dataset(str(annotations_file))
            
            # Save cleaned annotations
            cleaned_file = combined_dir / f'{split}_annotations_cleaned.csv'
            cleaned_df.to_csv(cleaned_file, index=False)
            
            # Update statistics
            stats = {
                'original_samples': len(pd.read_csv(annotations_file)),
                'cleaned_samples': len(cleaned_df),
                'removed_samples': len(pd.read_csv(annotations_file)) - len(cleaned_df),
                'defect_distribution': cleaned_df['defect_class'].value_counts().to_dict()
            }
            
            logger.info(f"{split} cleaning statistics:\n{stats}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    clean_combined_dataset('datasets') 