#!/usr/bin/env python3
"""
Direct data processing script for structural health monitoring.

This script directly processes the available crack detection and UAV inspection images,
bypassing the complex pipeline and avoiding the missing images.
"""
import os
import sys
import logging
import shutil
import pandas as pd
import yaml
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("direct_processor")

# Configuration
CONFIG = {
    'input_size': [224, 224],
    'normalize': True,
    'splits': {
        'train': 0.7,
        'val': 0.15,
        'test': 0.15
    }
}

class DirectProcessor:
    """Direct processor for the available image data."""
    
    def __init__(self):
        """Initialize the processor."""
        self.base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = self.base_dir / 'uploads' / 'datasets'
        self.output_dir = self.base_dir / 'data' / 'processed'
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for split in ['train', 'val', 'test']:
            (self.output_dir / split).mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Base directory: {self.base_dir}")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        
    def preprocess_image(self, image_path: Path, output_path: Path) -> bool:
        """
        Preprocess a single image.
        
        Args:
            image_path (Path): Path to the source image
            output_path (Path): Path to save the processed image
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                logger.warning(f"Failed to read image: {image_path}")
                return False
                
            # Resize
            img = cv2.resize(img, (CONFIG['input_size'][1], CONFIG['input_size'][0]), interpolation=cv2.INTER_AREA)
            
            # Normalize if needed
            if CONFIG['normalize']:
                img = img.astype(np.float32) / 255.0
                img = (img * 255).astype(np.uint8)
                
            # Save processed image
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(str(output_path), img)
            return True
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return False
    
    def process_crack_detection_data(self) -> pd.DataFrame:
        """
        Process the crack detection dataset.
        
        Returns:
            pd.DataFrame: Processed annotations
        """
        logger.info("Processing Crack Detection dataset...")
        
        crack_dir = self.data_dir / 'Crack_Detection'
        
        # Check if annotations file exists
        annotations_file = crack_dir / 'initial_annotations.csv'
        if not annotations_file.exists():
            logger.error(f"Annotations file not found: {annotations_file}")
            return pd.DataFrame()
        
        # Load annotations
        df = pd.read_csv(annotations_file)
        logger.info(f"Loaded {len(df)} annotations from {annotations_file}")
        
        # Process each image
        processed_rows = []
        
        for idx, row in tqdm(df.iterrows(), desc="Processing Crack Detection images", total=len(df)):
            src_path = crack_dir / row['image_path']
            
            # Skip if source file doesn't exist
            if not src_path.exists():
                logger.warning(f"Source image not found: {src_path}")
                continue
                
            # Create relative path for saving
            if 'Positive' in row['image_path']:
                rel_path = f"crack_detection/positive/{src_path.name}"
            else:
                rel_path = f"crack_detection/negative/{src_path.name}"
                
            # Create output paths for each split
            for split in ['train', 'val', 'test']:
                split_path = self.output_dir / split / rel_path
                if split == 'train':
                    # Process and save the image
                    if self.preprocess_image(src_path, split_path):
                        # Add to processed rows
                        processed_row = row.copy()
                        processed_row['image_path'] = str(split_path.relative_to(self.output_dir.parent))
                        processed_row['split'] = split
                        processed_rows.append(processed_row)
            
        # Create DataFrame
        processed_df = pd.DataFrame(processed_rows)
        logger.info(f"Processed {len(processed_df)} Crack Detection images")
        
        return processed_df
    
    def process_uav_inspection_data(self) -> pd.DataFrame:
        """
        Process the UAV inspection dataset.
        
        Returns:
            pd.DataFrame: Processed annotations
        """
        logger.info("Processing UAV Inspection dataset...")
        
        uav_dir = self.data_dir / 'UAV_Inspection'
        images_dir = uav_dir / 'images'
        
        # Check if images directory exists
        if not images_dir.exists():
            logger.error(f"Images directory not found: {images_dir}")
            return pd.DataFrame()
        
        # Process each image
        processed_rows = []
        
        for img_path in tqdm(list(images_dir.glob('*.png')) + list(images_dir.glob('*.PNG')), 
                         desc="Processing UAV Inspection images"):
            
            # Determine if image contains cracks (based on filename)
            has_crack = 'crack' in img_path.name.lower()
            
            # Create relative path for saving
            rel_path = f"uav_inspection/{img_path.name}"
                
            # Create output paths for each split
            for split in ['train', 'val', 'test']:
                split_path = self.output_dir / split / rel_path
                
                if split == 'train':
                    # Process and save the image
                    if self.preprocess_image(img_path, split_path):
                        # Add to processed rows
                        processed_row = {
                            'image_path': str(split_path.relative_to(self.output_dir.parent)),
                            'defect_class': 'Crack' if has_crack else 'No_Defect',
                            'has_defect': has_crack,
                            'dataset_source': 'uav_inspection',
                            'split': split
                        }
                        processed_rows.append(processed_row)
            
        # Create DataFrame
        processed_df = pd.DataFrame(processed_rows)
        logger.info(f"Processed {len(processed_df)} UAV Inspection images")
        
        return processed_df
    
    def split_data(self, df: pd.DataFrame) -> dict:
        """
        Split data into train/val/test sets.
        
        Args:
            df (pd.DataFrame): Data to split
            
        Returns:
            dict: Dataframes for each split
        """
        logger.info("Splitting data...")
        
        # Get unique classes
        classes = df['defect_class'].unique()
        
        splits = {
            'train': pd.DataFrame(),
            'val': pd.DataFrame(),
            'test': pd.DataFrame()
        }
        
        # Split for each class to maintain class balance
        for cls in classes:
            cls_df = df[df['defect_class'] == cls]
            
            # Shuffle
            cls_df = cls_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Calculate split indices
            train_idx = int(len(cls_df) * CONFIG['splits']['train'])
            val_idx = train_idx + int(len(cls_df) * CONFIG['splits']['val'])
            
            # Split data
            splits['train'] = pd.concat([splits['train'], cls_df[:train_idx]])
            splits['val'] = pd.concat([splits['val'], cls_df[train_idx:val_idx]])
            splits['test'] = pd.concat([splits['test'], cls_df[val_idx:]])
        
        # Reset indices
        for split in splits:
            splits[split] = splits[split].reset_index(drop=True)
            
        logger.info(f"Training samples: {len(splits['train'])}")
        logger.info(f"Validation samples: {len(splits['val'])}")
        logger.info(f"Test samples: {len(splits['test'])}")
        
        return splits
    
    def process_data(self) -> bool:
        """
        Process all data.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Process Crack Detection data
            crack_df = self.process_crack_detection_data()
            
            # Process UAV Inspection data
            uav_df = self.process_uav_inspection_data()
            
            # Combine data
            all_data = pd.concat([crack_df, uav_df]).reset_index(drop=True)
            
            if all_data.empty:
                logger.error("No data processed!")
                return False
                
            # Split data
            splits = self.split_data(all_data)
            
            # Save annotations for each split
            for split_name, split_df in splits.items():
                split_df.to_csv(self.output_dir / f"{split_name}_annotations.csv", index=False)
                logger.info(f"Saved {split_name} annotations")
            
            # Save dataset info
            dataset_info = {
                'splits': list(splits.keys()),
                'num_classes': len(all_data['defect_class'].unique()),
                'class_mapping': {cls: idx for idx, cls in enumerate(all_data['defect_class'].unique())},
                'preprocessing': {
                    'input_size': CONFIG['input_size'],
                    'normalize': CONFIG['normalize'],
                }
            }
            
            with open(self.output_dir / 'dataset_info.yaml', 'w') as f:
                yaml.dump(dataset_info, f)
            
            logger.info("Data processing completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            return False

def main():
    """Run the direct processing."""
    logger.info("Starting direct data processing...")
    
    processor = DirectProcessor()
    success = processor.process_data()
    
    if success:
        logger.info("\nDataset Summary:")
        logger.info("=" * 40)
        
        try:
            with open(processor.output_dir / 'dataset_info.yaml', 'r') as f:
                info = yaml.safe_load(f)
                
            logger.info(f"Classes: {list(info['class_mapping'].keys())}")
            logger.info(f"Input size: {info['preprocessing']['input_size']}")
            logger.info(f"Normalization: {info['preprocessing']['normalize']}")
            
            logger.info(f"\nProcessed data saved to: {processor.output_dir}")
        except Exception as e:
            logger.error(f"Error loading dataset info: {str(e)}")
    else:
        logger.error("Data processing failed!")
        
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 