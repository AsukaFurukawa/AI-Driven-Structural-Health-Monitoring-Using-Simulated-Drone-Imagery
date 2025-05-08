"""
Comprehensive data processing pipeline for structural health monitoring.
This script orchestrates the entire data flow from raw images to processed datasets ready for model training.
"""
import os
import sys
import logging
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import shutil
import random
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.prepare_dataset import prepare_public_crack_dataset, combine_datasets
from src.data.data_cleaning import DataCleaner, clean_combined_dataset
from src.data.preprocessor import ImagePreprocessor

class DataPipeline:
    """
    Complete data processing pipeline for structural health monitoring.
    
    This class orchestrates the entire process from data acquisition to
    preparing the final datasets for model training.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the data pipeline.
        
        Args:
            config_path (str): Path to the configuration file
        """
        self.logger = self._setup_logging()
        
        # Load configuration
        if config_path is None:
            self.config_path = os.path.join('config', 'config.yaml')
        else:
            self.config_path = config_path
            
        self.config = self._load_config()
        
        # Setup paths
        self.base_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.data_dir = self.base_dir / 'uploads' / 'datasets'
        self.output_dir = self.base_dir / 'data' / 'processed'
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Working directory: {self.base_dir}")
        self.logger.info(f"Data directory: {self.data_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        
    def _setup_logging(self) -> logging.Logger:
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            return {}
    
    def prepare_crack_detection_data(self) -> pd.DataFrame:
        """
        Prepare the crack detection dataset.
        
        Returns:
            pd.DataFrame: Annotations for the crack detection dataset
        """
        self.logger.info("Preparing crack detection dataset...")
        
        # Prepare the dataset
        annotations_df = prepare_public_crack_dataset(str(self.data_dir))
        
        if annotations_df.empty:
            self.logger.error("Failed to prepare crack detection dataset!")
            return pd.DataFrame()
        
        # Save initial annotations
        annotations_file = self.data_dir / 'Crack_Detection' / 'initial_annotations.csv'
        annotations_df.to_csv(annotations_file, index=False)
        self.logger.info(f"Saved initial annotations to {annotations_file}")
        
        # Clean the dataset
        self.logger.info("Cleaning the crack detection dataset...")
        cleaner = DataCleaner(str(self.data_dir / 'Crack_Detection'))
        
        # Clean the dataset with standard image size
        input_size = self.config.get('model', {}).get('input_size', [224, 224])
        target_size = (input_size[0], input_size[1])
        
        cleaned_df = cleaner.clean_dataset(
            str(annotations_file),
            target_size=target_size
        )
        
        # Save cleaned annotations
        cleaned_file = self.data_dir / 'Crack_Detection' / 'cleaned_annotations.csv'
        cleaned_df.to_csv(cleaned_file, index=False)
        
        # Print statistics
        self.logger.info("\nCrack Detection Dataset Statistics:")
        self.logger.info(f"Original samples: {len(annotations_df)}")
        self.logger.info(f"Cleaned samples: {len(cleaned_df)}")
        self.logger.info(f"Removed samples: {len(annotations_df) - len(cleaned_df)}")
        self.logger.info("\nDefect distribution:")
        self.logger.info(cleaned_df['defect_class'].value_counts())
        
        return cleaned_df
    
    def prepare_uav_inspection_data(self) -> pd.DataFrame:
        """
        Prepare the UAV inspection dataset.
        
        Returns:
            pd.DataFrame: Annotations for the UAV inspection dataset
        """
        self.logger.info("Processing UAV inspection dataset...")
        
        uav_dir = self.data_dir / 'UAV_Inspection'
        uav_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if there are images in the UAV inspection folder
        image_files = list(uav_dir.glob('**/*.jpg')) + list(uav_dir.glob('**/*.png'))
        
        if not image_files:
            self.logger.warning("No UAV inspection images found. Skipping this dataset.")
            return pd.DataFrame()
        
        # Process and organize the dataset
        annotations = []
        
        for img_path in tqdm(image_files, desc="Processing UAV images"):
            # Extract relative path
            rel_path = img_path.relative_to(uav_dir)
            
            # Determine defect class based on folder structure
            if 'defect' in str(rel_path).lower() or 'damage' in str(rel_path).lower():
                defect_class = 'Structural_Defect'
                has_defect = True
            else:
                defect_class = 'No_Defect'
                has_defect = False
            
            annotations.append({
                'image_path': str(rel_path),
                'defect_class': defect_class,
                'has_defect': has_defect,
                'dataset_source': 'uav_inspection'
            })
        
        # Create annotations DataFrame
        annotations_df = pd.DataFrame(annotations)
        
        # Save initial annotations
        annotations_file = uav_dir / 'initial_annotations.csv'
        annotations_df.to_csv(annotations_file, index=False)
        
        # Clean the dataset
        cleaner = DataCleaner(str(uav_dir))
        
        # Clean the dataset with standard image size
        input_size = self.config.get('model', {}).get('input_size', [224, 224])
        target_size = (input_size[0], input_size[1])
        
        cleaned_df = cleaner.clean_dataset(
            str(annotations_file),
            target_size=target_size
        )
        
        # Save cleaned annotations
        cleaned_file = uav_dir / 'cleaned_annotations.csv'
        cleaned_df.to_csv(cleaned_file, index=False)
        
        # Print statistics
        self.logger.info("\nUAV Inspection Dataset Statistics:")
        self.logger.info(f"Original samples: {len(annotations_df)}")
        self.logger.info(f"Cleaned samples: {len(cleaned_df)}")
        self.logger.info(f"Removed samples: {len(annotations_df) - len(cleaned_df)}")
        self.logger.info("\nDefect distribution:")
        self.logger.info(cleaned_df['defect_class'].value_counts())
        
        return cleaned_df
    
    def combine_all_datasets(self) -> bool:
        """
        Combine all prepared datasets into a unified format.
        
        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info("Combining all datasets...")
        
        try:
            # Call the combine_datasets function
            combine_datasets(str(self.data_dir))
            
            # Clean the combined dataset
            clean_combined_dataset(str(self.data_dir))
            
            return True
        except Exception as e:
            self.logger.error(f"Error combining datasets: {str(e)}")
            return False
    
    def preprocess_combined_dataset(self) -> bool:
        """
        Apply preprocessing to the combined dataset.
        
        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info("Preprocessing the combined dataset...")
        
        try:
            combined_dir = self.data_dir / 'combined_dataset'
            
            # Check if combined dataset exists
            if not combined_dir.exists():
                self.logger.error("Combined dataset not found. Run combine_all_datasets first.")
                return False
            
            # Preprocess each split
            for split in ['train', 'val', 'test']:
                annotations_file = combined_dir / f'{split}_annotations_cleaned.csv'
                
                if not annotations_file.exists():
                    annotations_file = combined_dir / f'{split}_annotations.csv'
                
                if not annotations_file.exists():
                    self.logger.warning(f"No annotations found for {split} split. Skipping.")
                    continue
                
                self.logger.info(f"Preprocessing {split} split...")
                
                # Load annotations
                df = pd.read_csv(annotations_file)
                
                # Create preprocessor
                preprocessor = ImagePreprocessor(
                    data_dir=str(combined_dir),
                    output_dir=str(self.output_dir / split),
                    target_size=self.config.get('model', {}).get('input_size', [224, 224]),
                    normalize=self.config.get('image_processing', {}).get('preprocessing', {}).get('normalize', True),
                    augmentation=split == 'train' and self.config.get('image_processing', {}).get('preprocessing', {}).get('augmentation', False)
                )
                
                # Apply preprocessing
                preprocessed_df = preprocessor.preprocess_dataset(df)
                
                # Save preprocessed annotations
                preprocessed_file = self.output_dir / f'{split}_preprocessed.csv'
                preprocessed_df.to_csv(preprocessed_file, index=False)
                
                self.logger.info(f"Saved preprocessed {split} annotations to {preprocessed_file}")
            
            # Save dataset info
            dataset_info = {
                'splits': ['train', 'val', 'test'],
                'num_classes': len(pd.read_csv(combined_dir / 'train_annotations.csv')['defect_class'].unique()),
                'class_mapping': {cls: idx for idx, cls in enumerate(pd.read_csv(combined_dir / 'train_annotations.csv')['defect_class'].unique())},
                'preprocessing': {
                    'input_size': self.config.get('model', {}).get('input_size', [224, 224]),
                    'normalize': self.config.get('image_processing', {}).get('preprocessing', {}).get('normalize', True),
                    'augmentation': self.config.get('image_processing', {}).get('preprocessing', {}).get('augmentation', False)
                }
            }
            
            # Save dataset info
            with open(self.output_dir / 'dataset_info.yaml', 'w') as f:
                yaml.dump(dataset_info, f)
            
            return True
        except Exception as e:
            self.logger.error(f"Error preprocessing dataset: {str(e)}")
            return False
    
    def run_full_pipeline(self) -> bool:
        """
        Run the complete data processing pipeline.
        
        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info("Starting the complete data processing pipeline...")
        
        # Step 1: Prepare crack detection data
        crack_df = self.prepare_crack_detection_data()
        if crack_df.empty:
            self.logger.warning("Failed to prepare crack detection data. Continuing...")
        
        # Step 2: Prepare UAV inspection data
        uav_df = self.prepare_uav_inspection_data()
        
        # Step 3: Combine datasets
        if not self.combine_all_datasets():
            self.logger.error("Failed to combine datasets.")
            return False
        
        # Step 4: Preprocess combined dataset
        if not self.preprocess_combined_dataset():
            self.logger.error("Failed to preprocess dataset.")
            return False
        
        self.logger.info("Data processing pipeline completed successfully!")
        return True

if __name__ == "__main__":
    # Create and run the data pipeline
    pipeline = DataPipeline()
    pipeline.run_full_pipeline() 