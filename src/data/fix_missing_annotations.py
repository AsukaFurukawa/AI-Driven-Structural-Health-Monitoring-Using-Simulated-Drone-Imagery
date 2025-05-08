#!/usr/bin/env python3
"""
Fix missing annotations utility.

This script identifies images without annotations and creates missing annotations
based on folder structure and naming conventions.
"""
import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml

# Add parent directory to the Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fix_annotations.log')
    ]
)
logger = logging.getLogger(__name__)

def find_unannotated_images(dataset_dir, annotations_file):
    """
    Find images that exist in the dataset but are not in the annotations file.
    
    Args:
        dataset_dir (Path): Path to the dataset directory
        annotations_file (Path): Path to the annotations CSV file
        
    Returns:
        tuple: (unannotated_images, existing_annotations)
            - unannotated_images: List of image paths without annotations
            - existing_annotations: DataFrame of existing annotations
    """
    # Load existing annotations if file exists
    if annotations_file.exists():
        try:
            annotations_df = pd.read_csv(annotations_file)
            logger.info(f"Loaded {len(annotations_df)} existing annotations from {annotations_file}")
            
            # Debug: Print the first few annotation paths
            if not annotations_df.empty:
                logger.info(f"Sample annotation paths: {annotations_df['image_path'].iloc[:5].tolist()}")
        except Exception as e:
            logger.error(f"Error loading annotations file: {str(e)}")
            annotations_df = pd.DataFrame(columns=['image_path', 'defect_class', 'has_defect', 'dataset_source'])
    else:
        logger.warning(f"Annotations file {annotations_file} not found, creating new.")
        annotations_df = pd.DataFrame(columns=['image_path', 'defect_class', 'has_defect', 'dataset_source'])
    
    # Get all image files in the dataset
    all_images = []
    
    # Check common image directories based on dataset structure
    for class_dir in ['Positive', 'Negative', 'images']:
        dir_path = dataset_dir / class_dir
        if dir_path.exists():
            for img_path in dir_path.glob('**/*.jpg'):
                all_images.append(str(img_path.relative_to(dataset_dir)))
            for img_path in dir_path.glob('**/*.png'):
                all_images.append(str(img_path.relative_to(dataset_dir)))
            for img_path in dir_path.glob('**/*.jpeg'):
                all_images.append(str(img_path.relative_to(dataset_dir)))
                
    logger.info(f"Found {len(all_images)} total images in {dataset_dir}")
    
    # Debug: Print some sample image paths
    if all_images:
        logger.info(f"Sample image paths: {all_images[:5]}")
    
    # Check which images are already annotated
    annotated_images = annotations_df['image_path'].tolist() if not annotations_df.empty else []
    
    # Debug: Try to match path formats
    # Convert Windows backslashes to forward slashes for consistent comparison
    normalized_annotated = [path.replace('\\', '/') for path in annotated_images]
    normalized_all_images = [path.replace('\\', '/') for path in all_images]
    
    # Find unannotated images using normalized paths
    unannotated_images = [img for img in all_images if img.replace('\\', '/') not in normalized_annotated]
    
    logger.info(f"Found {len(unannotated_images)} unannotated images")
    
    return unannotated_images, annotations_df

def generate_annotations(unannotated_images, dataset_name, dataset_dir):
    """
    Generate annotations for unannotated images based on folder structure and naming conventions.
    
    Args:
        unannotated_images (list): List of unannotated image paths
        dataset_name (str): Name of the dataset for source tracking
        dataset_dir (Path): Directory of the dataset
        
    Returns:
        list: List of annotation dictionaries
    """
    new_annotations = []
    
    for img_path in tqdm(unannotated_images, desc="Generating annotations"):
        # Create an annotation entry
        annotation = {
            'image_path': img_path,
            'dataset_source': dataset_name.lower()
        }
        
        # Determine defect class based on folder/filename
        path = Path(img_path)
        
        # Check if image is in Positive or Negative directory or has 'crack' or 'no_crack' in name
        if 'Positive' in img_path or 'crack_' in path.stem.lower() or 'crack' in path.stem.lower():
            annotation['defect_class'] = 'Crack'
            annotation['has_defect'] = True
        elif 'Negative' in img_path or 'no_crack' in path.stem.lower() or 'nocrack' in path.stem.lower():
            annotation['defect_class'] = 'No_Defect'
            annotation['has_defect'] = False
        else:
            # If can't determine from path, make an educated guess based on image name
            if any(kw in path.stem.lower() for kw in ['crack', 'defect', 'damage', 'broken']):
                annotation['defect_class'] = 'Crack'
                annotation['has_defect'] = True
            else:
                annotation['defect_class'] = 'No_Defect'
                annotation['has_defect'] = False
        
        new_annotations.append(annotation)
    
    return new_annotations

def update_annotations_file(annotations_df, new_annotations, output_file):
    """
    Update the annotations file with newly generated annotations.
    
    Args:
        annotations_df (DataFrame): Existing annotations
        new_annotations (list): New annotation dictionaries to add
        output_file (Path): Path to save the updated annotations
        
    Returns:
        DataFrame: Combined annotations
    """
    # Convert new annotations to DataFrame
    new_df = pd.DataFrame(new_annotations)
    
    # Combine with existing annotations
    if annotations_df.empty:
        combined_df = new_df
    else:
        combined_df = pd.concat([annotations_df, new_df], ignore_index=True)
    
    # Save the combined annotations
    combined_df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(combined_df)} annotations to {output_file}")
    
    return combined_df

def process_dataset(dataset_dir, dataset_name):
    """
    Process a dataset to fix missing annotations.
    
    Args:
        dataset_dir (Path): Path to the dataset directory
        dataset_name (str): Name of the dataset
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Processing {dataset_name} dataset at {dataset_dir}")
    
    try:
        # Identify annotation files
        initial_annotations = dataset_dir / 'initial_annotations.csv'
        cleaned_annotations = dataset_dir / 'cleaned_annotations.csv'
        
        # Choose the appropriate annotations file to update
        annotations_file = cleaned_annotations if cleaned_annotations.exists() else initial_annotations
        
        # Find unannotated images
        unannotated_images, existing_annotations = find_unannotated_images(dataset_dir, annotations_file)
        
        if not unannotated_images:
            logger.info(f"No unannotated images found in {dataset_name} dataset. All images already have annotations.")
            return True
        
        # Generate annotations for unannotated images
        new_annotations = generate_annotations(unannotated_images, dataset_name, dataset_dir)
        
        # Update annotations file
        combined_df = update_annotations_file(existing_annotations, new_annotations, annotations_file)
        
        # If we updated initial_annotations.csv, also update cleaned_annotations.csv
        if annotations_file == initial_annotations and not cleaned_annotations.exists():
            combined_df.to_csv(cleaned_annotations, index=False)
            logger.info(f"Also created {cleaned_annotations} with {len(combined_df)} entries")
        
        return True
    except Exception as e:
        logger.error(f"Error processing {dataset_name} dataset: {str(e)}")
        return False

def fix_all_datasets(base_dir):
    """
    Fix missing annotations for all datasets.
    
    Args:
        base_dir (Path): Base directory containing all datasets
        
    Returns:
        bool: True if successful, False otherwise
    """
    success = True
    
    # Process each dataset directory
    for dataset_dir in base_dir.glob('*'):
        if dataset_dir.is_dir() and dataset_dir.name not in ['.git', '__pycache__']:
            dataset_result = process_dataset(dataset_dir, dataset_dir.name)
            success = success and dataset_result
    
    return success

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fix missing annotations in datasets")
    
    parser.add_argument("--data_dir", type=str, default="uploads/datasets",
                       help="Base directory containing the datasets")
    
    parser.add_argument("--dataset", type=str, default=None,
                       help="Process only a specific dataset (e.g., 'Crack_Detection')")
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Resolve data directory path
    base_dir = Path(__file__).parent.parent.parent / args.data_dir
    
    if not base_dir.exists():
        logger.error(f"Data directory not found: {base_dir}")
        return 1
    
    logger.info(f"Processing datasets in {base_dir}")
    
    if args.dataset:
        # Process only the specified dataset
        dataset_dir = base_dir / args.dataset
        if not dataset_dir.exists():
            logger.error(f"Dataset directory not found: {dataset_dir}")
            return 1
        
        success = process_dataset(dataset_dir, args.dataset)
    else:
        # Process all datasets
        success = fix_all_datasets(base_dir)
    
    if success:
        logger.info("All annotations fixed successfully")
        return 0
    else:
        logger.error("There were errors fixing annotations")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 