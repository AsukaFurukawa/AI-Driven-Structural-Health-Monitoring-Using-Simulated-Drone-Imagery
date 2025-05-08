#!/usr/bin/env python3
"""
Fix missing annotations specifically for the UAV_Inspection dataset.

This script identifies unannotated images in the UAV_Inspection dataset and adds
appropriate annotations for them based on folder structure and image names.
"""
import os
import sys
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add parent directory to the Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fix_uav_annotations.log')
    ]
)
logger = logging.getLogger(__name__)

def fix_uav_annotations():
    """
    Fix missing annotations in the UAV_Inspection dataset.
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Define paths
    base_dir = Path(__file__).parent.parent.parent
    uav_dir = base_dir / "uploads/datasets/UAV_Inspection"
    annotation_file = uav_dir / "cleaned_annotations.csv"
    
    if not uav_dir.exists():
        logger.error(f"UAV_Inspection dataset directory not found: {uav_dir}")
        return False
    
    # Load existing annotations
    if annotation_file.exists():
        try:
            existing_df = pd.read_csv(annotation_file)
            logger.info(f"Loaded {len(existing_df)} existing annotations")
            
            # Debug: Show first few annotations
            if not existing_df.empty:
                logger.info(f"Sample existing annotations: {existing_df['image_path'].iloc[:3].tolist()}")
        except Exception as e:
            logger.error(f"Error loading annotations: {str(e)}")
            return False
    else:
        logger.warning("No existing annotations file found. Creating a new one.")
        existing_df = pd.DataFrame(columns=['image_path', 'defect_class', 'has_defect', 'dataset_source'])
    
    # Find all images in the dataset, including subdirectories
    all_images = []
    
    # Use glob to find all image files
    for ext in ['.jpg', '.jpeg', '.png']:
        for img_path in uav_dir.glob(f'**/*{ext}'):
            if img_path.is_file():
                rel_path = img_path.relative_to(uav_dir)
                all_images.append(str(rel_path))
        
        # Also try uppercase extensions
        for img_path in uav_dir.glob(f'**/*{ext.upper()}'):
            if img_path.is_file():
                rel_path = img_path.relative_to(uav_dir)
                all_images.append(str(rel_path))
    
    logger.info(f"Found {len(all_images)} total images in UAV_Inspection")
    
    # Debug: Show sample image paths
    if all_images:
        logger.info(f"Sample image paths: {all_images[:5]}")
    
    # Get paths that already have annotations (normalize for comparison)
    annotated_paths = [path.replace('\\', '/').lower() for path in existing_df['image_path']]
    
    # Debug: Show normalized annotated paths
    logger.info(f"Sample normalized annotated paths: {annotated_paths[:3] if annotated_paths else 'None'}")
    
    # Find unannotated images (normalize paths for comparison)
    unannotated_images = []
    for img in all_images:
        normalized_img = img.replace('\\', '/').lower()
        if normalized_img not in annotated_paths:
            unannotated_images.append(img)
    
    logger.info(f"Found {len(unannotated_images)} unannotated images")
    
    # Debug: Show some unannotated images
    if unannotated_images:
        logger.info(f"Sample unannotated images: {unannotated_images[:10]}")
    
    if not unannotated_images:
        logger.info("No unannotated images found. Nothing to do.")
        return True
    
    # Generate annotations for unannotated images
    new_annotations = []
    
    for img_path in tqdm(unannotated_images, desc="Generating annotations"):
        # Create annotation
        annotation = {
            'image_path': img_path,
            'dataset_source': 'uav_inspection'
        }
        
        # Determine class based on path
        path = Path(img_path)
        path_str = str(path).lower()
        
        # Check for common crack indicators in file or path
        if ('crack' in path_str or 'damage' in path_str or 'defect' in path_str) and 'no_crack' not in path_str:
            annotation['defect_class'] = 'Crack'
            annotation['has_defect'] = True
        else:
            annotation['defect_class'] = 'No_Defect'
            annotation['has_defect'] = False
        
        new_annotations.append(annotation)
    
    # Create DataFrame for new annotations
    new_df = pd.DataFrame(new_annotations)
    
    # Debug: Show some of the new annotations
    if not new_df.empty:
        logger.info(f"Generated {len(new_df)} new annotations")
        logger.info(f"Sample new annotations: {new_df.iloc[:3].to_dict('records')}")
    
    # Combine with existing annotations
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Save combined annotations
    combined_df.to_csv(annotation_file, index=False)
    logger.info(f"Saved {len(combined_df)} total annotations to {annotation_file}")
    
    # Save a backup of the initial annotations file if it doesn't exist
    initial_annotations = uav_dir / "initial_annotations.csv"
    if not initial_annotations.exists():
        combined_df.to_csv(initial_annotations, index=False)
        logger.info(f"Also saved a copy to {initial_annotations}")
    
    return True

def main():
    """Main function."""
    logger.info("Starting UAV_Inspection annotation fixing")
    
    success = fix_uav_annotations()
    
    if success:
        logger.info("Successfully fixed UAV_Inspection annotations")
        return 0
    else:
        logger.error("Failed to fix UAV_Inspection annotations")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 