"""
Script to process raw data for crack detection, including simulated drone imagery.
"""
import os
import sys
import logging
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_directory_structure():
    """Create necessary directories for data processing."""
    base_dir = Path(__file__).parent.parent.parent
    dirs = [
        base_dir / "data" / "raw",
        base_dir / "data" / "processed" / "positive",
        base_dir / "data" / "processed" / "negative",
        base_dir / "data" / "processed" / "drone_positive",
        base_dir / "data" / "processed" / "drone_negative"
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def process_image(image_path, output_path, target_size=(224, 224), convert_grayscale=True):
    """
    Process a single image.
    
    Args:
        image_path (Path): Path to input image
        output_path (Path): Path to save processed image
        target_size (tuple): Target size for resizing
        convert_grayscale (bool): Whether to convert to grayscale
    """
    try:
        # Open image
        with Image.open(image_path) as img:
            # Convert to grayscale if requested
            if convert_grayscale:
                img = img.convert('L')
            else:
                # Ensure image is RGB for consistency
                img = img.convert('RGB')
            
            # Resize
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Save processed image
            img.save(output_path, quality=95)
            
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")

def process_dataset(include_drone=True):
    """
    Process the raw dataset.
    
    Args:
        include_drone (bool): Whether to include drone imagery in processing
    """
    base_dir = Path(__file__).parent.parent.parent
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    
    # Check if raw data exists
    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        logger.error("No raw data found. Please place your dataset in the data/raw directory.")
        return
    
    # Process standard images
    logger.info("Processing standard images...")
    for root, _, files in os.walk(raw_dir):
        # Skip drone directories if they're in raw_dir but we're processing separately
        if include_drone and ('drone_positive' in root or 'drone_negative' in root):
            continue
            
        for file in tqdm(files, desc=f"Processing images in {Path(root).name}"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Determine if image is positive or negative
                is_positive = 'positive' in root.lower()
                output_dir = processed_dir / ('positive' if is_positive else 'negative')
                
                # Process and save image
                input_path = Path(root) / file
                output_path = output_dir / file
                process_image(input_path, output_path)
    
    # Process drone images if included
    if include_drone:
        logger.info("Processing drone images...")
        drone_dirs = [
            (raw_dir / "drone_positive", processed_dir / "drone_positive"),
            (raw_dir / "drone_negative", processed_dir / "drone_negative")
        ]
        
        for src_dir, dst_dir in drone_dirs:
            if src_dir.exists():
                for file in tqdm(os.listdir(src_dir), desc=f"Processing images in {src_dir.name}"):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        input_path = src_dir / file
                        output_path = dst_dir / file
                        # Keep drone images in RGB
                        process_image(input_path, output_path, convert_grayscale=False)
    
    # Count processed images
    positive_count = len(list((processed_dir / 'positive').glob('*')))
    negative_count = len(list((processed_dir / 'negative').glob('*')))
    
    total = positive_count + negative_count
    logger.info(f"Standard image processing completed: {positive_count} positive and {negative_count} negative images")
    
    if include_drone:
        drone_positive_count = len(list((processed_dir / 'drone_positive').glob('*')))
        drone_negative_count = len(list((processed_dir / 'drone_negative').glob('*')))
        drone_total = drone_positive_count + drone_negative_count
        
        logger.info(f"Drone image processing completed: {drone_positive_count} drone positive and {drone_negative_count} drone negative images")
        total += drone_total
    
    logger.info(f"Total processed images: {total}")

def main():
    """Main function to process data."""
    try:
        # Parse arguments
        import argparse
        parser = argparse.ArgumentParser(description="Process data for crack detection")
        parser.add_argument("--no-drone", action="store_true", help="Exclude drone imagery")
        args = parser.parse_args()
        
        # Create directory structure
        create_directory_structure()
        
        # Process dataset
        process_dataset(include_drone=not args.no_drone)
        
        logger.info("Data processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error during data processing: {str(e)}")
        raise

if __name__ == "__main__":
    main() 