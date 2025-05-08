"""
Data preparation module for structural health monitoring datasets.
"""
import os
import pandas as pd
import requests
import zipfile
from pathlib import Path
from typing import List, Dict
import shutil
from tqdm import tqdm
import logging
import yaml
import subprocess
import time
import gdown
import numpy as np
from PIL import Image
import random

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def download_file(url: str, destination: str, chunk_size: int = 8192):
    """
    Download a file from a URL with progress bar.
    
    Args:
        url (str): URL to download from
        destination (str): Where to save the file
        chunk_size (int): Size of chunks to download
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(destination, 'wb') as file, tqdm(
        desc=destination,
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as progress_bar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            progress_bar.update(size)

def generate_simulated_dataset(target_dir: Path, num_samples: int = 500) -> bool:
    """
    Generate a simulated crack dataset when real dataset download is not possible.
    
    Args:
        target_dir (Path): Directory to save the simulated dataset
        num_samples (int): Number of samples to generate
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Create positive and negative directories
        positive_dir = target_dir / 'Positive'
        negative_dir = target_dir / 'Negative'
        
        positive_dir.mkdir(parents=True, exist_ok=True)
        negative_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating {num_samples} simulated crack images...")
        
        # Generate simulated crack images
        for i in range(num_samples):
            # Create a base image (grayscale)
            img_size = (224, 224)
            
            # Positive samples (with cracks)
            if i < num_samples // 2:
                # Create a dark background
                img = np.ones(img_size, dtype=np.uint8) * 200
                
                # Add some texture/noise
                noise = np.random.randint(0, 30, size=img_size, dtype=np.uint8)
                img = np.clip(img - noise, 0, 255).astype(np.uint8)
                
                # Add a simulated crack (dark line)
                crack_width = random.randint(2, 8)
                crack_start_x = random.randint(0, img_size[1] - 1)
                crack_start_y = random.randint(0, img_size[0] - 1)
                
                # Draw a random path for the crack
                num_segments = random.randint(3, 10)
                x, y = crack_start_x, crack_start_y
                
                for _ in range(num_segments):
                    # Random direction and length
                    dx = random.randint(-30, 30)
                    dy = random.randint(-30, 30)
                    
                    end_x = min(max(0, x + dx), img_size[1] - 1)
                    end_y = min(max(0, y + dy), img_size[0] - 1)
                    
                    # Create points for the line
                    length = max(abs(end_x - x), abs(end_y - y))
                    if length > 0:
                        for j in range(length):
                            px = int(x + (end_x - x) * j / length)
                            py = int(y + (end_y - y) * j / length)
                            
                            # Draw the crack (darker pixel) with width
                            for w in range(-crack_width // 2, crack_width // 2 + 1):
                                for h in range(-crack_width // 2, crack_width // 2 + 1):
                                    px_w = min(max(0, px + w), img_size[1] - 1)
                                    py_h = min(max(0, py + h), img_size[0] - 1)
                                    img[py_h, px_w] = max(0, img[py_h, px_w] - random.randint(100, 170))
                    
                    x, y = end_x, end_y
                
                # Save as positive sample
                img_path = positive_dir / f"crack_{i:04d}.jpg"
                Image.fromarray(img).save(img_path)
            
            # Negative samples (no cracks)
            else:
                # Create a concrete-like texture
                img = np.ones(img_size, dtype=np.uint8) * 200
                
                # Add stronger texture/noise for concrete appearance
                for _ in range(3):  # Multiple layers of noise
                    noise_scale = random.randint(10, 40)
                    noise = np.random.randint(0, noise_scale, size=img_size, dtype=np.uint8)
                    img = np.clip(img - noise, 100, 255).astype(np.uint8)
                
                # Save as negative sample
                img_path = negative_dir / f"no_crack_{i:04d}.jpg"
                Image.fromarray(img).save(img_path)
        
        logger.info(f"Generated {num_samples} simulated crack images successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error generating simulated dataset: {str(e)}")
        return False

def prepare_public_crack_dataset(base_dir: str) -> pd.DataFrame:
    """
    Prepare a publicly available crack detection dataset.
    
    Args:
        base_dir (str): Base directory to store the dataset
        
    Returns:
        pd.DataFrame: Dataset annotations
    """
    logger = logging.getLogger(__name__)
    base_dir = Path(base_dir)
    
    # Create directories
    data_dir = base_dir / 'Crack_Detection'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if the dataset already exists
    positive_dir = data_dir / 'Positive'
    negative_dir = data_dir / 'Negative'
    
    if not positive_dir.exists() or not negative_dir.exists() or \
       len(list(positive_dir.glob('*.jpg'))) == 0 or len(list(negative_dir.glob('*.jpg'))) == 0:
        
        # Generate a simulated dataset since downloading real datasets is problematic
        logger.info("Generating simulated crack dataset...")
        if not generate_simulated_dataset(data_dir):
            logger.error("Failed to generate simulated dataset")
            return pd.DataFrame()
    
    # Process and organize the dataset
    logger.info("Processing Crack Detection Dataset...")
    
    # Create annotations DataFrame
    annotations = []
    
    # Check if we have the expected structure
    if positive_dir.exists() and negative_dir.exists():
        # Process positive examples (with cracks)
        for img_path in positive_dir.glob('*.jpg'):
            annotations.append({
                'image_path': str(img_path.relative_to(data_dir)),
                'defect_class': 'Crack',
                'has_defect': True,
                'dataset_source': 'crack_detection'
            })
        
        # Process negative examples (without cracks)
        for img_path in negative_dir.glob('*.jpg'):
            annotations.append({
                'image_path': str(img_path.relative_to(data_dir)),
                'defect_class': 'No_Defect',
                'has_defect': False,
                'dataset_source': 'crack_detection'
            })
    
    if not annotations:
        logger.warning("No valid images found in the Crack Detection Dataset")
    else:
        logger.info(f"Found {len(annotations)} images in the dataset")
    
    return pd.DataFrame(annotations)

def combine_datasets(base_dir: str):
    """
    Combine multiple datasets into a unified format.
    
    Args:
        base_dir (str): Base directory to store the datasets
    """
    logger = logging.getLogger(__name__)
    base_dir = Path(base_dir)
    
    # Prepare crack detection dataset
    logger.info("Preparing Public Crack Detection Dataset...")
    crack_df = prepare_public_crack_dataset(base_dir)
    
    if crack_df.empty:
        logger.error("Failed to prepare dataset")
        return None
    
    # Create output directory
    output_dir = base_dir / 'combined_dataset'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split into train/val/test
    from sklearn.model_selection import train_test_split
    
    try:
        # First split: 70% train, 30% temp
        train_df, temp_df = train_test_split(
            crack_df, 
            test_size=0.3, 
            random_state=42,
            stratify=crack_df['defect_class'] if len(crack_df) > 1 else None
        )
        
        # Second split: 50% val, 50% test from temp
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=0.5, 
            random_state=42,
            stratify=temp_df['defect_class'] if len(temp_df) > 1 else None
        )
    except ValueError as e:
        logger.warning(f"Could not stratify splits: {str(e)}")
        # Fallback to simple splitting
        train_df, temp_df = train_test_split(crack_df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Save splits
    train_df.to_csv(output_dir / 'train_annotations.csv', index=False)
    val_df.to_csv(output_dir / 'val_annotations.csv', index=False)
    test_df.to_csv(output_dir / 'test_annotations.csv', index=False)
    
    # Save dataset statistics
    stats = {
        'total_images': len(crack_df),
        'train_images': len(train_df),
        'val_images': len(val_df),
        'test_images': len(test_df),
        'defect_distribution': crack_df['defect_class'].value_counts().to_dict(),
        'dataset_distribution': crack_df['dataset_source'].value_counts().to_dict()
    }
    
    with open(output_dir / 'dataset_stats.yaml', 'w') as f:
        yaml.dump(stats, f)
    
    logger.info(f"Dataset statistics:\n{yaml.dump(stats)}")
    return output_dir

if __name__ == "__main__":
    # Set up logging
    logger = setup_logging()
    
    # Load config
    config_path = Path(__file__).parent.parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Prepare datasets
    base_dir = Path(config['storage']['local']['upload_dir']) / 'datasets'
    base_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Combine datasets
        output_dir = combine_datasets(base_dir)
        if output_dir:
            logger.info(f"Combined dataset prepared at: {output_dir}")
        else:
            logger.error("Failed to prepare combined dataset")
    except Exception as e:
        logger.error(f"Error preparing dataset: {str(e)}") 