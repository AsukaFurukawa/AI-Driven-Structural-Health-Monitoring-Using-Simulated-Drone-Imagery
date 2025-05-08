"""
Script to generate synthetic data for testing, including simulated drone imagery.
"""
import os
import sys
import logging
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import drone simulator
from src.data.drone_simulator import simulate_drone_image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_crack_image(size=(224, 224), crack_probability=0.5):
    """
    Generate a synthetic image with or without a crack.
    
    Args:
        size (tuple): Image size (width, height)
        crack_probability (float): Probability of generating an image with a crack
        
    Returns:
        tuple: (image, has_crack)
    """
    # Create a blank image with random background
    image = Image.new('L', size)
    draw = ImageDraw.Draw(image)
    
    # Generate random background texture
    background = np.random.normal(128, 20, size).astype(np.uint8)
    image = Image.fromarray(background)
    draw = ImageDraw.Draw(image)
    
    # Randomly decide if this image should have a crack
    has_crack = random.random() < crack_probability
    
    if has_crack:
        # Generate random crack parameters
        start_x = random.randint(0, size[0] - 1)
        start_y = random.randint(0, size[1] - 1)
        length = random.randint(size[0] // 4, size[0] // 2)
        angle = random.uniform(0, 360)
        
        # Draw the crack
        end_x = start_x + int(length * np.cos(np.radians(angle)))
        end_y = start_y + int(length * np.sin(np.radians(angle)))
        
        # Draw main crack line
        draw.line([(start_x, start_y), (end_x, end_y)], fill=0, width=random.randint(1, 3))
        
        # Add some random branches
        num_branches = random.randint(1, 3)
        for _ in range(num_branches):
            branch_start_x = random.randint(min(start_x, end_x), max(start_x, end_x))
            branch_start_y = random.randint(min(start_y, end_y), max(start_y, end_y))
            branch_length = random.randint(10, 30)
            branch_angle = angle + random.uniform(-45, 45)
            
            branch_end_x = branch_start_x + int(branch_length * np.cos(np.radians(branch_angle)))
            branch_end_y = branch_start_y + int(branch_length * np.sin(np.radians(branch_angle)))
            
            draw.line([(branch_start_x, branch_start_y), (branch_end_x, branch_end_y)], fill=0, width=1)
    
    # Convert grayscale to RGB for drone simulation
    rgb_image = Image.new('RGB', size)
    rgb_image.paste(image)
    
    return rgb_image, has_crack

def generate_dataset(num_images=1000, output_dir=None, include_drone_views=True, drone_variations=2):
    """
    Generate a synthetic dataset.
    
    Args:
        num_images (int): Number of images to generate
        output_dir (str): Output directory for the dataset
        include_drone_views (bool): Whether to include simulated drone imagery
        drone_variations (int): Number of drone variations per image
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    
    # Create output directories
    positive_dir = Path(output_dir) / "positive"
    negative_dir = Path(output_dir) / "negative"
    
    if include_drone_views:
        drone_positive_dir = Path(output_dir) / "drone_positive"
        drone_negative_dir = Path(output_dir) / "drone_negative"
        drone_positive_dir.mkdir(parents=True, exist_ok=True)
        drone_negative_dir.mkdir(parents=True, exist_ok=True)
    
    positive_dir.mkdir(parents=True, exist_ok=True)
    negative_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate images
    logger.info(f"Generating {num_images} synthetic images...")
    for i in tqdm(range(num_images)):
        # Generate image
        image, has_crack = generate_crack_image()
        
        # Save original image
        output_path = (positive_dir if has_crack else negative_dir) / f"image_{i:04d}.png"
        image.save(output_path)
        
        # Generate and save drone view variations if requested
        if include_drone_views:
            for j in range(drone_variations):
                # Convert PIL image to numpy array for drone simulation
                img_array = np.array(image)
                
                # Apply drone simulation
                drone_img_array = simulate_drone_image(img_array)
                
                # Convert back to PIL Image
                drone_image = Image.fromarray(drone_img_array)
                
                # Save drone image
                drone_dir = drone_positive_dir if has_crack else drone_negative_dir
                drone_path = drone_dir / f"drone_image_{i:04d}_{j+1}.png"
                drone_image.save(drone_path)
    
    # Count generated images
    positive_count = len(list(positive_dir.glob('*.png')))
    negative_count = len(list(negative_dir.glob('*.png')))
    
    logger.info(f"Generated {positive_count} positive and {negative_count} negative images")
    
    if include_drone_views:
        drone_positive_count = len(list(drone_positive_dir.glob('*.png')))
        drone_negative_count = len(list(drone_negative_dir.glob('*.png')))
        logger.info(f"Generated {drone_positive_count} drone positive and {drone_negative_count} drone negative images")
        
    total_count = positive_count + negative_count
    if include_drone_views:
        total_count += drone_positive_count + drone_negative_count
        
    logger.info(f"Total images generated: {total_count}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic data")
    parser.add_argument("--num_images", type=int, default=1000, help="Number of images to generate")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--include_drone", action="store_true", help="Include drone imagery")
    parser.add_argument("--drone_variations", type=int, default=2, help="Number of drone variations per image")
    
    args = parser.parse_args()
    
    generate_dataset(
        num_images=args.num_images, 
        output_dir=args.output_dir,
        include_drone_views=args.include_drone,
        drone_variations=args.drone_variations
    ) 