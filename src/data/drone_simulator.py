"""
Simulate drone imagery by applying transformations to existing images.

This module provides functions to transform regular structural images
to appear as if they were captured by drones from different angles.
"""
import os
import random
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List, Optional
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("drone_simulator")

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate an image by a specified angle.
    
    Args:
        image: Input image (numpy array)
        angle: Rotation angle in degrees
        
    Returns:
        Rotated image
    """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new image dimensions
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    
    # Adjust rotation matrix
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]
    
    # Rotate image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, 
                                  borderValue=(0, 0, 0))
    
    return rotated_image

def apply_perspective_transform(image: np.ndarray, strength: float = 0.2) -> np.ndarray:
    """
    Apply perspective transformation to simulate drone viewpoint.
    
    Args:
        image: Input image
        strength: Strength of the perspective effect (0.0-1.0)
        
    Returns:
        Perspective transformed image
    """
    height, width = image.shape[:2]
    
    # Define source points
    src_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    
    # Define destination points with perspective distortion
    offset_x = int(width * strength * random.uniform(0.5, 1.5))
    offset_y = int(height * strength * random.uniform(0.5, 1.5))
    
    # Randomly choose perspective type
    perspective_type = random.randint(0, 3)
    
    if perspective_type == 0:  # Top view
        dst_points = np.array([
            [offset_x, offset_y],
            [width - offset_x, offset_y],
            [width - offset_x // 2, height - offset_y // 2],
            [offset_x // 2, height - offset_y // 2]
        ], dtype=np.float32)
    elif perspective_type == 1:  # Side view 1
        dst_points = np.array([
            [offset_x, offset_y],
            [width - offset_x // 2, offset_y // 2],
            [width - offset_x // 3, height - offset_y // 3],
            [offset_x // 2, height - offset_y]
        ], dtype=np.float32)
    elif perspective_type == 2:  # Side view 2
        dst_points = np.array([
            [offset_x // 2, offset_y // 2],
            [width - offset_x, offset_y],
            [width - offset_x // 2, height - offset_y // 2],
            [offset_x, height - offset_y]
        ], dtype=np.float32)
    else:  # Bottom view
        dst_points = np.array([
            [offset_x // 2, offset_y // 2],
            [width - offset_x // 2, offset_y // 2],
            [width - offset_x, height - offset_y],
            [offset_x, height - offset_y]
        ], dtype=np.float32)
    
    # Calculate perspective transform matrix
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply transformation
    warped_image = cv2.warpPerspective(image, perspective_matrix, (width, height), 
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0))
    
    return warped_image

def adjust_lighting(image: np.ndarray, brightness: float = 0.0, contrast: float = 1.0) -> np.ndarray:
    """
    Adjust image lighting to simulate different drone lighting conditions.
    
    Args:
        image: Input image
        brightness: Brightness adjustment (-1.0 to 1.0)
        contrast: Contrast adjustment (0.0 to 2.0)
        
    Returns:
        Adjusted image
    """
    # Convert to float for processing
    image_float = image.astype(np.float32) / 255.0
    
    # Apply brightness adjustment
    image_float = image_float + brightness
    
    # Apply contrast adjustment
    image_float = (image_float - 0.5) * contrast + 0.5
    
    # Clip values to [0, 1]
    image_float = np.clip(image_float, 0, 1)
    
    # Convert back to uint8
    adjusted_image = (image_float * 255).astype(np.uint8)
    
    return adjusted_image

def add_drone_artifacts(image: np.ndarray, artifact_type: str = "random") -> np.ndarray:
    """
    Add simulated drone artifacts like motion blur, noise, etc.
    
    Args:
        image: Input image
        artifact_type: Type of artifact to add ("blur", "noise", "both", or "random")
        
    Returns:
        Image with artifacts
    """
    if artifact_type == "random":
        artifact_type = random.choice(["blur", "noise", "both", "none"])
    
    if artifact_type == "none":
        return image
    
    processed_image = image.copy()
    
    # Apply motion blur
    if artifact_type in ["blur", "both"]:
        kernel_size = random.randint(3, 7)
        kernel = np.zeros((kernel_size, kernel_size))
        
        # Random direction for motion blur
        if random.random() > 0.5:
            # Horizontal blur
            kernel[kernel_size // 2, :] = 1.0
        else:
            # Vertical blur
            kernel[:, kernel_size // 2] = 1.0
            
        kernel = kernel / kernel_size
        processed_image = cv2.filter2D(processed_image, -1, kernel)
    
    # Apply noise
    if artifact_type in ["noise", "both"]:
        noise_level = random.uniform(5, 20)
        noise = np.random.normal(0, noise_level, processed_image.shape).astype(np.int32)
        processed_image = np.clip(processed_image + noise, 0, 255).astype(np.uint8)
    
    return processed_image

def simulate_drone_image(
    image: np.ndarray, 
    rotation_angle: Optional[float] = None,
    perspective_strength: Optional[float] = None,
    brightness: Optional[float] = None,
    contrast: Optional[float] = None,
    artifact_type: str = "random"
) -> np.ndarray:
    """
    Apply multiple transformations to simulate a drone image.
    
    Args:
        image: Input image
        rotation_angle: Rotation angle (if None, a random angle is chosen)
        perspective_strength: Perspective transform strength (if None, a random value is chosen)
        brightness: Brightness adjustment (if None, a random value is chosen)
        contrast: Contrast adjustment (if None, a random value is chosen)
        artifact_type: Type of artifact to add
        
    Returns:
        Simulated drone image
    """
    # Set random values if parameters are None
    if rotation_angle is None:
        rotation_angle = random.uniform(-30, 30)
    
    if perspective_strength is None:
        perspective_strength = random.uniform(0.1, 0.3)
    
    if brightness is None:
        brightness = random.uniform(-0.2, 0.2)
    
    if contrast is None:
        contrast = random.uniform(0.8, 1.2)
    
    # Apply transformations
    processed_image = image.copy()
    
    # Step 1: Rotate image
    processed_image = rotate_image(processed_image, rotation_angle)
    
    # Step 2: Apply perspective transformation
    processed_image = apply_perspective_transform(processed_image, perspective_strength)
    
    # Step 3: Adjust lighting
    processed_image = adjust_lighting(processed_image, brightness, contrast)
    
    # Step 4: Add drone artifacts
    processed_image = add_drone_artifacts(processed_image, artifact_type)
    
    return processed_image

def process_directory(
    input_dir: str, 
    output_dir: str, 
    transformations_per_image: int = 3,
    maintain_class_structure: bool = True
) -> List[str]:
    """
    Process all images in a directory to create simulated drone images.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save processed images
        transformations_per_image: Number of variations to create per input image
        maintain_class_structure: Whether to maintain the same folder structure
        
    Returns:
        List of paths to generated images
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    if maintain_class_structure:
        # Get all subdirectories (classes)
        subdirs = [d for d in input_path.iterdir() if d.is_dir()]
        
        for subdir in subdirs:
            subdir_output = output_path / subdir.name
            subdir_output.mkdir(exist_ok=True)
            
            # Get images in this class
            for ext in image_extensions:
                image_files.extend([
                    (f, subdir_output) for f in subdir.glob(f'**/*{ext}')
                ])
    else:
        # Get all images regardless of structure
        for ext in image_extensions:
            image_files.extend([
                (f, output_path) for f in input_path.glob(f'**/*{ext}')
            ])
    
    generated_image_paths = []
    
    # Process each image
    for img_file, out_dir in tqdm(image_files, desc="Processing images"):
        try:
            # Read image
            image = cv2.imread(str(img_file))
            if image is None:
                logger.warning(f"Could not read image: {img_file}")
                continue
            
            # Create transformations
            for i in range(transformations_per_image):
                # Apply random transformations
                drone_image = simulate_drone_image(image)
                
                # Generate output filename
                base_name = img_file.stem
                output_filename = f"{base_name}_drone_{i+1}{img_file.suffix}"
                output_path = out_dir / output_filename
                
                # Save transformed image
                cv2.imwrite(str(output_path), drone_image)
                generated_image_paths.append(str(output_path))
                
        except Exception as e:
            logger.error(f"Error processing {img_file}: {str(e)}")
    
    logger.info(f"Generated {len(generated_image_paths)} drone images")
    return generated_image_paths

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Simulate drone imagery from structural images")
    parser.add_argument("--input", required=True, help="Input directory containing images")
    parser.add_argument("--output", required=True, help="Output directory for simulated drone images")
    parser.add_argument("--variations", type=int, default=3, help="Number of variations per image")
    parser.add_argument("--maintain-structure", action="store_true", help="Maintain folder structure")
    
    args = parser.parse_args()
    
    process_directory(
        args.input,
        args.output,
        transformations_per_image=args.variations,
        maintain_class_structure=args.maintain_structure
    ) 