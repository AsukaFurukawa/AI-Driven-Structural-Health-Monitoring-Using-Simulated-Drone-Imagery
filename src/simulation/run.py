"""
Script to run the drone simulation.
"""
import os
import sys
import logging
import yaml
import time
import random
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def simulate_drone_movement(config):
    """
    Simulate drone movement and image capture.
    
    Args:
        config (dict): Configuration dictionary
    """
    # Get simulation parameters
    sim_config = config.get('simulation', {})
    duration = sim_config.get('duration', 3600)  # Default 1 hour
    interval = sim_config.get('capture_interval', 10)  # Default 10 seconds
    output_dir = Path(__file__).parent.parent.parent / sim_config.get('output_dir', 'data/raw')
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Simulation parameters
    start_time = time.time()
    end_time = start_time + duration
    current_time = start_time
    
    logger.info(f"Starting drone simulation for {duration} seconds")
    
    while current_time < end_time:
        # Simulate drone position
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        z = random.uniform(10, 30)
        
        # Simulate image capture
        timestamp = datetime.fromtimestamp(current_time).strftime('%Y%m%d_%H%M%S')
        image_path = output_dir / f"drone_image_{timestamp}.png"
        
        # Generate synthetic image
        from src.data.generate_synthetic_data import generate_crack_image
        image, has_crack = generate_crack_image()
        image.save(image_path)
        
        logger.info(f"Captured image at position ({x:.1f}, {y:.1f}, {z:.1f}) - {'Crack detected' if has_crack else 'No crack'}")
        
        # Wait for next capture
        time.sleep(interval)
        current_time = time.time()
    
    logger.info("Drone simulation completed")

def main():
    """Main function to run the drone simulation."""
    try:
        # Load configuration
        config = load_config()
        
        # Run simulation
        simulate_drone_movement(config)
        
        logger.info("Drone simulation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during drone simulation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 