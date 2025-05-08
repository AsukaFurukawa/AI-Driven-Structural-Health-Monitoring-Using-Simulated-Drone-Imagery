"""
Setup script for AI-Driven Structural Health Monitoring.
"""
import os
import sys
import logging
import subprocess
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories for the project."""
    logger.info("Creating directory structure...")
    
    # Base directory
    base_dir = Path(__file__).parent
    
    # Directories to create
    directories = [
        base_dir / "data" / "raw",
        base_dir / "data" / "processed" / "positive",
        base_dir / "data" / "processed" / "negative",
        base_dir / "checkpoints",
        base_dir / "logs",
        base_dir / "uploads" / "api_uploads",
        base_dir / "uploads" / "web_uploads",
    ]
    
    # Create directories
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def install_dependencies():
    """Install required dependencies."""
    logger.info("Installing dependencies...")
    
    try:
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing dependencies: {str(e)}")
        logger.info("Trying to continue with available dependencies...")
        return False

def generate_synthetic_data():
    """Generate synthetic data for testing."""
    logger.info("Generating synthetic data...")
    
    try:
        # Generate data
        data_generator_path = Path(__file__).parent / "src" / "data" / "generate_synthetic_data.py"
        subprocess.check_call([sys.executable, str(data_generator_path)])
        logger.info("Synthetic data generated successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error generating synthetic data: {str(e)}")
        logger.warning("Continuing setup without synthetic data generation...")

def process_data():
    """Process raw data for training."""
    logger.info("Processing data...")
    
    try:
        # Process data
        data_processor_path = Path(__file__).parent / "src" / "data" / "process_data.py"
        subprocess.check_call([sys.executable, str(data_processor_path)])
        logger.info("Data processed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error processing data: {str(e)}")
        logger.warning("Continuing setup without data processing...")

def create_empty_model_checkpoint():
    """Create an empty model checkpoint to allow the API to start."""
    logger.info("Creating empty model checkpoint...")
    
    try:
        import torch
        from src.models.crack_detection_model import get_model
        
        # Create model
        model = get_model(num_classes=2, pretrained=False)
        
        # Create checkpoint
        checkpoint = {
            'epoch': 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': None,
            'val_loss': float('inf'),
            'val_acc': 0.0,
            'config': {}
        }
        
        # Save checkpoint
        checkpoint_dir = Path(__file__).parent / "checkpoints"
        checkpoint_path = checkpoint_dir / "model_best.pt"
        
        if not checkpoint_path.exists():
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Created empty model checkpoint at {checkpoint_path}")
    except Exception as e:
        logger.error(f"Error creating empty model checkpoint: {str(e)}")
        logger.warning("API may not start without a model checkpoint")

def main():
    """Main setup function."""
    try:
        logger.info("Starting setup...")
        
        # Create directories
        create_directories()
        
        # Install dependencies
        dependencies_installed = install_dependencies()
        
        # If dependencies failed, warn but continue
        if not dependencies_installed:
            logger.warning("Some dependencies could not be installed. The application may not function correctly.")
            logger.warning("You may need to manually install missing dependencies.")
        
        # Generate synthetic data
        generate_synthetic_data()
        
        # Process data
        process_data()
        
        # Create empty model checkpoint
        create_empty_model_checkpoint()
        
        logger.info("Setup completed")
        logger.info("You can now run the API with 'python run_api.py' or the full system with 'python run.py'")
        
    except Exception as e:
        logger.error(f"Error during setup: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 