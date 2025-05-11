"""
Script to train the crack detection model.
"""
import os
import sys
import logging
import yaml
import torch
import argparse
from pathlib import Path
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.crack_detection_model import get_model
from src.models.trainer import Trainer
from src.data.dataset import create_dataloaders

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

def train_model(model_type="cnn", include_drone=True):
    """
    Main training function.
    
    Args:
        model_type (str): Model architecture to use (cnn, drone_cnn, resnet18, etc.)
        include_drone (bool): Whether to include drone imagery in training
    """
    try:
        # Load configuration
        config = load_config()
        
        # Update config with parameters
        config['model']['name'] = model_type
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Create data loaders
        data_dir = Path(__file__).parent.parent.parent / "data" / "processed"
        dataloaders = create_dataloaders(
            data_dir=str(data_dir),
            batch_size=config['training']['batch_size'],
            num_workers=4,
            config=config,
            include_drone=include_drone
        )
        
        # Create model
        input_channels = 3 if include_drone else 1  # Use RGB for drone images
        model = get_model(
            model_name=model_type,
            num_classes=config['model']['num_classes'],
            pretrained=True,
            input_channels=input_channels
        )
        model = model.to(device)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            device=device,
            config=config['training'],
            checkpoint_dir=str(Path(__file__).parent.parent.parent / "checkpoints")
        )
        
        # Train the model
        logger.info("Starting training...")
        results = trainer.train()
        
        # Save training metrics
        metrics_path = Path(__file__).parent.parent.parent / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Training completed! Metrics saved to {metrics_path}")
        return results
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

def main():
    """Parse arguments and train the model."""
    parser = argparse.ArgumentParser(description="Train the crack detection model")
    parser.add_argument("--model_type", type=str, default="drone_cnn", 
                      help="Model architecture to use (cnn, drone_cnn, resnet18, etc.)")
    parser.add_argument("--include_drone", type=lambda x: x.lower() == 'true', default=True,
                      help="Whether to include drone imagery in training (true/false)")
    
    args = parser.parse_args()
    
    logger.info(f"Training with model_type={args.model_type}, include_drone={args.include_drone}")
    train_model(model_type=args.model_type, include_drone=args.include_drone)

if __name__ == "__main__":
    main() 