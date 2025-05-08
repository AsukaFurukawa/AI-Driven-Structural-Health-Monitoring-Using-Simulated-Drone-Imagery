#!/usr/bin/env python3
"""
Training script for structural health monitoring models.

This script trains a model for crack detection using the processed data.
"""
import os
import sys
import argparse
import logging
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from pathlib import Path

from src.data.dataset import create_dataloaders
from src.models.crack_detection_model import get_model
from src.models.trainer import Trainer

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Create logger
    logger = logging.getLogger("train_model")
    logger.setLevel(numeric_level)
    
    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(numeric_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a model for structural health monitoring.")
    
    parser.add_argument("--data_dir", type=str, default="data/processed", 
                        help="Directory containing processed data")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--model", type=str, default="cnn",
                        help="Model architecture to use (cnn, mobilenet, resnet18, efficientnet)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Initial learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of worker processes for data loading")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--pretrained", action="store_true",
                        help="Use pretrained model weights")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate model on test set after training")
    
    return parser.parse_args()

def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting model training")
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config = {}
        logger.warning(f"Configuration file {args.config} not found, using defaults")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.info("Creating data loaders")
    dataloaders = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Get class names and number of classes
    data_dir = Path(args.data_dir)
    dataset_info_path = data_dir / "dataset_info.yaml"
    
    if os.path.exists(dataset_info_path):
        with open(dataset_info_path, 'r') as f:
            dataset_info = yaml.safe_load(f)
        num_classes = len(dataset_info.get('class_names', ['negative', 'positive']))
    else:
        num_classes = 2  # Default to binary classification
        
    logger.info(f"Training with {num_classes} classes")
    
    # Create model
    logger.info(f"Creating model: {args.model}")
    model = get_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=args.pretrained
    )
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Define learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        verbose=True
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config={
            "num_epochs": args.epochs,
            "early_stopping_patience": 10
        },
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Train model
    logger.info("Starting training")
    training_history = trainer.train()
    
    # Evaluate model
    if args.evaluate:
        logger.info("Evaluating model on test set")
        test_metrics = trainer.evaluate(dataloaders['test'])
        
        # Log test metrics
        logger.info(f"Test accuracy: {test_metrics['accuracy']:.2f}%")
        logger.info(f"Test F1 score: {test_metrics['f1']:.4f}")
    
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {training_history['best_val_acc']:.2f}% at epoch {training_history['best_epoch'] + 1}")
    logger.info(f"Model checkpoints saved to {args.checkpoint_dir}")

if __name__ == "__main__":
    main() 