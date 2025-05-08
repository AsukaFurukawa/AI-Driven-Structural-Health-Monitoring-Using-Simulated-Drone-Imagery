"""
Training module for defect detection models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
import sys
import logging
from pathlib import Path
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.ai.models.defect_detector import create_model, load_model
from src.data.dataset import StructuralDefectDataset

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class DefectDetectorTrainer:
    """Trainer class for defect detection models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = create_model(config['model'])
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        
        # Create output directory
        self.output_dir = Path(config['training']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        
        # Load checkpoint if provided
        if config['training'].get('checkpoint_path'):
            self._load_checkpoint(config['training']['checkpoint_path'])
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create the optimizer based on configuration.
        
        Returns:
            Initialized optimizer
        """
        optim_config = self.config['training']['optimizer']
        optim_name = optim_config.get('type', 'adam').lower()
        lr = optim_config.get('learning_rate', 0.001)
        
        if optim_name == 'adam':
            return optim.Adam(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=optim_config.get('weight_decay', 0.0001)
            )
        elif optim_name == 'sgd':
            return optim.SGD(
                self.model.parameters(), 
                lr=lr, 
                momentum=optim_config.get('momentum', 0.9),
                weight_decay=optim_config.get('weight_decay', 0.0001)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optim_name}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """
        Load a training checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint['best_val_loss']
            self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
    
    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """
        Save a training checkpoint.
        
        Args:
            epoch: Current epoch number
            val_loss: Validation loss
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        
        # Save regular checkpoint
        torch.save(checkpoint, self.output_dir / f"checkpoint_epoch_{epoch}.pth")
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.output_dir / "best_model.pth")
            self.logger.info(f"Saved new best model with validation loss: {val_loss:.4f}")
    
    def train_one_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc="Training")
        
        for batch in pbar:
            # Get data
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100 * correct / total
            })
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / len(train_loader.dataset)
        accuracy = 100 * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Get data
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Update metrics
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate validation metrics
        avg_loss = val_loss / len(val_loader.dataset)
        accuracy = 100 * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            
        Returns:
            Dictionary with training history
        """
        # Training config
        num_epochs = self.config['training']['num_epochs']
        
        # Initialize training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Training loop
        for epoch in range(self.start_epoch, num_epochs):
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train for one epoch
            train_metrics = self.train_one_epoch(train_loader)
            self.logger.info(f"Training loss: {train_metrics['loss']:.4f}, accuracy: {train_metrics['accuracy']:.2f}%")
            
            # Validate
            val_metrics = self.validate(val_loader)
            self.logger.info(f"Validation loss: {val_metrics['loss']:.4f}, accuracy: {val_metrics['accuracy']:.2f}%")
            
            # Update learning rate scheduler
            self.scheduler.step(val_metrics['loss'])
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            self._save_checkpoint(epoch, val_metrics['loss'], is_best)
        
        self.logger.info("Training completed!")
        return history

def create_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger = logging.getLogger(__name__)
    
    # Data paths
    data_dir = Path(config['data']['base_dir'])
    train_csv = data_dir / 'combined_dataset' / 'train_annotations.csv'
    val_csv = data_dir / 'combined_dataset' / 'val_annotations.csv'
    test_csv = data_dir / 'combined_dataset' / 'test_annotations.csv'
    
    # Create datasets
    logger.info("Creating datasets...")
    
    train_dataset = StructuralDefectDataset(
        annotations_file=str(train_csv),
        root_dir=str(data_dir),
        transform=True,
        input_size=config['model']['input_size']
    )
    
    val_dataset = StructuralDefectDataset(
        annotations_file=str(val_csv),
        root_dir=str(data_dir),
        transform=False,
        input_size=config['model']['input_size']
    )
    
    test_dataset = StructuralDefectDataset(
        annotations_file=str(test_csv),
        root_dir=str(data_dir),
        transform=False,
        input_size=config['model']['input_size']
    )
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4)
    )
    
    logger.info(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

def main():
    """Main training function."""
    logger = setup_logging()
    
    # Load configuration
    config_path = Path(__file__).parent.parent.parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    # Initialize trainer
    trainer = DefectDetectorTrainer(config)
    
    # Train model
    history = trainer.train(train_loader, val_loader)
    
    # Save training history
    history_df = pd.DataFrame({
        'train_loss': history['train_loss'],
        'train_accuracy': history['train_acc'],
        'val_loss': history['val_loss'],
        'val_accuracy': history['val_acc']
    })
    
    history_path = Path(config['training']['output_dir']) / 'training_history.csv'
    history_df.to_csv(history_path, index=False)
    logger.info(f"Saved training history to {history_path}")
    
    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.validate(test_loader)
    logger.info(f"Test loss: {test_metrics['loss']:.4f}, accuracy: {test_metrics['accuracy']:.2f}%")

if __name__ == "__main__":
    main() 