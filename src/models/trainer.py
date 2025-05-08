"""
Model trainer for structural health monitoring.

This module provides utilities for training and evaluating models.
"""
import os
import sys
import time
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# For logging and model checkpoints
from datetime import datetime

class Trainer:
    """
    Model trainer for structural health monitoring.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: Any = None,
        device: str = None,
        config: Dict = None,
        checkpoint_dir: str = "checkpoints"
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            criterion: Loss function (default: CrossEntropyLoss)
            optimizer: Optimizer (default: Adam with lr=1e-4)
            scheduler: Learning rate scheduler (default: None)
            device: Device to use for training (default: cuda if available, else cpu)
            config: Configuration dictionary
            checkpoint_dir: Directory to save model checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Set device
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Set loss function
        self.criterion = criterion if criterion else nn.CrossEntropyLoss()
        
        # Set optimizer
        self.optimizer = optimizer if optimizer else Adam(model.parameters(), lr=1e-4)
        
        # Set scheduler
        self.scheduler = scheduler
        
        # Configuration
        self.config = config if config else {}
        self.num_epochs = self.config.get("num_epochs", 50)
        self.early_stopping_patience = self.config.get("early_stopping_patience", 10)
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Logging
        self.logger = logging.getLogger("model_trainer")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train the model for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average loss, accuracy) for the epoch
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero the gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = 100.0 * correct / total
            progress_bar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "acc": f"{accuracy:.2f}%"
            })
        
        # Calculate final metrics
        epoch_loss = total_loss / len(self.train_loader)
        epoch_accuracy = 100.0 * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def validate(self, epoch: int) -> Tuple[float, float]:
        """
        Validate the model on the validation set.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average loss, accuracy) for the validation set
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Val]")
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Accumulate metrics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                avg_loss = total_loss / (batch_idx + 1)
                accuracy = 100.0 * correct / total
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "acc": f"{accuracy:.2f}%"
                })
        
        # Calculate final metrics
        epoch_loss = total_loss / len(self.val_loader)
        epoch_accuracy = 100.0 * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def train(self) -> Dict:
        """
        Train the model for the specified number of epochs.
        
        Returns:
            Dictionary with training history and best model info
        """
        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        
        self.logger.info(f"Starting training for {self.num_epochs} epochs on {self.device}")
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            # Train one epoch
            train_loss, train_acc = self.train_one_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Update learning rate if scheduler is available
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Print metrics
            self.logger.info(f"Epoch {epoch+1}/{self.num_epochs} - "
                            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Check if current model is the best
            is_best = False
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                is_best = True
            else:
                patience_counter += 1
            
            # Save checkpoint
            checkpoint_path = self.checkpoint_dir / f"model_epoch_{epoch+1}.pt"
            self.save_checkpoint(checkpoint_path, epoch, val_loss, val_acc, is_best)
            
            # Early stopping
            if patience_counter >= self.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        self.logger.info(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch+1}")
        
        # Load the best model
        best_model_path = self.checkpoint_dir / "model_best.pt"
        if best_model_path.exists():
            self.load_checkpoint(best_model_path)
        
        # Plot training history
        self.plot_training_history()
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "training_time": training_time
        }
    
    def save_checkpoint(
        self, 
        path: Union[str, Path], 
        epoch: int, 
        val_loss: float, 
        val_acc: float, 
        is_best: bool = False
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save the checkpoint
            epoch: Current epoch
            val_loss: Validation loss
            val_acc: Validation accuracy
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        
        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / "model_best.pt"
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: Union[str, Path]) -> Dict:
        """
        Load model from checkpoint.
        
        Args:
            path: Path to the checkpoint
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore metrics history
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        if 'train_accuracies' in checkpoint:
            self.train_accuracies = checkpoint['train_accuracies']
        if 'val_accuracies' in checkpoint:
            self.val_accuracies = checkpoint['val_accuracies']
        
        return checkpoint
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate the model on a test set.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        all_targets = []
        all_predictions = []
        test_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc="Evaluating"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Get predictions
                _, predictions = outputs.max(1)
                
                # Accumulate metrics
                test_loss += loss.item()
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions) * 100.0
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted'
        )
        conf_matrix = confusion_matrix(all_targets, all_predictions)
        
        # Log results
        self.logger.info(f"Test Loss: {test_loss/len(test_loader):.4f}")
        self.logger.info(f"Test Accuracy: {accuracy:.2f}%")
        self.logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return {
            "loss": test_loss / len(test_loader),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": conf_matrix
        }
    
    def plot_training_history(self) -> None:
        """Plot training and validation metrics."""
        epochs = range(1, len(self.train_losses) + 1)
        
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        plt.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(self.checkpoint_dir / "training_history.png")
        plt.close() 