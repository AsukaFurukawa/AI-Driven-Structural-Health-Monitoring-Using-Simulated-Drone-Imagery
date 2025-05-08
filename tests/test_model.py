"""
Unit tests for model module.
"""
import os
import sys
import unittest
import torch
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models.crack_detection_model import get_model, CrackDetectionCNN, EfficientNetCrackDetector

class TestModel(unittest.TestCase):
    """Test case for model functions."""
    
    def test_crack_detection_cnn(self):
        """Test CrackDetectionCNN model."""
        # Create model instance
        model = CrackDetectionCNN(num_classes=2, pretrained=False)
        
        # Check model type
        self.assertIsInstance(model, torch.nn.Module)
        
        # Create a random input tensor
        batch_size = 2
        input_tensor = torch.randn(batch_size, 1, 224, 224)
        
        # Forward pass
        output = model(input_tensor)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, 2))
    
    def test_efficientnet_model(self):
        """Test EfficientNetCrackDetector model."""
        # Create model instance
        model = EfficientNetCrackDetector(num_classes=2, pretrained=False)
        
        # Check model type
        self.assertIsInstance(model, torch.nn.Module)
        
        # Create a random input tensor
        batch_size = 2
        input_tensor = torch.randn(batch_size, 1, 224, 224)
        
        # Forward pass
        output = model(input_tensor)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, 2))
    
    def test_get_model(self):
        """Test get_model function."""
        # Get CNN model
        cnn_model = get_model(model_name="cnn", num_classes=2, pretrained=False)
        self.assertIsInstance(cnn_model, CrackDetectionCNN)
        
        # Get EfficientNet model
        efficientnet_model = get_model(model_name="efficientnet", num_classes=2, pretrained=False)
        self.assertIsInstance(efficientnet_model, EfficientNetCrackDetector)
        
        # Test with invalid model name
        with self.assertRaises(ValueError):
            get_model(model_name="invalid_model", num_classes=2, pretrained=False)
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        checkpoint_path = os.path.join(temp_dir, "model_checkpoint.pt")
        
        try:
            # Create model
            model = get_model(model_name="cnn", num_classes=2, pretrained=False)
            
            # Create a random input and target
            input_tensor = torch.randn(2, 1, 224, 224)
            
            # Forward pass
            output_before = model(input_tensor)
            
            # Save checkpoint
            checkpoint = {
                'epoch': 0,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': None,
                'val_loss': 0.0,
                'val_acc': 0.0,
                'config': {}
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Create a new model
            new_model = get_model(model_name="cnn", num_classes=2, pretrained=False)
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path)
            new_model.load_state_dict(checkpoint['model_state_dict'])
            
            # Forward pass with new model
            output_after = new_model(input_tensor)
            
            # Check that outputs are the same
            self.assertTrue(torch.allclose(output_before, output_after))
            
        finally:
            # Clean up
            import shutil
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    unittest.main() 