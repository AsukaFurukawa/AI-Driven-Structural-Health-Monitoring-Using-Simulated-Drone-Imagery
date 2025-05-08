"""
Unit tests for data processing module.
"""
import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data.process_data import process_image, create_directory_structure
from src.data.dataset import preprocess_image, CrackDataset

class TestDataProcessing(unittest.TestCase):
    """Test case for data processing functions."""
    
    def setUp(self):
        """Setup test environment."""
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.temp_dir, "input")
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.input_dir)
        os.makedirs(self.output_dir)
        
        # Create test images
        for i in range(5):
            img = Image.new('L', (100, 100), color=128)
            img_path = os.path.join(self.input_dir, f"test_image_{i}.png")
            img.save(img_path)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_process_image(self):
        """Test image processing function."""
        # Get a test image
        input_path = os.path.join(self.input_dir, "test_image_0.png")
        output_path = os.path.join(self.output_dir, "processed_image.png")
        
        # Process the image
        process_image(input_path, output_path, target_size=(224, 224))
        
        # Check that the output file exists
        self.assertTrue(os.path.exists(output_path))
        
        # Check that the processed image has the correct size
        with Image.open(output_path) as img:
            self.assertEqual(img.size, (224, 224))
    
    def test_preprocess_image(self):
        """Test image preprocessing for the model."""
        # Get a test image
        input_path = os.path.join(self.input_dir, "test_image_0.png")
        
        # Preprocess the image
        tensor = preprocess_image(input_path)
        
        # Check that the output is a tensor
        self.assertIsNotNone(tensor)
        
        # Check tensor dimensions
        self.assertEqual(tensor.shape[0], 1)  # Batch dimension
        self.assertEqual(tensor.shape[1], 1)  # Channels (grayscale)
        self.assertEqual(tensor.shape[2], 224)  # Height
        self.assertEqual(tensor.shape[3], 224)  # Width
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        # Create image paths and labels
        image_paths = [os.path.join(self.input_dir, f"test_image_{i}.png") for i in range(5)]
        labels = [0, 1, 0, 1, 0]  # Alternate labels
        
        # Create dataset
        dataset = CrackDataset(image_paths, labels)
        
        # Check dataset length
        self.assertEqual(len(dataset), 5)
        
        # Check item retrieval
        image, label = dataset[0]
        self.assertEqual(label, 0)
        self.assertEqual(image.shape[0], 1)  # Channels
        self.assertEqual(image.shape[1], 224)  # Height
        self.assertEqual(image.shape[2], 224)  # Width
    
    def test_create_directory_structure(self):
        """Test directory structure creation."""
        # Mock the directory creation function
        original_mkdir = os.mkdir
        created_dirs = []
        
        def mock_mkdir(path, *args, **kwargs):
            created_dirs.append(path)
            return original_mkdir(path, *args, **kwargs)
        
        try:
            # Replace mkdir with mock function
            os.mkdir = mock_mkdir
            
            # Call the function
            base_dir = Path(self.temp_dir) / "test_create_dirs"
            base_dir.mkdir()
            os.chdir(base_dir)
            create_directory_structure()
            
            # Check that the expected directories are created
            expected_dirs = ['data', 'data/raw', 'data/processed', 
                            'data/processed/positive', 'data/processed/negative']
            
            # Check if all expected directories were created
            for dir_name in expected_dirs:
                dir_path = base_dir / dir_name
                self.assertTrue(dir_path.exists() or str(dir_path) in created_dirs)
        
        finally:
            # Restore original mkdir function
            os.mkdir = original_mkdir

if __name__ == '__main__':
    unittest.main() 