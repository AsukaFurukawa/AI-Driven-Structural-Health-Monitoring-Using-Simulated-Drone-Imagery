"""
Tests for the data processing pipeline.
"""
import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import yaml
import logging

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_pipeline import DataPipeline

class TestDataPipeline(unittest.TestCase):
    """Test cases for the data processing pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        # Disable logging during tests
        logging.disable(logging.CRITICAL)
        
        # Create temporary directory for test data
        cls.temp_dir = tempfile.mkdtemp()
        cls.data_dir = Path(cls.temp_dir) / 'data'
        cls.output_dir = Path(cls.temp_dir) / 'output'
        
        # Create necessary directories
        os.makedirs(cls.data_dir, exist_ok=True)
        os.makedirs(cls.output_dir, exist_ok=True)
        
        # Create temporary config
        cls.config_path = Path(cls.temp_dir) / 'test_config.yaml'
        
        # Basic test configuration
        test_config = {
            'app': {
                'name': 'Test SHM Pipeline',
                'environment': 'test'
            },
            'model': {
                'input_size': [224, 224],
                'num_classes': 2
            },
            'image_processing': {
                'preprocessing': {
                    'normalize': True,
                    'augmentation': False
                },
                'supported_formats': ['.jpg', '.png']
            },
            'storage': {
                'local': {
                    'upload_dir': str(cls.data_dir),
                    'results_dir': str(cls.output_dir)
                }
            }
        }
        
        with open(cls.config_path, 'w') as f:
            yaml.dump(test_config, f)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment after all tests."""
        # Remove temporary directory
        shutil.rmtree(cls.temp_dir)
        
        # Re-enable logging
        logging.disable(logging.NOTSET)
    
    def setUp(self):
        """Set up before each test."""
        # Create test dataset directories
        self.crack_dir = self.data_dir / 'Crack_Detection'
        self.crack_positive = self.crack_dir / 'Positive'
        self.crack_negative = self.crack_dir / 'Negative'
        
        self.uav_dir = self.data_dir / 'UAV_Inspection'
        self.uav_defect = self.uav_dir / 'defect'
        self.uav_normal = self.uav_dir / 'normal'
        
        # Create directories
        for directory in [self.crack_positive, self.crack_negative, self.uav_defect, self.uav_normal]:
            os.makedirs(directory, exist_ok=True)
            
        # Create a few test images
        self._create_test_images()
    
    def tearDown(self):
        """Clean up after each test."""
        # Clean data directories
        for directory in [self.crack_dir, self.uav_dir]:
            if directory.exists():
                shutil.rmtree(directory)
    
    def _create_test_images(self):
        """Create test images for the pipeline."""
        import numpy as np
        from PIL import Image
        
        # Create some simple test images
        for i in range(5):
            # Crack positive images (with simple 'crack')
            img = np.ones((224, 224), dtype=np.uint8) * 200
            # Add a line to simulate crack
            img[100:110, 50:200] = 50
            Image.fromarray(img).save(self.crack_positive / f'crack_{i}.jpg')
            
            # Crack negative images
            img = np.ones((224, 224), dtype=np.uint8) * 200
            Image.fromarray(img).save(self.crack_negative / f'no_crack_{i}.jpg')
            
            # UAV defect images
            img = np.ones((224, 224, 3), dtype=np.uint8) * 200
            # Add a red 'defect'
            img[80:100, 80:100, 0] = 255
            img[80:100, 80:100, 1:] = 0
            Image.fromarray(img).save(self.uav_defect / f'defect_{i}.jpg')
            
            # UAV normal images
            img = np.ones((224, 224, 3), dtype=np.uint8) * 200
            Image.fromarray(img).save(self.uav_normal / f'normal_{i}.jpg')
    
    def test_pipeline_initialization(self):
        """Test if pipeline initializes correctly."""
        pipeline = DataPipeline(config_path=str(self.config_path))
        
        self.assertEqual(pipeline.data_dir, self.data_dir)
        self.assertIsNotNone(pipeline.config)
        self.assertTrue(hasattr(pipeline, 'logger'))
    
    def test_prepare_crack_detection_data(self):
        """Test preparation of crack detection dataset."""
        pipeline = DataPipeline(config_path=str(self.config_path))
        
        # Run the crack detection preparation
        result_df = pipeline.prepare_crack_detection_data()
        
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertGreater(len(result_df), 0)
        self.assertTrue('image_path' in result_df.columns)
        self.assertTrue('defect_class' in result_df.columns)
        
        # Check if annotations file was created
        initial_annotations = self.crack_dir / 'initial_annotations.csv'
        cleaned_annotations = self.crack_dir / 'cleaned_annotations.csv'
        
        self.assertTrue(initial_annotations.exists())
        self.assertTrue(cleaned_annotations.exists())
    
    def test_prepare_uav_inspection_data(self):
        """Test preparation of UAV inspection dataset."""
        pipeline = DataPipeline(config_path=str(self.config_path))
        
        # Run the UAV inspection preparation
        result_df = pipeline.prepare_uav_inspection_data()
        
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertGreater(len(result_df), 0)
        self.assertTrue('image_path' in result_df.columns)
        self.assertTrue('defect_class' in result_df.columns)
        
        # Check if annotations file was created
        initial_annotations = self.uav_dir / 'initial_annotations.csv'
        cleaned_annotations = self.uav_dir / 'cleaned_annotations.csv'
        
        self.assertTrue(initial_annotations.exists())
        self.assertTrue(cleaned_annotations.exists())
    
    def test_combine_datasets(self):
        """Test combining multiple datasets."""
        pipeline = DataPipeline(config_path=str(self.config_path))
        
        # First prepare individual datasets
        pipeline.prepare_crack_detection_data()
        pipeline.prepare_uav_inspection_data()
        
        # Now combine them
        result = pipeline.combine_all_datasets()
        
        self.assertTrue(result)
        
        # Check combined dataset exists
        combined_dir = self.data_dir / 'combined_dataset'
        self.assertTrue(combined_dir.exists())
        
        # Check if split files were created
        for split in ['train', 'val', 'test']:
            split_file = combined_dir / f'{split}_annotations.csv'
            self.assertTrue(split_file.exists())
    
    def test_full_pipeline(self):
        """Test running the full pipeline end-to-end."""
        pipeline = DataPipeline(config_path=str(self.config_path))
        
        # Run full pipeline
        result = pipeline.run_full_pipeline()
        
        self.assertTrue(result)
        
        # Verify output directory has processed datasets
        for split in ['train', 'val', 'test']:
            split_dir = pipeline.output_dir / split
            self.assertTrue(split_dir.exists())
            
            # Should have at least some images
            image_files = list(split_dir.glob('*.jpg'))
            self.assertGreater(len(image_files), 0)
        
        # Verify dataset info was created
        info_file = pipeline.output_dir / 'dataset_info.yaml'
        self.assertTrue(info_file.exists())

if __name__ == '__main__':
    unittest.main() 