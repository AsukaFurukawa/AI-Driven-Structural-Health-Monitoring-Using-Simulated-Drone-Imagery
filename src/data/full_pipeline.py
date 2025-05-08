#!/usr/bin/env python3
"""
Full data processing pipeline for structural health monitoring.

This script combines all data preparation, cleaning, and preprocessing steps
into a single runnable pipeline for preparing the data for model training.
"""
import os
import sys
import argparse
import logging
import yaml
import time
from pathlib import Path

# Add parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.data_pipeline import DataPipeline

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('data_pipeline.log')
        ]
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the full data processing pipeline")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--clean-only", 
        action="store_true",
        help="Only clean and process existing data, don't generate synthetic data"
    )
    
    parser.add_argument(
        "--skip-augmentation", 
        action="store_true",
        help="Skip data augmentation step"
    )
    
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default=None,
        help="Override data directory from config"
    )
    
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level"
    )
    
    return parser.parse_args()

def update_config(config_path: str, args) -> dict:
    """Update configuration with command line arguments."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with command line arguments if provided
    if args.skip_augmentation:
        if 'image_processing' not in config:
            config['image_processing'] = {}
        if 'preprocessing' not in config['image_processing']:
            config['image_processing']['preprocessing'] = {}
        config['image_processing']['preprocessing']['augmentation'] = False
    
    if args.data_dir:
        if 'storage' not in config:
            config['storage'] = {}
        if 'local' not in config['storage']:
            config['storage']['local'] = {}
        config['storage']['local']['upload_dir'] = args.data_dir
    
    return config

def main():
    """Run the full data processing pipeline."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    start_time = time.time()
    logger.info("Starting full data processing pipeline")
    
    try:
        # Load and update configuration
        config_path = args.config
        updated_config = update_config(config_path, args)
        
        # Save updated config
        temp_config_path = "config/temp_config.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(updated_config, f)
        
        # Initialize the data pipeline
        pipeline = DataPipeline(config_path=temp_config_path)
        
        # Run the full pipeline
        success = pipeline.run_full_pipeline()
        
        if success:
            logger.info("Data processing pipeline completed successfully!")
            
            # Print summary statistics
            try:
                output_dir = pipeline.output_dir
                info_file = output_dir / 'dataset_info.yaml'
                if info_file.exists():
                    with open(info_file, 'r') as f:
                        info = yaml.safe_load(f)
                    
                    logger.info("\nDataset Summary:")
                    logger.info("-" * 40)
                    if 'class_mapping' in info:
                        logger.info(f"Classes: {list(info['class_mapping'].keys())}")
                    if 'preprocessing' in info:
                        logger.info(f"Input size: {info['preprocessing']['input_size']}")
                        logger.info(f"Normalization: {info['preprocessing']['normalize']}")
                        logger.info(f"Augmentation: {info['preprocessing']['augmentation']}")
            except Exception as e:
                logger.warning(f"Could not print summary statistics: {str(e)}")
        else:
            logger.error("Data processing pipeline failed!")
        
        # Clean up temporary config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        
    except Exception as e:
        logger.error(f"Error in data processing pipeline: {str(e)}", exc_info=True)
        return 1
        
    elapsed_time = time.time() - start_time
    logger.info(f"Pipeline execution time: {elapsed_time:.2f} seconds")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 