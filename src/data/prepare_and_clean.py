import os
import logging
from pathlib import Path
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.prepare_dataset import prepare_public_crack_dataset
from src.data.data_cleaning import DataCleaner

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()
    
    # Set up paths
    base_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    data_dir = base_dir / 'uploads' / 'datasets'
    
    logger.info(f"Working directory: {base_dir}")
    logger.info(f"Data directory: {data_dir}")
    
    # Step 1: Prepare the dataset
    logger.info("Preparing crack detection dataset...")
    annotations_df = prepare_public_crack_dataset(str(data_dir))
    
    if annotations_df.empty:
        logger.error("Failed to prepare dataset!")
        return
    
    # Save initial annotations
    annotations_file = data_dir / 'Crack_Detection' / 'initial_annotations.csv'
    annotations_df.to_csv(annotations_file, index=False)
    logger.info(f"Saved initial annotations to {annotations_file}")
    
    # Step 2: Clean the dataset
    logger.info("Cleaning the dataset...")
    cleaner = DataCleaner(str(data_dir / 'Crack_Detection'))
    
    # Clean the dataset with standard image size
    cleaned_df = cleaner.clean_dataset(
        str(annotations_file),
        target_size=(224, 224)  # Standard size for most modern CNNs
    )
    
    # Save cleaned annotations
    cleaned_file = data_dir / 'Crack_Detection' / 'cleaned_annotations.csv'
    cleaned_df.to_csv(cleaned_file, index=False)
    
    # Print statistics
    logger.info("\nDataset Statistics:")
    logger.info(f"Original samples: {len(annotations_df)}")
    logger.info(f"Cleaned samples: {len(cleaned_df)}")
    logger.info(f"Removed samples: {len(annotations_df) - len(cleaned_df)}")
    logger.info("\nDefect distribution:")
    logger.info(cleaned_df['defect_class'].value_counts())

if __name__ == "__main__":
    main() 