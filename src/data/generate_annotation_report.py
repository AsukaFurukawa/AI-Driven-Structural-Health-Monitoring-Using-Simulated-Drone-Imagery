#!/usr/bin/env python3
"""
Generate a detailed report of all images and their annotation status.

This script creates a CSV report showing which images have annotations and which don't.
"""
import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml

# Add parent directory to the Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('annotation_report.log')
    ]
)
logger = logging.getLogger(__name__)

def generate_dataset_report(dataset_dir, dataset_name):
    """
    Generate a detailed report for a dataset.
    
    Args:
        dataset_dir (Path): Path to the dataset directory
        dataset_name (str): Name of the dataset
        
    Returns:
        DataFrame: Report dataframe
    """
    logger.info(f"Generating report for {dataset_name} dataset at {dataset_dir}")
    
    try:
        # Identify annotation files
        initial_annotations = dataset_dir / 'initial_annotations.csv'
        cleaned_annotations = dataset_dir / 'cleaned_annotations.csv'
        
        # Load annotations if they exist
        annotations_df = None
        if cleaned_annotations.exists():
            annotations_df = pd.read_csv(cleaned_annotations)
            logger.info(f"Loaded {len(annotations_df)} cleaned annotations")
        elif initial_annotations.exists():
            annotations_df = pd.read_csv(initial_annotations)
            logger.info(f"Loaded {len(annotations_df)} initial annotations")
        
        # Get all images in the dataset
        all_images = []
        
        # Check common image directories based on dataset structure
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(root, file)
                    rel_path = os.path.relpath(img_path, dataset_dir)
                    all_images.append(rel_path)
        
        logger.info(f"Found {len(all_images)} total images")
        
        # Create report dataframe
        report_data = []
        
        for img_path in all_images:
            img_info = {
                'dataset': dataset_name,
                'image_path': img_path,
                'has_annotation': False,
                'defect_class': None,
                'annotation_source': None
            }
            
            # Check if image has annotation
            if annotations_df is not None:
                # Normalize paths for comparison
                norm_img_path = img_path.replace('\\', '/').lower()
                for idx, row in annotations_df.iterrows():
                    norm_anno_path = row['image_path'].replace('\\', '/').lower()
                    if norm_img_path == norm_anno_path:
                        img_info['has_annotation'] = True
                        img_info['defect_class'] = row.get('defect_class')
                        img_info['annotation_source'] = row.get('dataset_source', dataset_name)
                        break
            
            # Determine defect class from path if not annotated
            if not img_info['has_annotation']:
                path = Path(img_path)
                if 'Positive' in img_path or 'crack_' in path.stem.lower() or 'crack' in path.stem.lower():
                    img_info['defect_class'] = 'Crack (inferred)'
                elif 'Negative' in img_path or 'no_crack' in path.stem.lower() or 'nocrack' in path.stem.lower():
                    img_info['defect_class'] = 'No_Defect (inferred)'
            
            report_data.append(img_info)
        
        # Create DataFrame
        report_df = pd.DataFrame(report_data)
        
        return report_df
    
    except Exception as e:
        logger.error(f"Error generating report for {dataset_name} dataset: {str(e)}")
        return pd.DataFrame()

def generate_full_report(base_dir, output_file):
    """
    Generate a report for all datasets.
    
    Args:
        base_dir (Path): Base directory containing all datasets
        output_file (Path): Output file for the report
        
    Returns:
        bool: Success flag
    """
    try:
        all_reports = []
        
        # Process each dataset directory
        for dataset_dir in base_dir.glob('*'):
            if dataset_dir.is_dir() and dataset_dir.name not in ['.git', '__pycache__']:
                dataset_report = generate_dataset_report(dataset_dir, dataset_dir.name)
                if not dataset_report.empty:
                    all_reports.append(dataset_report)
        
        # Combine all reports
        if all_reports:
            combined_report = pd.concat(all_reports, ignore_index=True)
            
            # Save the report
            combined_report.to_csv(output_file, index=False)
            logger.info(f"Report saved to {output_file}")
            
            # Generate summary
            total_images = len(combined_report)
            annotated = combined_report['has_annotation'].sum()
            unannotated = total_images - annotated
            
            logger.info(f"Total images: {total_images}")
            logger.info(f"Annotated images: {annotated} ({(annotated/total_images)*100:.1f}%)")
            logger.info(f"Unannotated images: {unannotated} ({(unannotated/total_images)*100:.1f}%)")
            
            # Dataset breakdown
            for dataset in combined_report['dataset'].unique():
                dataset_df = combined_report[combined_report['dataset'] == dataset]
                dataset_total = len(dataset_df)
                dataset_annotated = dataset_df['has_annotation'].sum()
                logger.info(f"Dataset '{dataset}': {dataset_annotated}/{dataset_total} images annotated ({(dataset_annotated/dataset_total)*100:.1f}%)")
            
            return True
        else:
            logger.warning("No reports generated for any dataset")
            return False
    
    except Exception as e:
        logger.error(f"Error generating full report: {str(e)}")
        return False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate annotation report")
    
    parser.add_argument("--data_dir", type=str, default="uploads/datasets",
                       help="Base directory containing the datasets")
    
    parser.add_argument("--output", type=str, default="annotation_report.csv",
                       help="Output file for the report")
    
    parser.add_argument("--dataset", type=str, default=None,
                       help="Process only a specific dataset (e.g., 'Crack_Detection')")
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Resolve data directory path
    base_dir = Path(__file__).parent.parent.parent / args.data_dir
    output_file = Path(__file__).parent.parent.parent / args.output
    
    if not base_dir.exists():
        logger.error(f"Data directory not found: {base_dir}")
        return 1
    
    logger.info(f"Processing datasets in {base_dir}")
    
    if args.dataset:
        # Process only the specified dataset
        dataset_dir = base_dir / args.dataset
        if not dataset_dir.exists():
            logger.error(f"Dataset directory not found: {dataset_dir}")
            return 1
        
        dataset_report = generate_dataset_report(dataset_dir, args.dataset)
        if not dataset_report.empty:
            dataset_report.to_csv(output_file, index=False)
            logger.info(f"Report for dataset '{args.dataset}' saved to {output_file}")
            return 0
        else:
            return 1
    else:
        # Process all datasets
        success = generate_full_report(base_dir, output_file)
        return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 