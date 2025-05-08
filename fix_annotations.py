#!/usr/bin/env python3
"""
Wrapper script to fix missing annotations for all datasets.

This script runs the utility to identify and fix missing annotations in datasets.
"""
import os
import sys
import logging
import argparse
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the annotation fixer."""
    parser = argparse.ArgumentParser(description="Fix missing annotations in datasets")
    
    parser.add_argument("--data_dir", type=str, default="uploads/datasets",
                       help="Base directory containing the datasets")
    
    parser.add_argument("--dataset", type=str, default=None,
                       help="Process only a specific dataset (e.g., 'Crack_Detection')")
    
    args = parser.parse_args()
    
    # Build command to run the utility script
    script_path = os.path.join("src", "data", "fix_missing_annotations.py")
    
    command = [sys.executable, script_path]
    
    if args.data_dir:
        command.extend(["--data_dir", args.data_dir])
    
    if args.dataset:
        command.extend(["--dataset", args.dataset])
    
    # Run the command
    logger.info(f"Running annotation fixer: {' '.join(command)}")
    
    import subprocess
    result = subprocess.run(command, check=False)
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main()) 