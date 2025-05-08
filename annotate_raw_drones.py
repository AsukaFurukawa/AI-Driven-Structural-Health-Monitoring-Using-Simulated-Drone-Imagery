#!/usr/bin/env python3
"""
Wrapper script to annotate raw drone images.

This script runs the utility to annotate raw drone images in the data/raw directory.
"""
import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the raw drone annotation utility."""
    
    # Build command to run the utility script
    script_path = os.path.join("src", "data", "annotate_raw_drone_images.py")
    
    command = [sys.executable, script_path]
    
    # Run the command
    logger.info(f"Running raw drone annotation utility: {' '.join(command)}")
    
    import subprocess
    result = subprocess.run(command, check=False)
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main()) 