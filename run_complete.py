#!/usr/bin/env python3
"""
Complete Pipeline Script for Structural Health Monitoring

This script runs the complete pipeline:
1. Generates synthetic data including simulated drone imagery
2. Processes the data for model input
3. Trains the model
4. Starts the web UI for interactive use

Usage:
    python run_complete.py [--no-train] [--no-drone] [--no-ui]
"""
import os
import sys
import argparse
import logging
import subprocess
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(command, cwd=None):
    """Run a shell command and log the output."""
    logger.info(f"Running command: {command}")
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            cwd=cwd
        )
        
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if line:
                logger.info(line)
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code != 0:
            logger.error(f"Command failed with return code {return_code}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error running command: {str(e)}")
        return False

def generate_synthetic_data(num_images=1000, include_drone=True):
    """Generate synthetic data including drone imagery."""
    logger.info("Generating synthetic data...")
    
    drone_param = "" if include_drone else "--no-drone"
    command = f"python src/data/generate_synthetic_data.py --num_images {num_images} {drone_param}"
    
    return run_command(command)

def process_data(include_drone=True):
    """Process the raw data for model input."""
    logger.info("Processing data...")
    
    drone_param = "" if include_drone else "--no-drone"
    command = f"python src/data/process_data.py {drone_param}"
    
    return run_command(command)

def train_model(include_drone=True):
    """Train the model on the processed data."""
    logger.info("Training model...")
    
    # Modify the train script to include a drone parameter
    model_type = "drone_cnn" if include_drone else "cnn"
    command = f"python src/models/train.py --model_type {model_type} --include_drone {str(include_drone).lower()}"
    
    return run_command(command)

def start_api():
    """Start the API server."""
    logger.info("Starting the API server...")
    
    # Use a non-blocking subprocess to start the API
    api_process = subprocess.Popen(
        ["python", "run_api.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    
    # Give the API a moment to start
    time.sleep(2)
    
    # Check if it's running
    if api_process.poll() is None:
        logger.info("API server started successfully")
        logger.info("Web interface available at http://localhost:8000")
        return api_process
    else:
        logger.error("Failed to start API server")
        return None

def main():
    """Run the complete pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the complete structural health monitoring pipeline")
    parser.add_argument("--no-train", action="store_true", help="Skip model training")
    parser.add_argument("--no-drone", action="store_true", help="Exclude drone imagery")
    parser.add_argument("--no-ui", action="store_true", help="Skip starting the web UI")
    parser.add_argument("--num-images", type=int, default=1000, help="Number of synthetic images to generate")
    
    args = parser.parse_args()
    
    include_drone = not args.no_drone
    
    # 1. Generate synthetic data
    if not generate_synthetic_data(args.num_images, include_drone):
        logger.error("Data generation failed")
        return 1
    
    # 2. Process data
    if not process_data(include_drone):
        logger.error("Data processing failed")
        return 1
    
    # 3. Train model if requested
    if not args.no_train:
        if not train_model(include_drone):
            logger.error("Model training failed")
            return 1
    
    # 4. Start the web UI if requested
    if not args.no_ui:
        api_process = start_api()
        if api_process is None:
            return 1
        
        try:
            # Keep the script running to maintain the API
            logger.info("Press Ctrl+C to stop the server")
            api_process.wait()
        except KeyboardInterrupt:
            logger.info("Stopping API server...")
            api_process.terminate()
    
    logger.info("Pipeline completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 