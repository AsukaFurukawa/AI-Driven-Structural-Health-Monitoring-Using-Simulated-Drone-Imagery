"""
Script to run the entire system.
"""
import os
import sys
import logging
import yaml
import time
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent / "config" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def run_process(script_path, name):
    """
    Run a Python script as a subprocess.
    
    Args:
        script_path (str): Path to the script
        name (str): Name of the process
    """
    try:
        logger.info(f"Starting {name}")
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Log output
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logger.info(f"{name}: {output.strip()}")
        
        # Check for errors
        if process.returncode != 0:
            error = process.stderr.read()
            logger.error(f"{name} failed with error: {error}")
            raise RuntimeError(f"{name} failed with return code {process.returncode}")
        
        logger.info(f"{name} completed successfully")
        
    except Exception as e:
        logger.error(f"Error running {name}: {str(e)}")
        raise

def main():
    """Main function to run the entire system."""
    try:
        # Load configuration
        config = load_config()
        
        # Get script paths
        base_dir = Path(__file__).parent
        scripts = {
            'Drone Simulation': base_dir / "src" / "simulation" / "run.py",
            'Data Processing': base_dir / "src" / "data" / "process_data.py",
            'Model Training': base_dir / "src" / "models" / "train.py",
            'API Server': base_dir / "src" / "api" / "run.py",
            'Web Interface': base_dir / "src" / "web" / "run.py"
        }
        
        # Run processes in parallel
        with ThreadPoolExecutor(max_workers=len(scripts)) as executor:
            futures = []
            for name, script_path in scripts.items():
                if script_path.exists():
                    futures.append(executor.submit(run_process, str(script_path), name))
                else:
                    logger.warning(f"Script not found: {script_path}")
        
        # Wait for all processes to complete
        for future in futures:
            future.result()
        
        logger.info("All processes completed successfully")
        
    except Exception as e:
        logger.error(f"Error running system: {str(e)}")
        raise

if __name__ == "__main__":
    main() 