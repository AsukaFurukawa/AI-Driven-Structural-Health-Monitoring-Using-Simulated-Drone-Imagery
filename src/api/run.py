"""
Script to run the API server.
"""
import os
import sys
import logging
import yaml
import uvicorn
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def main():
    """Main function to run the API server."""
    try:
        # Load configuration
        config = load_config()
        
        # Get API configuration
        api_config = config.get('api', {})
        host = api_config.get('host', '0.0.0.0')
        port = api_config.get('port', 8000)
        debug = api_config.get('debug', True)
        
        # Create upload directory if it doesn't exist
        upload_dir = Path(__file__).parent.parent.parent / api_config.get('upload_dir', 'uploads/api_uploads')
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Run the server
        logger.info(f"Starting API server at http://{host}:{port}")
        uvicorn.run(
            "app:app",
            host=host,
            port=port,
            reload=debug
        )
        
    except Exception as e:
        logger.error(f"Error starting API server: {str(e)}")
        raise

if __name__ == "__main__":
    main() 