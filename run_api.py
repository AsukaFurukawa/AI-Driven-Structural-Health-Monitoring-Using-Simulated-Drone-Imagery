#!/usr/bin/env python3
"""
Run the API server for structural health monitoring.
"""
import os
import sys
import logging
import argparse
import uvicorn
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api_server")

def main():
    """Main function to run the API server."""
    parser = argparse.ArgumentParser(description="Run the API server for structural health monitoring")
    
    # Server arguments
    parser.add_argument("--host", type=str, default="0.0.0.0", 
                        help="Host to listen on")
    parser.add_argument("--port", type=int, default=8000, 
                        help="Port to listen on")
    parser.add_argument("--debug", action="store_true", 
                        help="Enable debug mode")
    parser.add_argument("--reload", action="store_true", 
                        help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Get API directory
    api_app_path = Path(__file__).parent / "src" / "api" / "app.py"
    
    if not api_app_path.exists():
        logger.error(f"API app not found at {api_app_path}")
        sys.exit(1)
        
    logger.info(f"Starting API server on {args.host}:{args.port}")
    
    # Add the project root to the Python path
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Run the server using the module path instead of changing directories
    logger.info(f"Python path: {sys.path}")
    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload or args.debug
    )

if __name__ == "__main__":
    main() 