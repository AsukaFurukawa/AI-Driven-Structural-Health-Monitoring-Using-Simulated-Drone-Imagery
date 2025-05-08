#!/usr/bin/env python3
"""
Command-line interface to run the data processing pipeline.

This script provides a convenient way to run the data processing pipeline
from the project root directory.
"""
import os
import sys
import typer
import logging
import time
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.data_pipeline import DataPipeline
from src.data.full_pipeline import update_config

# Initialize typer app
app = typer.Typer(help="AI-Driven Structural Health Monitoring - Data Processing")
console = Console()

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup rich logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    return logging.getLogger("rich")

@app.command()
def process(
    config_path: str = typer.Option("config/config.yaml", "--config", "-c", help="Path to configuration file"),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", "-d", help="Override data directory"),
    clean_only: bool = typer.Option(False, "--clean-only", help="Only clean existing data, don't generate new data"),
    skip_augmentation: bool = typer.Option(False, "--skip-augmentation", help="Skip data augmentation"),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level")
):
    """
    Run the full data processing pipeline.
    
    This command processes the raw drone imagery data, cleaning and preparing it for model training.
    """
    logger = setup_logging(log_level)
    
    try:
        console.print("[bold green]Starting data processing pipeline...[/bold green]")
        
        # Create arguments object with the values from typer
        class Args:
            def __init__(self):
                self.config = config_path
                self.data_dir = data_dir
                self.clean_only = clean_only
                self.skip_augmentation = skip_augmentation
                self.log_level = log_level
        
        args = Args()
        
        # Update configuration
        config_path = Path(config_path)
        if not config_path.exists():
            console.print(f"[bold red]Error: Configuration file {config_path} not found!")
            return 1
            
        updated_config = update_config(str(config_path), args)
        
        # Create temporary config
        temp_config_path = "config/temp_config.yaml"
        os.makedirs(os.path.dirname(temp_config_path), exist_ok=True)
        
        import yaml
        with open(temp_config_path, 'w') as f:
            yaml.dump(updated_config, f)
        
        console.print("[bold green]Initializing data pipeline...[/bold green]")
        
        # Initialize pipeline
        pipeline = DataPipeline(config_path=temp_config_path)
        
        # Start timer
        start_time = time.time()
        
        # Run pipeline
        console.print("[bold green]Processing data...[/bold green]")
        success = pipeline.run_full_pipeline()
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Clean up temp config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        
        if success:
            console.print(f"[bold green]✅ Data processing completed successfully in {execution_time:.2f}s![/bold green]")
            
            # Print dataset information
            try:
                output_dir = pipeline.output_dir
                info_file = output_dir / 'dataset_info.yaml'
                if info_file.exists():
                    with open(info_file, 'r') as f:
                        info = yaml.safe_load(f)
                    
                    console.print("\n[bold cyan]Dataset Summary:[/bold cyan]")
                    console.print("=" * 40)
                    
                    if 'class_mapping' in info:
                        console.print(f"[cyan]Classes:[/cyan] {list(info['class_mapping'].keys())}")
                    
                    if 'preprocessing' in info and 'input_size' in info['preprocessing']:
                        console.print(f"[cyan]Input size:[/cyan] {info['preprocessing']['input_size']}")
                    
                    console.print(f"\n[bold green]Processed data saved to:[/bold green] {output_dir}")
            except Exception as e:
                logger.warning(f"Could not print dataset information: {str(e)}")
        else:
            console.print("[bold red]❌ Data processing failed![/bold red]")
            return 1
            
    except KeyboardInterrupt:
        console.print("[bold yellow]Processing interrupted by user.[/bold yellow]")
        return 1
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        logger.exception("Exception occurred")
        return 1
    
    return 0

@app.command()
def info():
    """
    Display information about the data pipeline.
    """
    console.print("[bold cyan]AI-Driven Structural Health Monitoring[/bold cyan]")
    console.print("[cyan]Data Processing Pipeline[/cyan]")
    console.print("\nThis tool processes drone imagery data for structural health monitoring:")
    console.print("1. Prepares and cleans datasets from multiple sources")
    console.print("2. Combines datasets into unified format")
    console.print("3. Preprocesses images for model training")
    console.print("4. Creates train/validation/test splits")
    console.print("\n[bold cyan]Usage Examples:[/bold cyan]")
    console.print("Process data with default settings:")
    console.print("  [green]python process_data.py process[/green]")
    console.print("Skip data augmentation:")
    console.print("  [green]python process_data.py process --skip-augmentation[/green]")
    console.print("Specify a custom data directory:")
    console.print("  [green]python process_data.py process --data-dir /path/to/data[/green]")

if __name__ == "__main__":
    app() 