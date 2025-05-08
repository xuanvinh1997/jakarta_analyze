#!/usr/bin/env python
# ============ Base imports ======================
import argparse
import os
import sys
# ====== External package imports ================
# ====== Internal package imports ================
from jakarta_analyze.modules.models.downloader import download_model, download_from_ultralytics_hub, list_available_models
from jakarta_analyze.modules.utils.misc import run_and_catch_exceptions
# ============== Logging  ========================
import logging
from jakarta_analyze.modules.utils.setup import setup, IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from jakarta_analyze.modules.utils.config_loader import get_config, load_config
# ================================================


def main(argv=None):
    """
    Main entry point for model download functionality.
    
    Args:
        argv (list, optional): Command line arguments.
        
    Returns:
        int: Exit code. 0 for success, 1 for failure.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Download models from Ultralytics Hub")
    parser.add_argument('model_name', nargs='?',
                        help='Name of the model to download (e.g., yolov8n, yolo11m)')
    parser.add_argument('-o', '--output-dir',
                        help='Directory to save the model to (overrides config)')
    parser.add_argument('-f', '--force', action='store_true',
                        help='Force download even if model already exists')
    parser.add_argument('--api-key',
                        help='Ultralytics API key for private models (or set ULTRALYTICS_API_KEY env var)')
    parser.add_argument('--list', action='store_true',
                        help='List available models instead of downloading')
    parser.add_argument('--hub', action='store_true',
                        help='Download from Ultralytics Hub using API (requires ultralytics package)')
    parser.add_argument('-c', '--config', 
                        help='Path to YAML config file')
    
    # Parse arguments
    args = parser.parse_args(argv)
    
    # If a config was provided, load it
    if args.config:
        load_config(args.config)
    
    # Get configuration
    conf = get_config()
    
    # List available models if requested
    if args.list:
        logger.info("Available models:")
        models = list_available_models()
        for name, url in models.items():
            logger.info(f"- {name}")
        return 0
    
    # Check if model name was provided if not listing
    if not args.model_name and not args.list:
        logger.error("Model name is required unless --list is specified.")
        parser.print_help()
        return 1
    
    # Get output directory
    output_dir = args.output_dir
    if not output_dir:
        # Get models directory from config
        output_dir = conf.get('dirs', {}).get('models', 'models')
        
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Log operation
    if args.hub:
        logger.info(f"Downloading model '{args.model_name}' from Ultralytics Hub to {output_dir}")
        result = download_from_ultralytics_hub(args.model_name, api_key=args.api_key, output_dir=output_dir)
    else:
        logger.info(f"Downloading model '{args.model_name}' to {output_dir}")
        result = download_model(args.model_name, output_dir=output_dir, force=args.force)
    
    # Check result
    if result:
        logger.info(f"Successfully downloaded model to {result}")
        
        # Check if this model is specified in the pipeline configuration
        pipeline_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "pipeline.yml")
        if os.path.exists(pipeline_file):
            logger.info("Updating pipeline.yml with the downloaded model path...")
            try:
                import yaml
                with open(pipeline_file, 'r') as f:
                    pipeline_config = yaml.safe_load(f)
                
                # Update model path in pipeline config if it exists
                tasks = pipeline_config.get('pipeline', {}).get('tasks', [])
                for task in tasks:
                    if task.get('worker_type') == 'Yolo3Detect' and task.get('weights_path'):
                        relative_path = os.path.join('models', f"{args.model_name}.pt")
                        task['weights_path'] = relative_path
                        logger.info(f"Updated pipeline weights_path to '{relative_path}'")
                
                # Write updated pipeline config
                with open(pipeline_file, 'w') as f:
                    yaml.dump(pipeline_config, f, default_flow_style=False)
                
                logger.info(f"Pipeline configuration updated in {pipeline_file}")
            except Exception as e:
                logger.warning(f"Could not update pipeline configuration: {str(e)}")
        
        return 0
    else:
        logger.error(f"Failed to download model '{args.model_name}'")
        return 1


if __name__ == "__main__":
    setup("download_model")
    run_and_catch_exceptions(logger, main)