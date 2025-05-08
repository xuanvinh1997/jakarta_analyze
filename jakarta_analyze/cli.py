#!/usr/bin/env python
# ============ Base imports ======================
import argparse
import sys
import importlib
# ====== External package imports ================
# ====== Internal package imports ================
import jakarta_analyze
from jakarta_analyze.modules.utils.setup import setup, IndentLogger
# ============== Logging  ========================
import logging
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
# ================================================


def main():
    """Main entry point for the Jakarta Analyze CLI"""
    # Setup parser
    parser = argparse.ArgumentParser(
        description="Jakarta Traffic Analysis Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add version info
    parser.add_argument('--version', action='version', 
                      version=f'Jakarta Analyze {jakarta_analyze.__version__}')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Extract Video Metadata command
    extract_parser = subparsers.add_parser('extract-metadata', 
                                           help='Extract metadata from video files')
    extract_parser.add_argument('-c', '--config',
                              help='Path to YAML config file')
    extract_parser.add_argument('-v', '--videos-dir',
                              help='Directory containing videos to process (overrides config)')
    extract_parser.add_argument('-l', '--log', choices=['debug', 'info', 'warning', 'error'],
                              default='info', help='Logging level')
    
    # Visualize Metadata command
    visualize_parser = subparsers.add_parser('visualize-metadata',
                                           help='Visualize extracted video metadata')
    visualize_parser.add_argument('-c', '--config',
                              help='Path to YAML config file')
    visualize_parser.add_argument('-v', '--videos-dir',
                              help='Directory containing videos to process (overrides config)')
    visualize_parser.add_argument('-o', '--output-dir',
                              help='Directory to save visualizations (overrides config)')
    visualize_parser.add_argument('--format', choices=['pdf', 'png', 'jpg'], 
                              default='pdf', help='Output format for visualizations')
    visualize_parser.add_argument('-l', '--log', choices=['debug', 'info', 'warning', 'error'],
                              default='info', help='Logging level')
    
    # Download Videos command
    download_parser = subparsers.add_parser('download',
                                           help='Download videos from Jakarta CCTV cameras')
    download_parser.add_argument('-c', '--config', required=True,
                              help='Path to YAML config file with camera URLs')
    download_parser.add_argument('-m', '--minutes', type=int, default=5,
                              help='Minutes of footage to download (default: 5)')
    download_parser.add_argument('-o', '--output',
                              help='Directory to save videos (overrides config)')
    download_parser.add_argument('-l', '--log', choices=['debug', 'info', 'warning', 'error'],
                              default='info', help='Logging level')
    
    # Pipeline command - runs the full video processing pipeline
    pipeline_parser = subparsers.add_parser('pipeline',
                                           help='Run the full video processing pipeline')
    pipeline_parser.add_argument('-c', '--config', required=True,
                              help='Path to YAML pipeline configuration file')
    pipeline_parser.add_argument('-v', '--videos-dir',
                              help='Directory containing videos to process (overrides config)')
    pipeline_parser.add_argument('-o', '--output-dir',
                              help='Directory to save output (overrides config)')
    pipeline_parser.add_argument('-l', '--log', choices=['debug', 'info', 'warning', 'error'],
                              default='info', help='Logging level')
    
    # Download Model command
    model_parser = subparsers.add_parser('download-model',
                                        help='Download models from Ultralytics Hub')
    model_parser.add_argument('model_name',
                             help='Name of the model to download (e.g., yolov8n, yolo11m)')
    model_parser.add_argument('-o', '--output-dir',
                             help='Directory to save the model to (overrides config)')
    model_parser.add_argument('-f', '--force', action='store_true',
                             help='Force download even if model already exists')
    model_parser.add_argument('--api-key',
                             help='Ultralytics API key for private models (or set ULTRALYTICS_API_KEY env var)')
    model_parser.add_argument('-l', '--log', choices=['debug', 'info', 'warning', 'error'],
                              default='info', help='Logging level')
    model_parser.add_argument('--list', action='store_true',
                             help='List available models instead of downloading')
    model_parser.add_argument('--hub', action='store_true',
                             help='Download from Ultralytics Hub using API (requires ultralytics package)')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
        
    # Initialize logging
    setup(args.command)
    
    # Execute the appropriate command
    if args.command == 'extract-metadata':
        # Import the module dynamically
        module = importlib.import_module('jakarta_analyze.main.extract_video_metadata')
        # Call the main function with parsed arguments
        cmd_args = []
        if args.config:
            cmd_args.extend(['--config', args.config])
        if args.videos_dir:
            cmd_args.extend(['--videos-dir', args.videos_dir])
        return module.main(cmd_args)
    elif args.command == 'visualize-metadata':
        # Import the module dynamically
        module = importlib.import_module('jakarta_analyze.main.visualize_metadata')
        # Call the main function with parsed arguments
        cmd_args = []
        if args.config:
            cmd_args.extend(['--config', args.config])
        if args.videos_dir:
            cmd_args.extend(['--videos-dir', args.videos_dir])
        if args.output_dir:
            cmd_args.extend(['--output-dir', args.output_dir])
        cmd_args.extend(['--format', args.format])
        return module.main(cmd_args)
    elif args.command == 'download':
        # Import the module dynamically
        module = importlib.import_module('jakarta_analyze.main.download_videos')
        # Call the main function with parsed arguments
        return module.main(['--config', args.config, 
                           '--minutes', str(args.minutes)] + 
                           (['--output', args.output] if args.output else []))
    elif args.command == 'pipeline':
        # Import the module dynamically
        module = importlib.import_module('jakarta_analyze.main.run_pipeline')
        # Call the main function with parsed arguments
        cmd_args = ['--config', args.config]
        if args.videos_dir:
            cmd_args.extend(['--videos-dir', args.videos_dir])
        if args.output_dir:
            cmd_args.extend(['--output-dir', args.output_dir])
        return module.main(cmd_args)
    elif args.command == 'download-model':
        # Import the module dynamically
        module = importlib.import_module('jakarta_analyze.main.download_model')
        # Call the main function with parsed arguments
        cmd_args = [args.model_name]
        if args.output_dir:
            cmd_args.extend(['--output-dir', args.output_dir])
        if args.force:
            cmd_args.append('--force')
        if args.api_key:
            cmd_args.extend(['--api-key', args.api_key])
        if args.list:
            cmd_args.append('--list')
        if args.hub:
            cmd_args.append('--hub')
        return module.main(cmd_args)
    elif args.command == 'setup-mongodb':
        # Import the module dynamically
        module = importlib.import_module('jakarta_analyze.scripts.setup_mongodb')
        # Call the main function with parsed arguments
        cmd_args = []
        if args.drop:
            cmd_args.append('--drop')
        return module.setup_mongodb(cmd_args)
    else:
        logger.error(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())