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
    extract_parser.add_argument('-l', '--log', choices=['debug', 'info', 'warning', 'error'],
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
        # Call the main function
        return module.main()
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
    else:
        logger.error(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())