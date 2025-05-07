#!/usr/bin/env python
# ============ Base imports ======================
import os
import sys
import argparse
# ====== External package imports ================
# ====== Internal package imports ================
from jakarta_analyze.modules.data.download import download_videos
from jakarta_analyze.modules.utils.setup import setup, IndentLogger
# ============== Logging  ========================
import logging
logger = IndentLogger(logging.getLogger(''), {})
# ================================================


def parse_args(args):
    """Parse command line arguments
    
    Args:
        args: Command line arguments
        
    Returns:
        Parsed arguments object
    """
    parser = argparse.ArgumentParser(description='Download videos from Jakarta CCTV cameras')
    
    parser.add_argument('--config', '-c', required=True,
                        help='Path to YAML config file with camera URLs')
    
    parser.add_argument('--minutes', '-m', type=int, default=5,
                        help='Minutes of footage to download (default: 5)')
    
    parser.add_argument('--output', '-o',
                        help='Directory to save videos (overrides config)')
    
    return parser.parse_args(args)


def main(args=None):
    """Main entry point
    
    Args:
        args: Command line arguments (optional)
        
    Returns:
        int: Exit code (0 on success, non-zero on error)
    """
    if args is None:
        args = sys.argv[1:]
        
    try:
        parsed_args = parse_args(args)
        
        # Check if config file exists
        if not os.path.isfile(parsed_args.config):
            logger.error(f"Config file not found: {parsed_args.config}")
            return 1
            
        # Download videos
        downloaded_files = download_videos(
            parsed_args.config,
            minutes=parsed_args.minutes,
            output_dir=parsed_args.output
        )
        
        # Report results
        if downloaded_files:
            logger.info(f"Successfully downloaded {len(downloaded_files)} videos")
            return 0
        else:
            logger.error("No videos were downloaded")
            return 1
            
    except Exception as e:
        logger.error(f"Error downloading videos: {str(e)}")
        return 1


if __name__ == "__main__":
    setup("download_videos")
    sys.exit(main())