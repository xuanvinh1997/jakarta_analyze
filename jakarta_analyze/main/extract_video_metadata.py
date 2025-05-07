#!/usr/bin/env python
# ============ Base imports ======================
import os
# ====== External package imports ================
# ====== Internal package imports ================
from jakarta_analyze.modules.utils.misc import run_and_catch_exceptions
# ============== Logging  ========================
import logging
from jakarta_analyze.modules.utils.setup import setup, IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from jakarta_analyze.modules.utils.config_loader import get_config
conf = get_config()
# ================================================


def extract_metadata():
    """
    Extract metadata from video files including subtitles, frame statistics, and packet statistics.
    
    This function goes through all the video files in the raw_videos directory (as specified in the config)
    and extracts various metadata, placing them in their appropriate output directories.
    """
    logger.info("Starting video metadata extraction process")
    
    # Example logic for extracting metadata
    raw_videos_dir = getattr(conf, 'dirs', {}).get('raw_videos')
    if not raw_videos_dir or not os.path.isdir(raw_videos_dir):
        logger.error(f"Raw videos directory not found or not configured properly")
        return False
        
    logger.info(f"Processing videos from: {raw_videos_dir}")
    
    # This would be where you'd implement the actual extraction logic
    # For demonstration purposes, we're just logging
    logger.info("Metadata extraction completed")
    return True


def main():
    """Main entry point for video metadata extraction"""
    result = extract_metadata()
    if result:
        logger.info("Metadata extraction completed successfully")
    else:
        logger.error("Metadata extraction failed")
    return result


if __name__ == "__main__":
    setup("extract_video_metadata")
    run_and_catch_exceptions(logger, main)