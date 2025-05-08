#!/usr/bin/env python
# ============ Base imports ======================
import os
import glob
import gc
import argparse
# ====== External package imports ================
# ====== Internal package imports ================
from jakarta_analyze.modules.data.video_file import VideoFile
from jakarta_analyze.modules.utils.misc import run_and_catch_exceptions
# ============== Logging  ========================
import logging
from jakarta_analyze.modules.utils.setup import setup, IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from jakarta_analyze.modules.utils.config_loader import get_config, load_config

# ================================================


def extract_metadata(videos_dir=None):
    """
    Extract metadata from video files including subtitles, frame statistics, and packet statistics.
    
    Args:
        videos_dir: Optional directory path to override the config setting
    """
    
    # Get raw videos directory from arguments or config
    conf = get_config()
    logger.info("Starting video metadata extraction process with config: %s", conf)

    raw_videos_dir = videos_dir
    if not raw_videos_dir:
        raw_videos_dir = getattr(conf, 'dirs', {}).get('raw_videos')
    
    if not raw_videos_dir or not os.path.isdir(raw_videos_dir):
        logger.error(f"Raw videos directory not found or not configured properly: {raw_videos_dir}")
        return False
        
    logger.info(f"Processing videos from: {raw_videos_dir}")
    
    # Look for MKV files by default, but process files with any extension if they exist
    video_pattern = os.path.join(raw_videos_dir, "*.mkv")
    video_files = glob.glob(video_pattern)
    
    # If no MKV files found, check for MP4 files which are in the downloaded_videos directory
    if not video_files and os.path.exists(os.path.join(os.path.dirname(raw_videos_dir), "downloaded_videos")):
        mp4_dir = os.path.join(os.path.dirname(raw_videos_dir), "downloaded_videos")
        video_pattern = os.path.join(mp4_dir, "*.mp4")
        video_files = glob.glob(video_pattern)
    
    if not video_files:
        logger.error(f"No video files found in {raw_videos_dir} using pattern {video_pattern}")
        return False
    
    for i, f_path in enumerate(video_files):
        gc.collect()
        logger.info(f"Processing {i+1}/{len(video_files)}: {f_path}")
        try:
            vid = VideoFile(f_path)
            logger.info("Extracting subtitles")
            vid.extract_subtitles()
            logger.info("Extracting Frame statistics")
            vid.extract_frame_stats()
            logger.info("Extracting Packet statistics")
            vid.extract_packet_stats()
            logger.info(f"Done with {f_path}")
        except Exception as e:
            import traceback
            logger.error(traceback.format_exc())
            logger.error(f"Error processing {f_path}: {e}")
    
    logger.info("Metadata extraction completed")
    return True


def main(argv=None):
    """Main entry point for video metadata extraction"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Extract metadata from video files")
    parser.add_argument('-c', '--config', help='Path to YAML config file')
    parser.add_argument('-v', '--videos-dir', help='Directory containing videos to process (overrides config)')
    
    # Parse arguments
    args = parser.parse_args(argv)
    logger.info("Ready to extract metadata from videos with config: %s", args.config)
    logger.info("Videos directory: %s", args.videos_dir)
    # If a config was provided, load it
    if args.config:
        load_config(args.config)
    
    result = extract_metadata(videos_dir=args.videos_dir)
    if result:
        logger.info("Metadata extraction completed successfully")
    else:
        logger.error("Metadata extraction failed")
    return result


if __name__ == "__main__":
    setup("extract_video_metadata")
    run_and_catch_exceptions(logger, main)