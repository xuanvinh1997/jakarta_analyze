#!/usr/bin/env python
# ============ Base imports ======================
import os
import requests
import datetime
import yaml
# ====== External package imports ================
# ====== Internal package imports ================
# ============== Logging  ========================
import logging
from jakarta_analyze.modules.utils.setup import IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# ================================================


class VideoDownloader:
    """Class to handle downloading videos from Jakarta's CCTV cameras"""
    
    def __init__(self, config):
        """Initialize the downloader with configuration
        
        Args:
            config (dict): Configuration containing camera URLs and download paths
        """
        self.config = config
        self.download_dir = config.get('dirs', {}).get('downloaded_videos', './downloaded_videos')
        
        # Ensure the download directory exists
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)
    
    def download_videos(self, cameras, minutes=5, timestamp=None):
        """Download videos from the specified cameras
        
        Args:
            cameras (list): List of camera URLs to download from
            minutes (int): Minutes of footage to download
            timestamp (float): Unix timestamp for end of video segment. If None, current time is used.
            
        Returns:
            list: List of downloaded file paths
        """
        logger.info(f"Starting downloads for {len(cameras)} cameras")
        
        if timestamp is None:
            current_time = datetime.datetime.now(datetime.timezone.utc)
            timestamp = current_time.timestamp()
            
        downloaded_files = []
        
        for vid in cameras:
            logger.info(f"Downloading vid: {vid}")
            seconds = minutes * 60  # Convert minutes to seconds
            unix_time = int(timestamp - seconds)
            u_time = f"{unix_time}-{seconds}.mp4"
            
            url = f"{vid}/archive-{u_time}"
            
            # Create filename based on camera ID from URL
            try:
                # Extract the camera ID from the URL
                camera_id = vid.split(".co.id/")[1]
                local_filename = os.path.join(self.download_dir, f"{camera_id}-{u_time}")
            except IndexError:
                # Fallback if URL format is unexpected
                camera_id = vid.split("/")[-1]
                local_filename = os.path.join(self.download_dir, f"camera-{camera_id}-{u_time}")
                
            # Download the file
            try:
                r = requests.get(url, stream=True, timeout=30)
                r.raise_for_status()  # Raise an exception for HTTP errors
                
                file_size = 0
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024): 
                        if chunk:  # Filter out keep-alive new chunks
                            file_size += len(chunk)
                            f.write(chunk)
                            
                logger.info(f"Downloaded {local_filename}, file size: {file_size / (1024*1024):.2f} MB")
                downloaded_files.append(local_filename)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to download {url}: {str(e)}")
                continue
                
        return downloaded_files
        
    def download_from_config(self, config_file, minutes=5):
        """Download videos from cameras listed in a config file
        
        Args:
            config_file (str): Path to YAML config file with camera URLs
            minutes (int): Minutes of footage to download
            
        Returns:
            list: List of downloaded file paths
        """
        try:
            with open(config_file, 'r') as f:
                cameras = yaml.safe_load(f)
                
            if not cameras or not isinstance(cameras, list):
                logger.error(f"Invalid camera configuration in {config_file}")
                return []
                
            return self.download_videos(cameras, minutes)
            
        except Exception as e:
            logger.error(f"Error loading camera config from {config_file}: {str(e)}")
            return []


def download_videos(config_file, minutes=5, output_dir=None):
    """Main function to download videos from config file
    
    Args:
        config_file (str): Path to YAML file with camera URLs
        minutes (int): Minutes of footage to download
        output_dir (str): Directory to save videos (overrides config)
        
    Returns:
        list: List of downloaded file paths
    """
    # Create basic config
    config = {'dirs': {}}
    
    if output_dir:
        config['dirs']['downloaded_videos'] = output_dir
    
    # Create downloader
    downloader = VideoDownloader(config)
    
    # Download videos
    return downloader.download_from_config(config_file, minutes)