#!/usr/bin/env python
# ============ Base imports ======================
import os
import glob
import argparse
import csv
from datetime import datetime
# ====== External package imports ================
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import seaborn as sns
# ====== Internal package imports ================
from jakarta_analyze.modules.utils.misc import run_and_catch_exceptions
# ============== Logging  ========================
import logging
from jakarta_analyze.modules.utils.setup import setup, IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from jakarta_analyze.modules.utils.config_loader import get_config, load_config

# ================================================


def load_metadata_files(frame_stats_dir, packet_stats_dir):
    """
    Load the extracted metadata files from the specified directories.
    
    Args:
        frame_stats_dir: Directory containing frame statistics CSV files
        packet_stats_dir: Directory containing packet statistics CSV files
        
    Returns:
        Dictionary containing dataframes with metadata by video
    """
    metadata = {}
    
    # Load frame statistics
    frame_files = glob.glob(os.path.join(frame_stats_dir, "*.csv"))
    for file_path in frame_files:
        try:
            video_name = os.path.basename(file_path).split('.')[0]
            if video_name not in metadata:
                metadata[video_name] = {}
            
            df = pd.read_csv(file_path)
            metadata[video_name]['frame_stats'] = df
            logger.info(f"Loaded frame stats for {video_name}: {len(df)} records")
        except Exception as e:
            logger.error(f"Error loading frame stats from {file_path}: {e}")
    
    # Load packet statistics
    packet_files = glob.glob(os.path.join(packet_stats_dir, "*.csv"))
    for file_path in packet_files:
        try:
            video_name = os.path.basename(file_path).split('.')[0]
            if video_name not in metadata:
                metadata[video_name] = {}
            
            df = pd.read_csv(file_path)
            metadata[video_name]['packet_stats'] = df
            logger.info(f"Loaded packet stats for {video_name}: {len(df)} records")
        except Exception as e:
            logger.error(f"Error loading packet stats from {file_path}: {e}")
    
    return metadata


def generate_frame_type_distribution(metadata, output_dir, file_format='pdf'):
    """
    Generate visualization of frame type distributions.
    
    Args:
        metadata: Dictionary containing metadata dataframes
        output_dir: Directory to save output visualizations
        file_format: Output file format (pdf, png, jpg)
    """
    for video_name, data in metadata.items():
        if 'frame_stats' not in data:
            continue
        
        try:
            frame_stats = data['frame_stats']
            if 'pict_type' not in frame_stats.columns:
                logger.warning(f"No picture type data for {video_name}, skipping frame type distribution")
                continue
                
            # Count frame types
            frame_type_counts = frame_stats['pict_type'].value_counts()
            
            # Create visualization
            plt.figure(figsize=(10, 6))
            ax = frame_type_counts.plot(kind='bar', color='skyblue')
            plt.title(f'Frame Type Distribution for {video_name}')
            plt.xlabel('Frame Type')
            plt.ylabel('Count')
            plt.xticks(rotation=0)
            
            # Add count labels on top of bars
            for i, count in enumerate(frame_type_counts):
                ax.text(i, count + (max(frame_type_counts) * 0.02), str(count), 
                        ha='center', va='bottom', fontweight='bold')
            
            # Save figure
            output_path = os.path.join(output_dir, f"{video_name}_frame_distribution.{file_format}")
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            logger.info(f"Generated frame type distribution for {video_name}")
        except Exception as e:
            logger.error(f"Error generating frame type distribution for {video_name}: {e}")


def generate_bitrate_analysis(metadata, output_dir, file_format='pdf'):
    """
    Generate visualization of bitrate over time.
    
    Args:
        metadata: Dictionary containing metadata dataframes
        output_dir: Directory to save output visualizations
        file_format: Output file format (pdf, png, jpg)
    """
    for video_name, data in metadata.items():
        if 'packet_stats' not in data:
            continue
            
        try:
            packet_stats = data['packet_stats']
            if 'size' not in packet_stats.columns or 'pts_time' not in packet_stats.columns:
                logger.warning(f"Missing required columns for {video_name}, skipping bitrate analysis")
                continue
                
            # Convert pts_time to float if it's not already
            packet_stats['pts_time'] = pd.to_numeric(packet_stats['pts_time'], errors='coerce')
            
            # Sort by time
            packet_stats = packet_stats.sort_values('pts_time')
            
            # Create 1-second bins
            packet_stats['time_bin'] = packet_stats['pts_time'].apply(lambda x: int(x))
            
            # Calculate size per bin (bits per second)
            bitrate = packet_stats.groupby('time_bin')['size'].sum() * 8 / 1000  # kbps
            
            # Create visualization
            plt.figure(figsize=(12, 6))
            plt.plot(bitrate.index, bitrate.values, '-', lw=2)
            plt.title(f'Bitrate Over Time for {video_name}')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Bitrate (kbps)')
            plt.grid(True, alpha=0.3)
            
            # Add average bitrate line
            avg_bitrate = bitrate.mean()
            plt.axhline(y=avg_bitrate, color='r', linestyle='--', alpha=0.7, 
                        label=f'Avg: {avg_bitrate:.2f} kbps')
            plt.legend()
            
            # Save figure
            output_path = os.path.join(output_dir, f"{video_name}_bitrate_analysis.{file_format}")
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            logger.info(f"Generated bitrate analysis for {video_name}")
        except Exception as e:
            logger.error(f"Error generating bitrate analysis for {video_name}: {e}")


def generate_frame_size_analysis(metadata, output_dir, file_format='pdf'):
    """
    Generate visualization of frame sizes by type.
    
    Args:
        metadata: Dictionary containing metadata dataframes
        output_dir: Directory to save output visualizations
        file_format: Output file format (pdf, png, jpg)
    """
    for video_name, data in metadata.items():
        if 'frame_stats' not in data:
            continue
            
        try:
            frame_stats = data['frame_stats']
            if 'pkt_size' not in frame_stats.columns or 'pict_type' not in frame_stats.columns:
                logger.warning(f"Missing required columns for {video_name}, skipping frame size analysis")
                continue
                
            # Convert pkt_size to numeric if it's not already
            frame_stats['pkt_size'] = pd.to_numeric(frame_stats['pkt_size'], errors='coerce')
            
            # Create visualization
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='pict_type', y='pkt_size', data=frame_stats)
            plt.title(f'Frame Size by Type for {video_name}')
            plt.xlabel('Frame Type')
            plt.ylabel('Size (bytes)')
            
            # Save figure
            output_path = os.path.join(output_dir, f"{video_name}_frame_size_analysis.{file_format}")
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            logger.info(f"Generated frame size analysis for {video_name}")
        except Exception as e:
            logger.error(f"Error generating frame size analysis for {video_name}: {e}")


def generate_keyframe_interval_analysis(metadata, output_dir, file_format='pdf'):
    """
    Generate visualization of keyframe intervals.
    
    Args:
        metadata: Dictionary containing metadata dataframes
        output_dir: Directory to save output visualizations
        file_format: Output file format (pdf, png, jpg)
    """
    for video_name, data in metadata.items():
        if 'frame_stats' not in data:
            continue
            
        try:
            frame_stats = data['frame_stats']
            if 'key_frame' not in frame_stats.columns or 'pkt_pts_time' not in frame_stats.columns:
                logger.warning(f"Missing required columns for {video_name}, skipping keyframe interval analysis")
                continue
                
            # Convert columns to appropriate types
            frame_stats['key_frame'] = frame_stats['key_frame'].astype(int)
            frame_stats['pkt_pts_time'] = pd.to_numeric(frame_stats['pkt_pts_time'], errors='coerce')
            
            # Find keyframes
            keyframes = frame_stats[frame_stats['key_frame'] == 1]
            
            if len(keyframes) <= 1:
                logger.warning(f"Not enough keyframes in {video_name} for interval analysis")
                continue
                
            # Calculate intervals between keyframes
            keyframe_times = keyframes['pkt_pts_time'].values
            intervals = np.diff(keyframe_times)
            
            # Create visualization
            plt.figure(figsize=(12, 6))
            
            # Histogram of intervals
            plt.subplot(1, 2, 1)
            plt.hist(intervals, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Keyframe Interval Distribution')
            plt.xlabel('Interval (seconds)')
            plt.ylabel('Frequency')
            
            # Time series of intervals
            plt.subplot(1, 2, 2)
            plt.plot(keyframe_times[:-1], intervals, 'o-', markersize=4)
            plt.title('Keyframe Intervals Over Time')
            plt.xlabel('Video Time (seconds)')
            plt.ylabel('Interval to Next Keyframe (seconds)')
            plt.grid(True, alpha=0.3)
            
            # Save figure
            output_path = os.path.join(output_dir, f"{video_name}_keyframe_interval_analysis.{file_format}")
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            logger.info(f"Generated keyframe interval analysis for {video_name}")
        except Exception as e:
            logger.error(f"Error generating keyframe interval analysis for {video_name}: {e}")


def generate_summary_report(metadata, output_dir, file_format='pdf'):
    """
    Generate a summary report with key metrics for all videos.
    
    Args:
        metadata: Dictionary containing metadata dataframes
        output_dir: Directory to save output visualizations
        file_format: Output file format (pdf, png, jpg)
    """
    summary_data = []
    
    for video_name, data in metadata.items():
        video_summary = {'video_name': video_name}
        
        # Process frame statistics
        if 'frame_stats' in data:
            frame_stats = data['frame_stats']
            
            # Count frames by type
            if 'pict_type' in frame_stats.columns:
                frame_types = frame_stats['pict_type'].value_counts().to_dict()
                for ftype, count in frame_types.items():
                    video_summary[f'frame_type_{ftype}'] = count
                    
            # Total frames
            video_summary['total_frames'] = len(frame_stats)
            
            # Average frame size
            if 'pkt_size' in frame_stats.columns:
                video_summary['avg_frame_size'] = frame_stats['pkt_size'].mean()
            
        # Process packet statistics
        if 'packet_stats' in data:
            packet_stats = data['packet_stats']
            
            # Calculate video duration if possible
            if 'pts_time' in packet_stats.columns:
                video_summary['duration'] = packet_stats['pts_time'].max()
            
            # Calculate average bitrate
            if 'size' in packet_stats.columns and 'pts_time' in packet_stats.columns and video_summary.get('duration', 0) > 0:
                total_bits = packet_stats['size'].sum() * 8
                video_summary['avg_bitrate_kbps'] = total_bits / (video_summary['duration'] * 1000)
        
        summary_data.append(video_summary)
    
    # Create summary dataframe
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Save as CSV
        summary_csv_path = os.path.join(output_dir, "video_metadata_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        logger.info(f"Generated summary report: {summary_csv_path}")
        
        # Create visualization of key metrics
        try:
            plt.figure(figsize=(15, 10))
            
            # Plot total frames by video
            plt.subplot(2, 2, 1)
            if 'total_frames' in summary_df.columns:
                sns.barplot(x='video_name', y='total_frames', data=summary_df)
                plt.title('Total Frames by Video')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
            
            # Plot average bitrate by video
            plt.subplot(2, 2, 2)
            if 'avg_bitrate_kbps' in summary_df.columns:
                sns.barplot(x='video_name', y='avg_bitrate_kbps', data=summary_df)
                plt.title('Average Bitrate by Video (kbps)')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
            
            # Plot video duration
            plt.subplot(2, 2, 3)
            if 'duration' in summary_df.columns:
                sns.barplot(x='video_name', y='duration', data=summary_df)
                plt.title('Video Duration (seconds)')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
            
            # Plot average frame size
            plt.subplot(2, 2, 4)
            if 'avg_frame_size' in summary_df.columns:
                sns.barplot(x='video_name', y='avg_frame_size', data=summary_df)
                plt.title('Average Frame Size (bytes)')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
            
            # Save figure
            summary_viz_path = os.path.join(output_dir, f"video_summary_metrics.{file_format}")
            plt.tight_layout()
            plt.savefig(summary_viz_path)
            plt.close()
            logger.info(f"Generated summary visualization: {summary_viz_path}")
        except Exception as e:
            logger.error(f"Error generating summary visualization: {e}")
    else:
        logger.warning("No summary data available to generate report")


def visualize_metadata(videos_dir=None, output_dir=None, file_format='pdf'):
    """
    Visualize metadata extracted from video files.
    
    Args:
        videos_dir: Optional directory path to override the config setting
        output_dir: Optional directory path to override the config setting
        file_format: Output file format (pdf, png, jpg)
        
    Returns:
        True if visualization completed successfully, False otherwise
    """
    # Get configuration
    conf = get_config()
    logger.info("Starting video metadata visualization process")
    
    # Get directories from arguments or config
    frame_stats_dir = os.path.join(conf.get('dirs', {}).get('frame_stats', 'outputs/frame_stats/'))
    packet_stats_dir = os.path.join(conf.get('dirs', {}).get('packet_stats', 'outputs/packet_stats/'))
    
    # Set output directory
    if not output_dir:
        output_dir = os.path.join(conf.get('dirs', {}).get('visualizations', 'outputs/visualizations/'))
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metadata
    metadata = load_metadata_files(frame_stats_dir, packet_stats_dir)
    if not metadata:
        logger.error("No metadata found to visualize")
        return False
    
    logger.info(f"Loaded metadata for {len(metadata)} videos")
    
    # Generate visualizations
    generate_frame_type_distribution(metadata, output_dir, file_format)
    generate_bitrate_analysis(metadata, output_dir, file_format)
    generate_frame_size_analysis(metadata, output_dir, file_format) 
    generate_keyframe_interval_analysis(metadata, output_dir, file_format)
    generate_summary_report(metadata, output_dir, file_format)
    
    logger.info(f"Visualization completed. Output saved to {output_dir}")
    return True


def main(argv=None):
    """Main entry point for video metadata visualization"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Visualize video metadata")
    parser.add_argument('-c', '--config', help='Path to YAML config file')
    parser.add_argument('-v', '--videos-dir', help='Directory containing videos (overrides config)')
    parser.add_argument('-o', '--output-dir', help='Directory to save visualizations (overrides config)')
    parser.add_argument('--format', choices=['pdf', 'png', 'jpg'], default='pdf',
                        help='Output format for visualizations')
    
    # Parse arguments
    args = parser.parse_args(argv)
    
    # If a config was provided, load it
    if args.config:
        load_config(args.config)
    
    # Run visualization
    result = visualize_metadata(videos_dir=args.videos_dir, 
                               output_dir=args.output_dir,
                               file_format=args.format)
    
    if result:
        logger.info("Metadata visualization completed successfully")
    else:
        logger.error("Metadata visualization failed")
    return result


if __name__ == "__main__":
    setup("visualize_metadata")
    run_and_catch_exceptions(logger, main)