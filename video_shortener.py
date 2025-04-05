#!/usr/bin/env python3
"""
Video Shortener Script

This script processes videos in an input directory and creates 10-second versions
of each video while preserving the same progression/storyline, FPS, and quality.
Audio is removed from the output videos.

The output directory structure mirrors the input directory structure.

Requirements:
- Python 3.6+
- ffmpeg-python
"""

import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging
import json
import subprocess
# Import ffmpeg-python correctly
import ffmpeg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Common video extensions
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']

def is_video_file(file_path):
    """Check if the file is a video based on its extension."""
    return os.path.splitext(file_path)[1].lower() in VIDEO_EXTENSIONS

def get_video_duration(video_path):
    """Get the duration of a video file using ffprobe."""
    try:
        probe = ffmpeg.probe(video_path)
        duration = float(probe['format']['duration'])
        return duration
    except Exception as e:
        logger.error(f"Error getting duration for {video_path}: {e}")
        return None

def create_short_video(video_path, output_path, duration=10):
    """
    Create a 10-second version of the video that preserves the progression/storyline.
    
    This works by speeding up the video to compress it to 10 seconds while
    preserving the full storyline.
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get original video duration
        original_duration = get_video_duration(video_path)
        if not original_duration:
            logger.error(f"Could not determine duration for {video_path}")
            return False
        
        # If video is already short enough, just copy it (removing audio)
        if original_duration <= duration:
            logger.info(f"Video {video_path} is already under {duration} seconds, copying with audio removed")
            # Use ffmpeg to copy the video without audio
            stream = ffmpeg.input(video_path)
            video = stream.video.copy()
            stream = ffmpeg.output(video, output_path, **{'c:v': 'copy'})
            ffmpeg.run(stream, quiet=True, overwrite_output=True)
            return True
        
        # Calculate the speed factor needed to compress the video to 10 seconds
        speed_factor = original_duration / duration
        
        # Use the setpts filter to adjust the speed of the video (speed up)
        stream = ffmpeg.input(video_path)
        video = stream.video.filter('setpts', f'{1/speed_factor}*PTS')
        stream = ffmpeg.output(
            video, 
            output_path,
            **{
                'c:v': 'libx264',  # Use H.264 codec
                'crf': '18',         # Maintain quality (lower is better)
                'preset': 'medium'    # Balance between encoding speed and compression
            }
        )
        
        # Run the ffmpeg command
        ffmpeg.run(stream, quiet=True, overwrite_output=True)
        
        logger.info(f"Created shortened video: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating short video for {video_path}: {e}")
        return False

def process_video(video_path, input_base, output_base):
    """Process a single video file, creating a short version in the output directory."""
    # Determine the relative path from the input base
    rel_path = os.path.relpath(video_path, input_base)
    
    # Create the equivalent path in the output directory
    output_path = os.path.join(output_base, rel_path)
    
    logger.info(f"Processing video: {rel_path}")
    return create_short_video(video_path, output_path)

def find_videos(directory):
    """Find all video files in a directory and its subdirectories."""
    video_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if is_video_file(file_path):
                video_files.append(file_path)
    return video_files

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Create 10-second versions of videos while preserving progression/storyline.')
    parser.add_argument('--input_dir', type=str, help='Input directory containing videos', dest='input_dir')
    parser.add_argument('--output_dir', type=str, help='Output directory for shortened videos', dest='output_dir')
    parser.add_argument('--threads', type=int, default=4, help='Number of parallel processing threads')
    
    # Process args to handle both formats:
    # 1. python script.py --input_dir=path --output_dir=path
    # 2. python script.py input_dir=path output_dir=path
    # 3. python script.py /path/to/input /path/to/output (positional)
    
    args, unknown = parser.parse_known_args()
    
    # Process any keyword args passed without '--'
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    for arg in unknown:
        if '=' in arg:
            key, value = arg.split('=', 1)
            if key == 'input_dir':
                input_dir = value
            elif key == 'output_dir':
                output_dir = value
    
    # Check for positional arguments if input_dir or output_dir are still None
    remaining_args = [arg for arg in unknown if '=' not in arg]
    if input_dir is None and len(remaining_args) >= 1:
        input_dir = remaining_args[0]
    if output_dir is None and len(remaining_args) >= 2:
        output_dir = remaining_args[1]
    
    # Ensure we have required arguments
    if input_dir is None or output_dir is None:
        logger.error("Both input_dir and output_dir are required arguments")
        parser.print_help()
        return 1
    
    # Ensure input directory exists
    if not os.path.isdir(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        return 1
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all video files in the input directory
    logger.info(f"Scanning for videos in: {input_dir}")
    video_files = find_videos(input_dir)
    logger.info(f"Found {len(video_files)} video files")
    
    if not video_files:
        logger.warning("No video files found in the input directory")
        return 0
    
    # Process videos in parallel
    success_count = 0
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [
            executor.submit(process_video, video, input_dir, output_dir)
            for video in video_files
        ]
        
        for i, future in enumerate(futures):
            if future.result():
                success_count += 1
            logger.info(f"Progress: {i+1}/{len(video_files)} videos processed")
    
    logger.info(f"Processing complete. Successfully processed {success_count}/{len(video_files)} videos.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
