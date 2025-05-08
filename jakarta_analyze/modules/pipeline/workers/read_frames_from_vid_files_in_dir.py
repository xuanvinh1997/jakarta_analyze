# ============ Base imports ======================
import os
import re
import shlex
import subprocess as sp
from functools import partial
# ====== External package imports ================
import numpy as np
# ====== Internal package imports ================
from jakarta_analyze.modules.pipeline.pipeline_worker import PipelineWorker
from jakarta_analyze.modules.data.database_io import DatabaseIO
# ============== Logging  ========================
import logging
from jakarta_analyze.modules.utils.setup import IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from jakarta_analyze.modules.utils.config_loader import get_config
conf = get_config()
# ================================================


class ReadFramesFromVidFilesInDir(PipelineWorker):
    """Breaks videos in a directory into individual frames that can be processed through the pipeline
    """
    def initialize(self, vid_dir, file_regex, **kwargs):
        """Initialize with directory and file pattern
        
        Args:
            vid_dir (str): Directory containing video files
            file_regex (str): Regular expression to match video files
        """
        self.vid_dir = vid_dir
        self.file_regex = file_regex
        self.dbio = DatabaseIO()
        self.logger.info(f"Initialized with directory: {vid_dir}, regex: {file_regex}")

    def startup(self):
        """Startup operations
        """
        self.logger.info(f"Starting up ReadFramesFromVidFilesInDir worker for {self.vid_dir}")

    def run(self, *args, **kwargs):
        """For each video in a folder, break into individual frames and pass to pipeline
        
        Reads frames from multiple video files in a directory using ffmpeg and sends them
        to the next worker.
        """
        # Find video files matching regex
        vid_files = [f for f in os.listdir(self.vid_dir) if re.search(self.file_regex, f)]
        self.logger.info(f"Found {len(vid_files)} files matching regex: {self.file_regex}")
        
        if len(vid_files) == 0:
            self.logger.warning(f"No video files found in {self.vid_dir} matching regex: {self.file_regex}")
            return
            
        nl = "\n"
        self.logger.debug(f"Files to process: {nl.join(vid_files)}")
        
        # Process each video file
        for i, vid_file in enumerate(vid_files):
            # Get video info from database or file
            info_dict = self.dbio.get_video_info(vid_file)
            
            if info_dict is None:  # For video chunks not in the database
                # Try to get info from the original video it came from
                self.logger.info(f"Video info not found for {vid_file}, trying to find parent video")
                substring = vid_file[:vid_file.find("_", vid_file.find("part")+1)] + ".%"
                info_dict = self.dbio.get_video_info(substring)
                
            if info_dict is None:
                # Auto-register the video file in the database
                self.logger.info(f"Attempting to auto-register video file: {vid_file}")
                full_path = os.path.join(self.vid_dir, vid_file)
                info_dict = self.dbio.register_video_file(full_path)
                
            if info_dict is None:
                self.logger.error(f"Cannot get video info for: {vid_file}, skipping")
                continue
            # self.logger.warning(f"Video info: {info_dict}")
            # Extract video parameters
            self.height = info_dict["height"]
            self.width = info_dict["width"]
            self.path = info_dict["file_path"]
            self.fps = info_dict["fps"]
            self.uuid = info_dict["_id"]
            
            # Full path to the video file
            path = os.path.join(self.vid_dir, vid_file)
            
            # Calculate buffer size for ffmpeg
            imsize = 3 * self.height * self.width  # 3 bytes per pixel (RGB)
            
            self.logger.info(f"Reading from file {i+1} of {len(vid_files)}: {path}, "
                            f"height:{self.height}, width:{self.width}, fps:{self.fps}, uuid:{self.uuid}")
            
            # Use ffmpeg to read video frames
            commands = shlex.split(f'ffmpeg -r {self.fps} -i {path} -f image2pipe -pix_fmt rgb24 -vsync 0 -vcodec rawvideo -')
            p = sp.Popen(commands, stdout=sp.PIPE, stderr=sp.DEVNULL, bufsize=int(imsize))
            
            # Process each frame
            frame_count = 0
            for raw_frame in iter(partial(p.stdout.read, imsize), b''):
                frame_count += 1
                try:
                    frame = np.frombuffer(raw_frame, dtype='uint8').reshape((self.height, self.width, 3))
                    
                    # Create item to send to next worker
                    item = {
                        "ops": [],
                        "video_info": {
                            "id": self.uuid,
                            "file_name": vid_file,
                            "fps": self.fps,
                            "height": self.height,
                            "width": self.width,
                        },
                        "frame_number": frame_count,
                        "frame": frame,
                    }
                    self.done_with_item(item)
                    
                    # Log progress periodically
                    if frame_count % 100 == 0:
                        self.logger.info(f"Processed {frame_count} frames from {vid_file}")
                        
                except Exception as e:
                    self.logger.info(f"Done reading from file: {path}")
                    break
                    
            self.logger.info(f"Completed processing {vid_file}, {frame_count} frames processed")
            
        self.logger.info(f"Done reading all files in directory: {self.vid_dir}")

    def shutdown(self):
        """Shutdown operations
        """
        self.logger.info("Shutting down ReadFramesFromVidFilesInDir worker")