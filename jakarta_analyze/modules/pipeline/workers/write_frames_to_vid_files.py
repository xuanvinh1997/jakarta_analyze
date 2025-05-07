# ============ Base imports ======================
import os
import shlex
import subprocess as sp
# ====== External package imports ================
import numpy as np
# ====== Internal package imports ================
from jakarta_analyze.modules.pipeline.pipeline_worker import PipelineWorker
# ============== Logging  ========================
import logging
from jakarta_analyze.modules.utils.setup import IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from jakarta_analyze.modules.utils.config_loader import get_config
conf = get_config()
# ================================================


class WriteFramesToVidFiles(PipelineWorker):
    """Put frames back together into a video file either in the middle or at the end of the pipeline
    """
    def initialize(self, buffer_size, frame_key, **kwargs):
        """Initialize with buffer size and frame key
        
        Args:
            buffer_size (int): Number of frames to buffer before writing to file
            frame_key (str): Key to use to access frame data in the item
        """
        self.buffer_size = buffer_size
        self.frame_key = frame_key if frame_key else "frame"
        self.buffer = []
        self.vid_info = None
        self.base_name = None
        self.frame_count = 0
        self.part = 0
        self.imsize = None
        self.logger.info(f"Initialized with buffer size: {buffer_size}, frame key: {self.frame_key}")

    def startup(self):
        """Startup operations
        """
        self.logger.info(f"Starting up WriteFramesToVidFiles worker")

    def run(self, item):
        """Waits for a specified number of frames to fill, then appends them to each other and writes a video file
        
        Args:
            item: Item containing frame data
        """
        if self.vid_info is None:
            # Initialize with video info from the first frame
            self.vid_info = item["video_info"]
            self.base_name = self.vid_info["file_name"].split('.')[0] if "file_name" in self.vid_info else "output"
            self.imsize = 3 * self.vid_info["height"] * self.vid_info["width"]
            self.logger.info(f"New video info: {self.vid_info['file_name'] if 'file_name' in self.vid_info else 'unknown'}")
            
        # Add frame to buffer
        if self.frame_key in item:
            frame = item[self.frame_key]
            self.buffer.append(frame.tobytes())
            self.frame_count += 1
            
            # If buffer is full, write to file
            if len(self.buffer) >= self.buffer_size:
                outpath = os.path.join(self.out_path, f"{self.base_name}_{self.frame_key}_model_{self.model_number}_part_{self.part}.mkv")
                self.logger.info(f"Writing {len(self.buffer)} frames to video file: {outpath}")
                
                # Use ffmpeg to write frames to file
                commands = shlex.split(f'ffmpeg -y -f rawvideo -vcodec rawvideo '
                                      f'-s {self.vid_info["width"]}x{self.vid_info["height"]} '
                                      f'-pix_fmt rgb24 -r {self.vid_info["fps"]} '
                                      f'-i - -an -vcodec libx264 -vsync 0 -pix_fmt yuv420p {outpath}')
                
                p = sp.Popen(commands, stdin=sp.PIPE, bufsize=int(self.imsize))
                p.communicate(input=b''.join(self.buffer))
                
                # Clear buffer and increment part number
                self.buffer = []
                self.part += 1
        else:
            self.logger.warning(f"Frame key '{self.frame_key}' not found in item")

    def shutdown(self):
        """Send videos to outpath and shutdown
        """
        # Write any remaining frames to file
        if self.vid_info is not None and self.buffer:
            outpath = os.path.join(self.out_path, f"{self.base_name}_{self.frame_key}_model_{self.model_number}_part_{self.part}.mkv")
            self.logger.info(f"Writing final {len(self.buffer)} frames to video file: {outpath}")
            
            commands = shlex.split(f'ffmpeg -y -f rawvideo -vcodec rawvideo '
                                  f'-s {self.vid_info["width"]}x{self.vid_info["height"]} '
                                  f'-pix_fmt rgb24 -r {self.vid_info["fps"]} '
                                  f'-i - -an -vcodec libx264 -vsync 0 -pix_fmt yuv420p {outpath}')
            
            p = sp.Popen(commands, stdin=sp.PIPE, bufsize=int(self.imsize))
            p.communicate(input=b''.join(self.buffer))
            
        self.logger.info(f"Processed a total of {self.frame_count} frames")
        self.logger.info("Shutting down WriteFramesToVidFiles worker")