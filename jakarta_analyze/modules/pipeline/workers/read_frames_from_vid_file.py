# ============ Base imports ======================
import os
import shlex
import subprocess as sp
from functools import partial
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


class ReadFramesFromVidFile(PipelineWorker):
    """Breaks a video into individual frames that can be processed through the pipeline
    """
    def initialize(self, path, height, width, uuid, fps, **kwargs):
        """Initialize with video file parameters
        
        Args:
            path (str): Path to video file
            height (int): Height of video in pixels
            width (int): Width of video in pixels
            uuid (str): Unique identifier for the video
            fps (float): Frames per second
        """
        self.path = path
        self.height = height
        self.width = width
        self.uuid = uuid
        self.fps = fps
        self.logger.info(f"Initialized with video: {path}")

    def startup(self):
        """Startup operations
        """
        self.logger.info(f"Starting up ReadFramesFromVidFile worker for {self.path}")

    def run(self, *args, **kwargs):
        """Read frames from video and store video info
        
        Reads frames from a video file using ffmpeg and sends them to the next worker.
        """
        imsize = 3 * self.height * self.width  # 3 bytes per pixel (RGB)
        self.logger.info(f"Reading from file: {self.path}")
        
        # Check if file exists
        if (not os.path.exists(self.path)) or (not os.path.isfile(self.path)):
            raise FileNotFoundError(f"Not a valid video file: {self.path}")
        
        # Use ffmpeg to read video frames
        commands = shlex.split(f'ffmpeg -r {self.fps} -i {self.path} -f image2pipe -pix_fmt rgb24 -vsync 0 -vcodec rawvideo -')
        p = sp.Popen(commands, stdout=sp.PIPE, stderr=sp.DEVNULL, bufsize=int(imsize))
        
        # Process each frame
        i = 0
        for raw_frame in iter(partial(p.stdout.read, imsize), b''):
            i += 1
            try:
                frame = np.frombuffer(raw_frame, dtype='uint8').reshape((self.height, self.width, 3))
                
                # Create item to send to next worker
                item = {
                    "ops": [],
                    "frame_number": i,
                    "frame": frame,
                    "video_info": {
                        "id": self.uuid,
                        "file_path": self.path,
                        "fps": self.fps,
                        "height": self.height,
                        "width": self.width,
                    },
                }
                self.done_with_item(item)
                
                # Log progress periodically
                if i % 100 == 0:
                    self.logger.info(f"Processed {i} frames")
                
            except Exception as e:
                self.logger.error(f"Error processing frame {i}: {str(e)}")
                break
                
        self.logger.info(f"Done reading from file: {self.path}, processed {i} frames")

    def shutdown(self):
        """Shutdown operations
        """
        self.logger.info("Shutting down ReadFramesFromVidFile worker")