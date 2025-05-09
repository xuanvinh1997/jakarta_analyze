# ============ Base imports ======================
import os
import shlex
import subprocess as sp
import time
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
        
        # Ensure output directory exists
        if hasattr(self, 'out_path') and self.out_path:
            if not os.path.exists(self.out_path):
                try:
                    os.makedirs(self.out_path, exist_ok=True)
                    self.logger.info(f"Created output directory: {self.out_path}")
                except Exception as e:
                    self.logger.error(f"Failed to create output directory {self.out_path}: {str(e)}")
        else:
            self.logger.warning("Output path not specified, videos may not be saved correctly")

    def startup(self):
        """Startup operations
        """
        self.logger.info(f"Starting up WriteFramesToVidFiles worker")
        
        # Verify output directory and FFmpeg at startup
        if not os.path.exists(self.out_path):
            try:
                os.makedirs(self.out_path, exist_ok=True)
                self.logger.info(f"Created output directory: {self.out_path}")
            except Exception as e:
                self.logger.error(f"Failed to create output directory {self.out_path}: {str(e)}")
        
        # Check if ffmpeg is available
        try:
            result = sp.run(['ffmpeg', '-version'], stdout=sp.PIPE, stderr=sp.PIPE)
            if result.returncode == 0:
                self.logger.info("FFmpeg available for video encoding")
            else:
                self.logger.warning("FFmpeg check returned non-zero exit code, video writing may fail")
        except Exception as e:
            self.logger.error(f"FFmpeg not found or error checking FFmpeg: {str(e)}")

    def __init__(self, name=None, input_queue=None, output_queues=None, pipeline_config=None, start_time=None, model_number=None, out_path=None, **kwargs):
        """Initialize the worker
        
        Args:
            name (str): Worker name
            input_queue (Queue): Input queue
            output_queues (list): List of output queues
            pipeline_config (dict): Pipeline configuration
            start_time (float): Pipeline start time
            model_number (str): Model identifier
            out_path (str): Output path
            **kwargs: Additional parameters including buffer_size and frame_key
        """
        # Extract required parameters for initialize before calling parent constructor
        self.buffer_size = kwargs.get('buffer_size', 100)
        self.frame_key = kwargs.get('frame_key', 'frame')
        
        # Call parent constructor - pass kwargs through since initialize will be called by parent
        super().__init__(input_queue=input_queue, output_queues=output_queues, pipeline_config=pipeline_config, 
                         start_time=start_time, model_number=model_number, out_path=out_path, **kwargs)
        
        # Initialize time-based flushing parameters
        self.last_write_time = time.time()
        self.write_interval = kwargs.get('write_interval', 30)  # Seconds between forced writes

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
        # self.logger.warning(f"Processing item: {item.keys()}")
        
        # Look for the primary frame key, or fall back to alternatives if not found
        frame_key_to_use = self.frame_key
        if self.frame_key not in item:
            # Try fallback keys: boxed_frame, frame or any key ending with "frame"
            fallback_keys = ["boxed_frame", "frame"]
            frame_keys = [k for k in item.keys() if "frame" in k.lower()]
            possible_keys = fallback_keys + frame_keys
            
            for key in possible_keys:
                if key in item:
                    frame_key_to_use = key
                    # self.logger.warning(f"Frame key '{self.frame_key}' not found, using '{key}' instead")
                    break
        
        # Add frame to buffer if a usable frame key was found
        if frame_key_to_use in item:
            frame = item[frame_key_to_use]
            self.buffer.append(frame.tobytes())
            self.frame_count += 1
            
            # Determine if we should write frames based on buffer size or time interval
            current_time = time.time()
            time_elapsed = current_time - self.last_write_time
            
            # If buffer is full or enough time has passed since last write, write to file
            if (len(self.buffer) >= self.buffer_size) or (len(self.buffer) > 0 and time_elapsed >= self.write_interval):
                outpath = os.path.join(self.out_path, f"{self.base_name}_{frame_key_to_use}_model_{self.model_number}_part_{self.part}.mkv")
                self.logger.warning(f"Writing {len(self.buffer)} frames to video file: {outpath} (buffer_full={len(self.buffer) >= self.buffer_size}, time_elapsed={time_elapsed:.1f}s)")
                
                # Ensure output directory exists before writing
                os.makedirs(os.path.dirname(outpath), exist_ok=True)
                
                # Use ffmpeg to write frames to file
                commands = shlex.split(f'ffmpeg -y -f rawvideo -vcodec rawvideo '
                                      f'-s {self.vid_info["width"]}x{self.vid_info["height"]} '
                                      f'-pix_fmt rgb24 -r {self.vid_info["fps"]} '
                                      f'-i - -an -vcodec libx264 -vsync 0 -pix_fmt yuv420p {outpath}')
                
                try:
                    p = sp.Popen(commands, stdin=sp.PIPE, stderr=sp.PIPE, bufsize=int(self.imsize))
                    stdout, stderr = p.communicate(input=b''.join(self.buffer))
                    
                    if p.returncode != 0:
                        self.logger.error(f"FFmpeg error writing video file: {stderr.decode('utf-8', errors='ignore') if stderr else 'Unknown error'}")
                    else:
                        # Check if the file was actually created and has content
                        if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
                            self.logger.info(f"Successfully written {len(self.buffer)} frames to {outpath}")
                        else:
                            self.logger.error(f"Failed to write video file or file is empty: {outpath}")
                            
                except Exception as e:
                    self.logger.error(f"Error writing video file {outpath}: {str(e)}")
                
                # Clear buffer, increment part number, and update last write time
                self.buffer = []
                self.part += 1
                self.last_write_time = current_time
        else:
            self.logger.warning(f"Frame key '{self.frame_key}' not found in item and no suitable alternatives found. Available keys: {item.keys()}")

    def shutdown(self):
        """Send videos to outpath and shutdown
        """
        # Write any remaining frames to file
        if self.vid_info is not None and self.buffer:
            outpath = os.path.join(self.out_path, f"{self.base_name}_{self.frame_key}_model_{self.model_number}_part_{self.part}.mkv")
            self.logger.info(f"Writing final {len(self.buffer)} frames to video file: {outpath}")
            
            # Ensure output directory exists before writing
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            
            commands = shlex.split(f'ffmpeg -y -f rawvideo -vcodec rawvideo '
                                  f'-s {self.vid_info["width"]}x{self.vid_info["height"]} '
                                  f'-pix_fmt rgb24 -r {self.vid_info["fps"]} '
                                  f'-i - -an -vcodec libx264 -vsync 0 -pix_fmt yuv420p {outpath}')
            
            try:
                p = sp.Popen(commands, stdin=sp.PIPE, stderr=sp.PIPE, bufsize=int(self.imsize))
                stdout, stderr = p.communicate(input=b''.join(self.buffer))
                
                if p.returncode != 0:
                    self.logger.error(f"FFmpeg error writing final video file: {stderr.decode('utf-8', errors='ignore') if stderr else 'Unknown error'}")
                else:
                    # Check if the file was actually created and has content
                    if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
                        self.logger.info(f"Successfully written {len(self.buffer)} frames to {outpath}")
                    else:
                        self.logger.error(f"Failed to write final video file or file is empty: {outpath}")
                        
            except Exception as e:
                self.logger.error(f"Error writing final video file {outpath}: {str(e)}")
            
        self.logger.info(f"Processed a total of {self.frame_count} frames")
        self.logger.info("Shutting down WriteFramesToVidFiles worker")