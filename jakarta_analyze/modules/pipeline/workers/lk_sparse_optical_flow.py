# ============ Base imports ======================
# ====== External package imports ================
import numpy as np
import cv2
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


class LKSparseOpticalFlow(PipelineWorker):
    """Implements Lucas-Kanade sparse optical flow for tracking points in video frames
    """
    def initialize(self, frame_key, annotate_frame_key, annotate_result_frame_key, new_point_detect_interval, 
                  path_track_length, good_flow_difference_threshold, new_point_occlusion_radius, bg_mask_key, 
                  winSize, maxLevel, maxCorners, qualityLevel, minDistance, blockSize, backward_pass, 
                  new_point_detect_interval_per_second, how_many_track_new_points_before_clearing_points, **kwargs):
        """Initialize with optical flow parameters
        
        Args:
            frame_key (str): Key to access frame data
            annotate_frame_key (str): Key to store annotated frame
            annotate_result_frame_key (str): Key to store result frame
            new_point_detect_interval (int): Interval to detect new points
            path_track_length (int): Length of path to track
            good_flow_difference_threshold (float): Threshold for good flow points
            new_point_occlusion_radius (int): Radius around detected points
            bg_mask_key (str): Key for background mask
            winSize (list): Size of search window at each pyramid level
            maxLevel (int): Maximal pyramid level number
            maxCorners (int): Maximum number of corners to track
            qualityLevel (float): Minimum corner quality
            minDistance (float): Minimum distance between corners
            blockSize (int): Size of the block for corner detection
            backward_pass (bool): Whether to perform backward pass
            new_point_detect_interval_per_second (int): Interval to detect new points per second
            how_many_track_new_points_before_clearing_points (int): Track count before clearing points
        """
        self.frame_key = frame_key
        self.annotate_frame_key = annotate_frame_key
        self.annotate_result_frame_key = annotate_result_frame_key
        self.new_point_detect_interval = new_point_detect_interval
        self.path_track_length = path_track_length
        self.good_flow_difference_threshold = good_flow_difference_threshold
        self.new_point_occlusion_radius = new_point_occlusion_radius
        self.bg_mask_key = bg_mask_key
        self.winSize = tuple(winSize)
        self.maxLevel = maxLevel
        self.maxCorners = maxCorners
        self.qualityLevel = qualityLevel
        self.minDistance = minDistance
        self.blockSize = blockSize
        self.backward_pass = backward_pass
        self.new_point_detect_interval_per_second = new_point_detect_interval_per_second
        self.how_many_track_new_points_before_clearing_points = how_many_track_new_points_before_clearing_points
        
        # Initialize tracking structures
        self.old_gray = None
        self.old_points = None
        self.paths = []  # Each path is a list of points
        self.point_id_counter = 0
        self.point_ids = []  # ID for each point
        self.point_start_frames = []  # Frame number where each point starts
        self.tracking_count = 0  # Counter for tracking cycles

        # Set up optical flow parameters
        self.lk_params = dict(winSize=self.winSize, 
                             maxLevel=self.maxLevel,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.feature_params = dict(maxCorners=self.maxCorners,
                                 qualityLevel=self.qualityLevel,
                                 minDistance=self.minDistance,
                                 blockSize=self.blockSize)
                                 
        self.logger.info(f"Initialized with winSize: {self.winSize}, maxLevel: {self.maxLevel}, "
                        f"maxCorners: {self.maxCorners}, qualityLevel: {self.qualityLevel}")

    def startup(self):
        """Startup operations
        """
        self.logger.info("Starting up LKSparseOpticalFlow worker")

    def run(self, item):
        """Track points across frames using Lucas-Kanade optical flow
        
        Args:
            item: Item containing frame data
        """
        # Check if frame exists
        if self.frame_key not in item:
            self.logger.warning(f"Frame key '{self.frame_key}' not found in item")
            self.done_with_item(item)
            return
            
        frame = item[self.frame_key]
        frame_number = item.get('frame_number', -1)
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Create annotated frame if requested
        if self.annotate_frame_key:
            item[self.annotate_frame_key] = frame.copy()
            
        # For the first frame
        if self.old_gray is None:
            self.detect_new_points(gray, frame_number, None)
            self.old_gray = gray.copy()
            self.tracking_count = 0
            
            # Initialize output data
            item["points"] = np.zeros((0, 2))
            item["point_ids"] = np.zeros(0, dtype=np.int32)
            item["paths"] = [[] for _ in range(len(self.paths))]
            item["point_start_frames"] = self.point_start_frames.copy()
            item["flows"] = np.zeros((0, 2))
            
            self.done_with_item(item)
            return
            
        # Calculate optical flow
        if len(self.old_points) > 0:
            new_points, status, err = cv2.calcOpticalFlowPyrLK(
                self.old_gray, gray, self.old_points, None, **self.lk_params)
                
            # If backward pass is requested, verify points
            if self.backward_pass and len(new_points) > 0:
                old_points_back, status_back, err_back = cv2.calcOpticalFlowPyrLK(
                    gray, self.old_gray, new_points, None, **self.lk_params)
                
                # Calculate absolute difference between original points and back-tracked points
                abs_diff = np.abs(self.old_points - old_points_back).reshape(-1, 2).max(-1)
                good_points = abs_diff < self.good_flow_difference_threshold
                
                # Update status with good_points
                status = status & good_points.reshape(-1, 1)
            
            # Filter and keep good points
            good_new_points = new_points[status.flatten() == 1]
            good_old_points = self.old_points[status.flatten() == 1]
            good_point_ids = [self.point_ids[i] for i, s in enumerate(status.flatten()) if s == 1]
            good_point_start_frames = [self.point_start_frames[i] for i, s in enumerate(status.flatten()) if s == 1]
            
            # Calculate flow vectors
            flows = good_new_points - good_old_points
            
            # Update paths
            for i, (new, pid) in enumerate(zip(good_new_points, good_point_ids)):
                found = False
                for j, path in enumerate(self.paths):
                    if path and path[0][2] == pid:  # Match by point ID
                        # Add new point to path (x, y, id)
                        path.append((new[0][0], new[0][1], pid))
                        if len(path) > self.path_track_length:
                            path.pop(0)
                        found = True
                        break
                        
                if not found:
                    # Create new path if not found
                    self.paths.append([(good_new_points[i][0][0], good_new_points[i][0][1], pid)])
            
            # Draw tracks on annotated frame if requested
            if self.annotate_frame_key and self.annotate_frame_key in item:
                for path in self.paths:
                    if len(path) > 1:
                        for i in range(1, len(path)):
                            cv2.line(item[self.annotate_frame_key], 
                                    (int(path[i-1][0]), int(path[i-1][1])), 
                                    (int(path[i][0]), int(path[i][1])), 
                                    (0, 255, 0), 2)
                        
                        # Draw current point
                        cv2.circle(item[self.annotate_frame_key], 
                                 (int(path[-1][0]), int(path[-1][1])), 
                                 3, (0, 0, 255), -1)
            
            # Update output data in item
            item["points"] = good_new_points.reshape(-1, 2)
            item["point_ids"] = np.array(good_point_ids, dtype=np.int32)
            item["point_start_frames"] = np.array(good_point_start_frames, dtype=np.int32)
            item["paths"] = self.paths
            item["flows"] = flows.reshape(-1, 2)
            
            # Update tracking points
            self.old_points = good_new_points.reshape(-1, 1, 2)
            self.point_ids = good_point_ids
            self.point_start_frames = good_point_start_frames
        
        # Detect new points periodically
        self.tracking_count += 1
        fps = item.get('video_info', {}).get('fps', 30)
        interval = self.new_point_detect_interval_per_second * fps if self.new_point_detect_interval_per_second else self.new_point_detect_interval
        
        if self.tracking_count >= interval:
            if len(self.paths) >= self.how_many_track_new_points_before_clearing_points:
                # Clear all points and start fresh
                self.paths = []
                self.old_points = None
                self.point_ids = []
                self.point_start_frames = []
            
            # Get mask for background subtraction if available
            mask = item.get(self.bg_mask_key) if self.bg_mask_key and self.bg_mask_key in item else None
            
            # Detect new points
            self.detect_new_points(gray, frame_number, mask)
            self.tracking_count = 0
        
        # Update old frame
        self.old_gray = gray.copy()
        
        # Pass the item to the next worker
        self.done_with_item(item)

    def shutdown(self):
        """Shutdown operations
        """
        self.logger.info("Shutting down LKSparseOpticalFlow worker")
        
    def detect_new_points(self, gray, frame_number, mask=None):
        """Detect new points to track in the current frame
        
        Args:
            gray: Grayscale frame
            frame_number: Current frame number
            mask: Optional mask for point detection
        """
        # Create mask to avoid detecting points near existing points
        if mask is None:
            mask = np.ones_like(gray, dtype=np.uint8) * 255
        
        # Avoid detecting points near existing points
        if self.old_points is not None and len(self.old_points) > 0:
            for point in self.old_points:
                x, y = point[0]
                cv2.circle(mask, (int(x), int(y)), self.new_point_occlusion_radius, 0, -1)
        
        # Detect new points
        new_points = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
        
        if new_points is not None and len(new_points) > 0:
            # Initialize or update points
            if self.old_points is None:
                self.old_points = new_points
                self.point_ids = list(range(len(new_points)))
                self.point_start_frames = [frame_number] * len(new_points)
                self.point_id_counter = len(new_points)
            else:
                # Append new points to existing ones
                self.old_points = np.concatenate([self.old_points, new_points], axis=0)
                
                # Create new IDs for new points
                new_ids = list(range(self.point_id_counter, self.point_id_counter + len(new_points)))
                self.point_ids.extend(new_ids)
                self.point_start_frames.extend([frame_number] * len(new_points))
                self.point_id_counter += len(new_points)
            
            # Initialize new paths
            for i, point in enumerate(new_points):
                point_id = self.point_ids[len(self.old_points) - len(new_points) + i]
                self.paths.append([(point[0][0], point[0][1], point_id)])
                
            self.logger.debug(f"Detected {len(new_points)} new points, total: {len(self.old_points)}")