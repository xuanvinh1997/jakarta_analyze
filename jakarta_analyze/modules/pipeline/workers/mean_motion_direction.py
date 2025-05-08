# ============ Base imports ======================
import math
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


class MeanMotionDirection(PipelineWorker):
    """Calculates the mean motion direction for detected objects using optical flow data
    """
    def initialize(self, annotate_result_frame_key=None, points_key="tracked_points", 
                  flows_key="tracked_flows", boxes_key="boxes", stationary_threshold=1, **kwargs):
        """Initialize with key references and parameters
        
        Args:
            annotate_result_frame_key (str): Key to store annotated frame with motion vectors
            points_key (str): Key for points data (defaults to "tracked_points")
            flows_key (str): Key for flows data (defaults to "tracked_flows")
            boxes_key (str): Key for detected objects boxes (defaults to "boxes")
            stationary_threshold (float): Threshold below which motion is considered stationary
        """
        self.points_key = points_key
        self.flows_key = flows_key
        self.boxes_key = boxes_key
        self.annotate_result_frame_key = annotate_result_frame_key
        self.stationary_threshold = stationary_threshold
        self.logger.info(f"Initialized with points_key: {points_key}, flows_key: {flows_key}, "
                        f"boxes_key: {boxes_key}, stationary_threshold: {stationary_threshold}")

    def startup(self):
        """Startup operations
        """
        self.logger.info("Starting up MeanMotionDirection worker")

    def run(self, item):
        """Calculate mean motion direction for detected objects
        
        Args:
            item: Item containing optical flow and object detection data
        """
        # self.logger.warning(f"Processing item: {item.keys()}")
        
        # Define fallback keys for points and flows
        points_fallbacks = ["points", "tracked_points"]
        flows_fallbacks = ["flows", "tracked_flows"]
        boxes_fallbacks = ["boxes", "detected_boxes"]
        
        # Try to get the actual keys to use
        points_key_to_use = self._get_first_available_key(item, self.points_key, points_fallbacks)
        flows_key_to_use = self._get_first_available_key(item, self.flows_key, flows_fallbacks)
        boxes_key_to_use = self._get_first_available_key(item, self.boxes_key, boxes_fallbacks)
        
        # Check if we have all the keys we need
        missing_data = False
        if points_key_to_use is None:
            # self.logger.warning(f"Missing points data. Tried keys: {self.points_key} and {points_fallbacks}")
            missing_data = True
        
        if flows_key_to_use is None:
            # self.logger.warning(f"Missing flows data. Tried keys: {self.flows_key} and {flows_fallbacks}")
            missing_data = True
            
        if boxes_key_to_use is None:
            # self.logger.warning(f"Missing boxes data. Tried keys: {self.boxes_key} and {boxes_fallbacks}")
            missing_data = True
            
        if missing_data:
            # Create empty results to avoid errors downstream
            item["points_grouped_by_box"] = []
            item["points_grouped_by_box_header"] = "box_idx,num_points,mean_dx,mean_dy,magnitude,angle_radians,angle_degrees"
            item["box_id"] = []
            self.done_with_item(item)
            return
            
        # Get data from item using the keys we found
        points = item[points_key_to_use]
        flows = item[flows_key_to_use]
        boxes = item[boxes_key_to_use]
        
        # Validate data before processing
        if not isinstance(points, np.ndarray) or not isinstance(flows, np.ndarray):
            self.logger.warning(f"Invalid data types: points={type(points)}, flows={type(flows)}")
            self.done_with_item(item)
            return
            
        if len(points) == 0 or len(flows) == 0:
            self.logger.debug("Empty points or flows data, skipping")
            # Create empty results to avoid errors downstream
            item["points_grouped_by_box"] = []
            item["points_grouped_by_box_header"] = "box_idx,num_points,mean_dx,mean_dy,magnitude,angle_radians,angle_degrees"
            item["box_id"] = []
            self.done_with_item(item)
            return
        
        try:
            # Group points by which box they belong to
            points_grouped_by_box, points_grouped_by_box_header, box_ids = self._group_points_by_box(points, flows, boxes)
            
            # Add results to item
            item["points_grouped_by_box"] = points_grouped_by_box
            item["points_grouped_by_box_header"] = points_grouped_by_box_header
            item["box_id"] = box_ids
            
            # Log periodically
            if item.get("frame_number", 0) % 100 == 0:
                self.logger.debug(f"Processed frame {item.get('frame_number', 0)}, found {len(box_ids)} boxes with motion")
        except Exception as e:
            # Log error but don't crash the pipeline
            self.logger.error(f"Error processing motion data: {str(e)}")
            # Create empty results to avoid errors downstream
            item["points_grouped_by_box"] = []
            item["points_grouped_by_box_header"] = "box_idx,num_points,mean_dx,mean_dy,magnitude,angle_radians,angle_degrees"
            item["box_id"] = []
        
        # Pass the item to the next worker
        self.done_with_item(item)

    def shutdown(self):
        """Shutdown operations
        """
        self.logger.info("Shutting down MeanMotionDirection worker")
    
    def _get_first_available_key(self, item, primary_key, fallback_keys):
        """Find the first available key from a list of keys
        
        Args:
            item (dict): Dictionary to search for keys
            primary_key (str): First key to try
            fallback_keys (list): List of fallback keys to try
            
        Returns:
            str or None: First key found in the item, or None if none found
        """
        # First try the primary key
        if primary_key in item:
            return primary_key
            
        # Then try the fallbacks
        for key in fallback_keys:
            if key in item:
                # self.logger.warning(f"Key '{primary_key}' not found, using '{key}' instead")
                return key
                
        # No keys found
        return None
    
    def _group_points_by_box(self, points, flows, boxes):
        """Group tracked points by the box they belong to
        
        Args:
            points (ndarray): Array of tracked points
            flows (ndarray): Array of optical flow vectors
            boxes (list): List of bounding boxes
            
        Returns:
            tuple: (points_by_box, header, box_ids)
                points_by_box (list): List of point data grouped by box
                header (str): CSV header for the data
                box_ids (list): List of box IDs that contain points
        """
        # Prepare result containers
        points_grouped_by_box = []
        box_ids = []
        header = "box_idx,num_points,mean_dx,mean_dy,magnitude,angle_radians,angle_degrees"
        
        # Check if we have valid data
        if len(points) == 0 or len(boxes) == 0:
            return points_grouped_by_box, header, box_ids
            
        # For each box, find points that are inside it
        for box_idx, box in enumerate(boxes):
            # Extract box coordinates (assuming format [x1, y1, x2, y2])
            if isinstance(box, dict):  # Handle dict format
                x1, y1 = box.get("x1", 0), box.get("y1", 0)
                x2, y2 = box.get("x2", 0), box.get("y2", 0)
            else:  # Handle list/tuple format
                try:
                    x1, y1, x2, y2 = box[:4]
                except (ValueError, IndexError):
                    self.logger.warning(f"Invalid box format: {box}")
                    continue
            
            # Find points inside this box
            inside_box = ((points[:, 0] >= x1) & 
                         (points[:, 0] <= x2) & 
                         (points[:, 1] >= y1) & 
                         (points[:, 1] <= y2))
            
            box_points = points[inside_box]
            box_flows = flows[inside_box]
            
            # If we have points in this box, calculate statistics
            if len(box_points) > 0:
                # Calculate mean flow
                mean_dx = np.mean(box_flows[:, 0])
                mean_dy = np.mean(box_flows[:, 1])
                
                # Calculate magnitude and angle
                magnitude = math.sqrt(mean_dx**2 + mean_dy**2)
                angle_radians = math.atan2(mean_dy, mean_dx)
                angle_degrees = math.degrees(angle_radians)
                
                # Only consider non-stationary objects
                if magnitude > self.stationary_threshold:
                    points_grouped_by_box.append([
                        box_idx,
                        len(box_points),
                        mean_dx,
                        mean_dy,
                        magnitude,
                        angle_radians,
                        angle_degrees
                    ])
                    box_ids.append(box_idx)
        
        return points_grouped_by_box, header, box_ids