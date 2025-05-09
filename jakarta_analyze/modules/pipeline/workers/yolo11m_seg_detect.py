# ============ Base imports ======================
import os
import time
# ====== External package imports ================
import numpy as np
import cv2
from ultralytics import YOLO
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

# Color mapping for segmentation classes
SEGMENT_COLORS = {
    'sidewalk': np.array([232, 35, 244]),  # Pink/purple for sidewalk
    'road': np.array([128, 64, 128]),      # Purple-blue for road
}

class Yolo11mSegDetect(PipelineWorker):
    """Object detection and segmentation using Ultralytics YOLOv11m with segmentation
    
    This worker uses YOLOv11m-seg to detect objects and provide segmentation masks.
    It has special logic to detect and classify motorcycles on sidewalks.
    """
    def initialize(self, frame_key, annotate_result_frame_key=None, weights_path=None, 
                  object_detect_threshold=0.5, non_maximal_box_suppression_threshold=0.3, 
                  draw_boxes=True, class_nonzero_threshold=0.5, non_maximal_box_suppression=True,
                  classes_filter=None, verify_boxes=True, min_box_area=100, aspect_ratio_range=(0.2, 5.0),
                  sidewalk_overlap_threshold=0.3, **kwargs):
        """Initialize YOLO detection
        
        Args:
            frame_key (str): Key for accessing frame in item dictionary
            annotate_result_frame_key (str): Key for storing annotated frame
            weights_path (str): Path to YOLO weights
            object_detect_threshold (float): Confidence threshold for detections
            non_maximal_box_suppression_threshold (float): IoU threshold for NMS
            draw_boxes (bool): Whether to draw boxes on annotated frame
            class_nonzero_threshold (float): Threshold for class confidence
            non_maximal_box_suppression (bool): Whether to apply NMS
            classes_filter (list): List of class indices to keep (None for all classes)
            verify_boxes (bool): Whether to apply additional verification to boxes
            min_box_area (int): Minimum box area in pixels to be considered valid
            aspect_ratio_range (tuple): Valid range for aspect ratio (width/height)
            sidewalk_overlap_threshold (float): Threshold to determine if a motorcycle is on sidewalk
        """
        self.frame_key = frame_key
        self.annotate_frame_key = annotate_result_frame_key
        self.weights_path = weights_path
        self.confidence_threshold = object_detect_threshold
        self.nms_threshold = non_maximal_box_suppression_threshold
        self.draw_boxes = draw_boxes
        self.classes_filter = classes_filter
        self.non_maximal_box_suppression = non_maximal_box_suppression
        self.class_threshold = class_nonzero_threshold
        self.verify_boxes = verify_boxes
        self.min_box_area = min_box_area
        self.aspect_ratio_range = aspect_ratio_range
        self.sidewalk_overlap_threshold = sidewalk_overlap_threshold
        
        # YOLO model will be loaded in startup()
        self.model = None
        
        self.logger.info(f"Initialized with weights: {weights_path}, "
                        f"confidence threshold: {object_detect_threshold}, "
                        f"nms threshold: {non_maximal_box_suppression_threshold}, "
                        f"verify_boxes: {verify_boxes}, "
                        f"sidewalk_overlap_threshold: {sidewalk_overlap_threshold}")
                        
        # Generate random colors for class visualization
        np.random.seed(42)  # for reproducibility
        self.colors = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)  # More than enough colors

    def startup(self):
        """Startup operations - load YOLO model
        """
        self.logger.info("Starting up Ultralytics YOLOv11m-seg detector")
        
        # Make sure YOLO weights file exists
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"YOLO weights file not found: {self.weights_path}")
        
        # Load YOLO model using Ultralytics
        self.logger.info(f"Loading YOLOv11m-seg model from {self.weights_path}")
        start_time = time.time()
        
        # Load the model with segmentation support
        self.model = YOLO(self.weights_path)
        
        self.logger.info(f"YOLOv11m-seg model loaded in {time.time() - start_time:.2f} seconds")
        self.logger.info(f"Model information: {self.model.info()}")
        
        # Check if model has segmentation capability
        if not hasattr(self.model, 'names') or not hasattr(self.model, 'task') or self.model.task != 'segment':
            self.logger.warning("The loaded model doesn't appear to support segmentation. Using detection only.")

    def verify_detection(self, box):
        """Verify if the detected box meets additional quality criteria
        
        Args:
            box (dict): Detection box with x1, y1, x2, y2, confidence, class_id
            
        Returns:
            bool: True if the box passes verification, False otherwise
        """
        # Extract box dimensions
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        width = x2 - x1
        height = y2 - y1
        
        # Check if box has valid dimensions
        if width <= 0 or height <= 0:
            return False
        
        # Check minimum area requirement
        area = width * height
        if area < self.min_box_area:
            return False
        
        # Check aspect ratio is within reasonable range
        aspect_ratio = width / height
        min_ratio, max_ratio = self.aspect_ratio_range
        if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
            return False
            
        # Additional checks based on class
        class_id = box["class_id"]
        confidence = box["confidence"]
        
        # For motorcycle class, verify proportions
        if class_id == 3:  # motorcycle class in COCO
            if aspect_ratio > 3.0 or height > width * 1.5:
                # Unusual proportions for motorcycle
                if confidence < 0.7:  # require higher confidence 
                    return False
        
        return True
    
    def check_motorcycle_on_sidewalk(self, detection, mask, segmentation_masks):
        """Check if a detected motorcycle is on the sidewalk
        
        Args:
            detection (dict): Detection box
            mask (ndarray): Object mask from YOLO
            segmentation_masks (ndarray): Segmentation masks from YOLO
            
        Returns:
            bool: True if the motorcycle is on the sidewalk, False otherwise
        """
        # Extract box dimensions
        x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
        
        # Get the bottom part of the motorcycle (where it touches the ground)
        # We focus on the lower 1/3 of the bounding box
        bottom_y = int(y2 - (y2 - y1) * 0.3)
        
        # Create a mask for the bottom part of the motorcycle
        bottom_mask = np.zeros_like(mask)
        bottom_mask[bottom_y:y2, x1:x2] = mask[bottom_y:y2, x1:x2]
        
        # Check overlap with sidewalk segmentation
        sidewalk_mask = np.zeros_like(mask)
        for segment_mask, segment_class_id in segmentation_masks:
            # Assuming segment_class_id corresponds to a class name in the model
            class_name = self.model.names[segment_class_id].lower()
            if 'sidewalk' in class_name or 'pavement' in class_name:
                sidewalk_mask = np.logical_or(sidewalk_mask, segment_mask)
        
        # Calculate overlap ratio
        overlap = np.logical_and(bottom_mask, sidewalk_mask).sum()
        total_area = bottom_mask.sum()
        
        # Return True if the overlap ratio exceeds the threshold
        if total_area > 0:
            overlap_ratio = overlap / total_area
            return overlap_ratio >= self.sidewalk_overlap_threshold
        
        return False
    
    def run(self, item):
        """Detect objects in a frame using Ultralytics YOLO with segmentation
        
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
        
        # Run inference with Ultralytics YOLO
        results = self.model.predict(
            frame, 
            conf=self.confidence_threshold,  # Confidence threshold
            iou=self.nms_threshold,          # NMS IOU threshold
            classes=self.classes_filter,     # Filter by class
            verbose=False                    # Suppress detailed outputs
        )
        
        # Get the first result (single image)
        result = results[0]
        
        # Create the list of detected boxes
        detected_boxes = []
        motorcycle_sidewalk_boxes = []  # Special list for motorcycles on sidewalks
        
        # Create annotated frame if requested
        if self.annotate_frame_key:
            # Use the plotted result from Ultralytics if available
            if hasattr(result, 'plot') and callable(getattr(result, 'plot')):
                annotated_frame = result.plot() if self.draw_boxes else frame.copy()
            else:
                annotated_frame = frame.copy()
        else:
            annotated_frame = None
        
        # Get segmentation masks if available
        segmentation_masks = []
        if hasattr(result, 'masks') and result.masks is not None:
            for i in range(len(result.masks)):
                if len(result.boxes) > i:  # Make sure we have corresponding box
                    mask = result.masks[i].data.cpu().numpy()[0]  # Get mask data
                    class_id = int(result.boxes[i].cls[0])        # Get class ID
                    segmentation_masks.append((mask, class_id))
        
        # Process detection results
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            boxes = result.boxes
            
            for i, box in enumerate(boxes):
                # Get box coordinates in (x1, y1, x2, y2) format
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Get confidence and class ID
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Get class name
                class_name = result.names[class_id]
                
                # Create detection object
                detection = {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name
                }
                
                # Apply additional verification if enabled
                is_valid = True
                if self.verify_boxes:
                    is_valid = self.verify_detection(detection)
                
                if is_valid:
                    detected_boxes.append(detection)
                    
                    # Check if this is a motorcycle
                    if class_name.lower() == "motorcycle" and i < len(segmentation_masks):
                        mask = segmentation_masks[i][0]
                        
                        # Check if the motorcycle is on the sidewalk
                        if self.check_motorcycle_on_sidewalk(detection, mask, segmentation_masks):
                            # Mark this as a motorcycle on sidewalk
                            detection["on_sidewalk"] = True
                            motorcycle_sidewalk_boxes.append(detection)
                            
                            # Draw special annotation for motorcycles on sidewalks
                            if annotated_frame is not None and self.draw_boxes:
                                color = (0, 0, 255)  # Red color for violations
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                                
                                # Draw warning text
                                warning_text = "VIOLATION: Motorcycle on sidewalk"
                                cv2.putText(annotated_frame, warning_text, (x1, y1 - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Draw bounding box on annotated frame if not already drawn by result.plot()
                    elif annotated_frame is not None and self.draw_boxes and not hasattr(result, 'plot'):
                        color = tuple(map(int, self.colors[class_id % len(self.colors)]))
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Store results in item
        item["boxes"] = detected_boxes
        item["boxes_header"] = "x1,y1,x2,y2,confidence,class_id,class_name,on_sidewalk"
        item["motorcycle_sidewalk_violations"] = motorcycle_sidewalk_boxes
        
        if self.annotate_frame_key:
            item[self.annotate_frame_key] = annotated_frame
        
        # Log periodically
        if frame_number % 100 == 0:
            n_violations = len(motorcycle_sidewalk_boxes)
            self.logger.debug(f"Processed frame {frame_number}, detected {len(detected_boxes)} valid objects"
                              f" including {n_violations} motorcycle on sidewalk violations")
        
        # Pass the item to the next worker
        self.done_with_item(item)

    def shutdown(self):
        """Shutdown operations - cleanup resources
        """
        self.logger.info("Shutting down Ultralytics YOLOv11m-seg detector")
        # Release model resources if needed
        self.model = None
