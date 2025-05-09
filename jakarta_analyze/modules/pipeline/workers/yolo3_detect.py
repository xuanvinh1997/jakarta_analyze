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


class Yolo3Detect(PipelineWorker):
    """Object detection using Ultralytics YOLO
    """
    def initialize(self, frame_key, annotate_result_frame_key=None, weights_path=None, 
                  object_detect_threshold=0.5, non_maximal_box_suppression_threshold=0.3, 
                  draw_boxes=True, class_nonzero_threshold=0.5, non_maximal_box_suppression=True,
                  classes_filter=None, verify_boxes=True, min_box_area=100, aspect_ratio_range=(0.2, 5.0), **kwargs):
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
        
        # YOLO model will be loaded in startup()
        self.model = None
        
        self.logger.info(f"Initialized with weights: {weights_path}, "
                        f"confidence threshold: {object_detect_threshold}, "
                        f"nms threshold: {non_maximal_box_suppression_threshold}, "
                        f"verify_boxes: {verify_boxes}")
                        
        # Generate random colors for class visualization
        np.random.seed(42)  # for reproducibility
        self.colors = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)  # More than enough colors

    def startup(self):
        """Startup operations - load YOLO model
        """
        self.logger.info("Starting up Ultralytics YOLO detector")
        
        # Make sure YOLO weights file exists
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"YOLO weights file not found: {self.weights_path}")
        
        # Load YOLO model using Ultralytics
        self.logger.info(f"Loading YOLO model from {self.weights_path}")
        start_time = time.time()
        
        # Load the model with specified parameters
        self.model = YOLO(self.weights_path)
        
        self.logger.info(f"YOLO model loaded in {time.time() - start_time:.2f} seconds")
        self.logger.info(f"Model information: {self.model.info()}")

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
        
        # For vehicle classes (cars, trucks, buses), verify higher confidence for larger objects
        vehicle_classes = [2, 5, 7]  # car, bus, truck
        if class_id in vehicle_classes and area > 10000 and confidence < 0.6:
            return False
            
        # For pedestrians, verify proportions
        if class_id == 0 and (aspect_ratio > 0.8 or height < width):
            # Pedestrians are typically taller than wide
            if confidence < 0.7:  # require higher confidence for unusual proportions
                return False
        
        return True

    def run(self, item):
        """Detect objects in a frame using Ultralytics YOLO
        
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
        
        # Create annotated frame if requested
        if self.annotate_frame_key:
            # Use the plotted result from Ultralytics if available
            if hasattr(result, 'plot') and callable(getattr(result, 'plot')):
                annotated_frame = result.plot() if self.draw_boxes else frame.copy()
            else:
                annotated_frame = frame.copy()
        else:
            annotated_frame = None
        
        # Process detection results
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            boxes = result.boxes
            
            for box in boxes:
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
                    
                    # Draw bounding box on annotated frame if not already drawn by result.plot()
                    if annotated_frame is not None and self.draw_boxes and not hasattr(result, 'plot'):
                        color = tuple(map(int, self.colors[class_id % len(self.colors)]))
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Store results in item
        item["boxes"] = detected_boxes
        item["boxes_header"] = "x1,y1,x2,y2,confidence,class_id,class_name"
        
        if self.annotate_frame_key:
            item[self.annotate_frame_key] = annotated_frame
        
        # Log periodically
        if frame_number % 100 == 0:
            self.logger.debug(f"Processed frame {frame_number}, detected {len(detected_boxes)} valid objects out of {len(result.boxes)} detections")
        
        # Pass the item to the next worker
        self.done_with_item(item)

    def shutdown(self):
        """Shutdown operations - cleanup resources
        """
        self.logger.info("Shutting down Ultralytics YOLO detector")
        # Release model resources if needed
        self.model = None