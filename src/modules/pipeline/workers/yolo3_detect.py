# ============ Base imports ======================
import os
import struct
# ====== External package imports ================
import numpy as np
import colorsys
import cv2
import torch
from ultralytics import YOLO
# ====== Internal package imports ================
from src.modules.pipeline.workers.pipeline_worker import PipelineWorker
# ============== Logging  ========================
import logging
from src.modules.utils.setup import setup, IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from src.modules.utils.config_loader import get_config
conf = get_config()
# ================================================


class Yolo3Detect(PipelineWorker):
    """Implementation of YOLOv11 for object detection and classification using Ultralytics.
    """
    def initialize(self, buffer_size, gpu, gpu_share, weights_path, frame_key, annotate_result_frame_key,
                   object_detect_threshold, class_nonzero_threshold, non_maximal_box_suppression,
                   non_maximal_box_suppression_threshold, annotation_font_scale, **kwargs):
        """
        buffer_size = how many frames to wait for
        frame_key = frame ID
        annotate_result_frame_key = result from previous process, if applicable
        weights_path = path to weights file or model name (e.g., 'yolov11n.pt')
        gpu = activate GPU
        gpu_share = how much of the GPU should be used in %
        object_detect_threshold = confidence threshold for drawing a bounding box
        class_nonzero_threshold = confidence threshold for keeping bounding boxes
        non_maximal_box_suppression = whether to apply NMS
        non_maximal_box_suppression_threshold = IOU threshold for NMS
        annotation_font_scale = font scale for annotations
        """
        self.font_scale = annotation_font_scale
        self.buffer_size = buffer_size
        self.frame_key = frame_key
        self.annotate_result_frame_key = annotate_result_frame_key
        
        # Model loading is now handled by Ultralytics
        # Use a specified weights path or a default model
        self.weights_path = weights_path or "yolov11n.pt"
        
        # GPU settings
        self.gpu = gpu
        self.gpu_frac = gpu_share
        
        # Detection thresholds
        self.object_detect_threshold = object_detect_threshold
        self.class_nonzero_threshold = class_nonzero_threshold
        self.non_maximal_box_suppression = non_maximal_box_suppression
        self.non_maximal_box_suppression_threshold = non_maximal_box_suppression_threshold
        
        # YOLOv11 uses COCO classes by default, filter to our important traffic classes
        self.important_labels = ["person", "bicycle", "car", "motorcycle", "bus", "train", "truck"]
        
        # Map legacy class names to YOLOv11 names
        self.legacy_to_new_labels = {
            "pedestrian": "person",
            "motorbike": "motorcycle"
        }
        
        # Set up colors for annotating frames
        n = len(self.important_labels)
        hsv_tuples = [(x*1.0/n, 0.5, 0.5) for x in range(n)]
        self.label_colors = list(map(lambda x: tuple([255*val for val in colorsys.hsv_to_rgb(*x)]), hsv_tuples))

    def startup(self):
        """Startup. Set up GPU environment and load the model.
        """
        # Set environment variables for GPU usage
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(self.gpu)
        
        # Load model with Ultralytics - much simpler than the custom implementation
        device = "cuda:0" if self.gpu is not None else "cpu"
        self.logger.info(f"Loading YOLO model {self.weights_path} on {device}")
        self.model = YOLO(self.weights_path)
        
        # Create buffers
        self.buffer_fill = 0
        self.buffer = []
        self.failed_draw = 0
        self.box_id = 0

    def run(self, item):
        """Process frames with object detection and assign class predictions.
        """
        self.buffer.append(item)
        self.buffer_fill += 1
        if self.buffer_fill == self.buffer_size:
            self.logger.info("Buffer full, processing")
            frames = [item[self.frame_key] for item in self.buffer]
            
            # Process frames in batch
            results = self.detect_images(frames, self.buffer)
            
            # Process detection results for each frame
            for i, (result, item) in enumerate(zip(results, self.buffer)):
                # Prepare the annotated image with boxes
                annotated_image = result.plot(
                    conf=self.object_detect_threshold,
                    line_width=2,
                    font_size=self.font_scale*13,
                    labels=True
                )
                
                # Extract boxes, objectness, and class information
                boxes, objectness, predicted_classes, classify_scores = self.extract_detection_info(result)
                
                # Add detection data to the item
                item[self.annotate_result_frame_key] = annotated_image
                item["boxes"] = boxes
                num_boxes = len(boxes)
                box_ids = list(range(self.box_id, self.box_id + num_boxes))
                self.box_id += num_boxes
                item["box_id"] = box_ids
                item["object_classes"] = self.important_labels
                item["classes"] = self.important_labels
                item["objectness"] = objectness
                item["predicted_classes"] = predicted_classes
                item["classify_scores"] = classify_scores
                item["boxes_header"] = ["xtl", "ytl", "xbr", "ybr", "objectness"] + self.important_labels
                
                self.done_with_item(item)
            
            # Reset buffer
            self.buffer_fill = 0
            self.buffer = []

    def shutdown(self):
        """Process any remaining frames in the buffer before shutting down.
        """
        if self.buffer_fill != 0:
            self.logger.debug(f"Cleaning out buffer: {self.buffer_fill} frames")
            frames = [item[self.frame_key] for item in self.buffer]
            
            # Process remaining frames
            results = self.detect_images(frames, self.buffer)
            
            # Process detection results
            for i, (result, item) in enumerate(zip(results, self.buffer)):
                annotated_image = result.plot(
                    conf=self.object_detect_threshold,
                    line_width=2,
                    font_size=self.font_scale*13,
                    labels=True
                )
                
                boxes, objectness, predicted_classes, classify_scores = self.extract_detection_info(result)
                
                item[self.annotate_result_frame_key] = annotated_image
                item["boxes"] = boxes
                num_boxes = len(boxes)
                box_ids = list(range(self.box_id, self.box_id + num_boxes))
                self.box_id += num_boxes
                item["box_id"] = box_ids
                item["object_classes"] = self.important_labels
                item["classes"] = self.important_labels
                item["objectness"] = objectness
                item["predicted_classes"] = predicted_classes
                item["classify_scores"] = classify_scores
                item["boxes_header"] = ["xtl", "ytl", "xbr", "ybr", "objectness"] + self.important_labels
                
                self.done_with_item(item)
            
            # Reset buffer
            self.buffer_fill = 0
            self.buffer = []
        
        if self.failed_draw > 0:
            self.logger.error(f"Could not draw {self.failed_draw} boxes")

    #==============================
    #= Support Functions/Classes ==
    #==============================
    def detect_images(self, images, items):
        """Detect objects in a batch of images using YOLOv11.
        
        Args:
            images: List of images to process
            items: List of frame dictionary items
            
        Returns:
            List of YOLO results objects
        """
        self.logger.debug("Starting batch detect")
        
        # Run inference with the model
        # Ultralytics YOLO handles batching automatically
        results = self.model(
            images,
            conf=self.class_nonzero_threshold,  # Class confidence threshold
            iou=self.non_maximal_box_suppression_threshold,  # NMS IOU threshold
            classes=[0, 1, 2, 3, 5, 6, 7],  # Filter classes to our important ones (COCO indices)
            verbose=False
        )
        
        self.logger.debug("Finished batch detect")
        return results

    def extract_detection_info(self, result):
        """Extract detection information from YOLO result.
        
        Args:
            result: A single YOLO result object
            
        Returns:
            Tuple of (boxes, objectness, predicted_classes, classify_scores)
        """
        # Check if there are any detections
        if len(result.boxes) == 0:
            return np.array([]), np.array([[]]), np.array([[]]), np.array([[]])
        
        # Get boxes in the same format as the original implementation
        boxes = []
        objectness = []
        predicted_classes = []
        classify_scores = []
        
        # Map COCO class indices to our important labels
        coco_to_important = {
            0: "person",      # person
            1: "bicycle",     # bicycle
            2: "car",         # car
            3: "motorcycle",  # motorcycle
            5: "bus",         # bus
            6: "train",       # train
            7: "truck"        # truck
        }
        
        for box in result.boxes:
            # Get box coordinates [x1, y1, x2, y2]
            xyxy = box.xyxy.cpu().numpy()[0]
            
            # Get confidence and class information
            conf = float(box.conf.cpu().numpy()[0])
            cls_idx = int(box.cls.cpu().numpy()[0])
            
            # Only process if it's one of our important classes
            if cls_idx in coco_to_important:
                # Add box coordinates and confidence
                box_entry = list(xyxy)
                box_entry.append(conf)  # objectness
                
                # Add class probabilities (one-hot style like the original code)
                class_scores = [0.0] * len(self.important_labels)
                class_name = coco_to_important.get(cls_idx)
                
                if class_name in self.important_labels:
                    class_idx = self.important_labels.index(class_name)
                    class_scores[class_idx] = conf
                
                box_entry.extend(class_scores)
                boxes.append(box_entry)
                
                # Store detection info
                objectness.append(conf)
                predicted_classes.append(class_name)
                classify_scores.append(conf)
        
        if boxes:
            boxes_array = np.array(boxes)
            return boxes_array, np.array(objectness), predicted_classes, np.array(classify_scores)
        else:
            return np.array([]), np.array([[]]), np.array([[]]), np.array([[]])
