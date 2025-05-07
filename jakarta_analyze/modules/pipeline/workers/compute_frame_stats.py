# ============ Base imports ======================
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


class ComputeFrameStats(PipelineWorker):
    """Pipeline worker which computes frame level statistics based on keys passed to the object
    """
    def initialize(self, stats_type, input_key, output_key, **kwargs):
        """Initialize with statistics type and keys
        
        Args:
            stats_type (str): Type of statistics to compute
            input_key (str): Key to use to access input data
            output_key (str): Key to use to store output data
        """
        self.stats_type = stats_type
        self.input_key = input_key
        self.output_key = output_key
        self.logger.info(f"Initialized with stats type: {stats_type}, input key: {input_key}, output key: {output_key}")

    def startup(self):
        """Startup operations
        """
        self.logger.info(f"Starting up ComputeFrameStats worker")

    def run(self, item):
        """Compute statistics on frame data
        
        Args:
            item: Item containing detection data
        """
        if self.input_key not in item:
            self.logger.warning(f"Input key '{self.input_key}' not found in item")
            self.done_with_item(item)
            return
            
        # Process based on statistics type
        if self.stats_type == "count_by_class":
            result = self._count_objects_by_class(item[self.input_key])
        elif self.stats_type == "confidence_by_class":
            result = self._compute_confidence_by_class(item[self.input_key])
        else:
            self.logger.warning(f"Unknown statistics type: {self.stats_type}")
            result = {}
        
        # Store result and pass item to next worker
        item[self.output_key] = result
        self.done_with_item(item)
        
        # Log periodically
        if item["frame_number"] % 100 == 0:
            self.logger.debug(f"Processed frame {item['frame_number']}, stats: {result}")

    def shutdown(self):
        """Shutdown operations
        """
        self.logger.info("Shutting down ComputeFrameStats worker")
        
    def _count_objects_by_class(self, detections):
        """Count objects by class
        
        Args:
            detections: List of detection objects
            
        Returns:
            dict: Counts of detected objects by class
        """
        counts = {}
        try:
            for det in detections:
                class_name = det.get("class", "unknown")
                counts[class_name] = counts.get(class_name, 0) + 1
        except Exception as e:
            self.logger.error(f"Error counting objects by class: {str(e)}")
        return counts
        
    def _compute_confidence_by_class(self, detections):
        """Compute average confidence by class
        
        Args:
            detections: List of detection objects
            
        Returns:
            dict: Average confidence scores by class
        """
        confidences = {}
        counts = {}
        try:
            for det in detections:
                class_name = det.get("class", "unknown")
                conf = det.get("confidence", 0.0)
                
                if class_name not in confidences:
                    confidences[class_name] = conf
                    counts[class_name] = 1
                else:
                    confidences[class_name] += conf
                    counts[class_name] += 1
                    
            # Calculate averages
            for cls in confidences:
                if counts[cls] > 0:
                    confidences[cls] = confidences[cls] / counts[cls]
        except Exception as e:
            self.logger.error(f"Error computing confidence by class: {str(e)}")
        return confidences