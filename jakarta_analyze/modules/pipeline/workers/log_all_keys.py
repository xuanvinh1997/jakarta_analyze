# ============ Base imports ======================
import json
# ====== External package imports ================
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


class LogAllKeys(PipelineWorker):
    """Debug worker that logs all keys present in each item passing through the pipeline
    
    This worker doesn't modify the data but logs all keys it encounters, making it
    useful for debugging and understanding what data is available at different pipeline stages.
    """
    def initialize(self, log_level="INFO", log_values=False, log_sample_interval=20, **kwargs):
        """Initialize the worker
        
        Args:
            log_level (str): Logging level ("INFO", "DEBUG", etc.)
            log_values (bool): If True, also log values for simple data types (not arrays)
            log_sample_interval (int): Only log every N-th item to avoid overwhelming logs
        """
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.log_values = log_values
        self.log_sample_interval = log_sample_interval
        self.item_count = 0
        self.has_logged_summary = False
        self.all_keys_encountered = set()
        self.value_samples = {}
        self.logger.info(f"Initialized with log_level: {log_level}, log_values: {log_values}, log_sample_interval: {log_sample_interval}")

    def startup(self):
        """Startup operations
        """
        self.logger.info(f"Starting up LogAllKeys worker")

    def run(self, item):
        """Process an item by logging its keys
        
        Args:
            item: Item to process
        """
        if item is None:
            self.logger.warning("Received None item")
            return
            
        # Update statistics
        self.item_count += 1
        
        # Add keys to our master set of all keys encountered
        for key in item.keys():
            self.all_keys_encountered.add(key)
            
            # Log max one sample per key
            if self.log_values and key not in self.value_samples:
                value = item[key]
                if hasattr(value, 'shape'): 
                    # For numpy arrays, store shape information instead of contents
                    self.value_samples[key] = f"Array with shape {value.shape}, dtype {value.dtype}"
                elif isinstance(value, (str, int, float, bool)) and len(str(value)) < 100:
                    # For simple types, store value directly
                    self.value_samples[key] = value
                elif isinstance(value, dict):
                    # For dictionaries, store a count of keys
                    self.value_samples[key] = f"Dict with {len(value)} keys: {list(value.keys())[:5]}..."
                elif isinstance(value, list):
                    # For lists, store length and sample
                    sample = str(value[:3])[:100] + "..." if len(value) > 3 else str(value)
                    self.value_samples[key] = f"List with {len(value)} items: {sample}"
                else:
                    # For other types, just note the type
                    self.value_samples[key] = f"Type: {type(value).__name__}"
        
        # Only log details periodically to avoid overwhelming logs
        if self.item_count % self.log_sample_interval == 0:
            keys_list = ", ".join(sorted(item.keys()))
            self.logger.log(self.log_level, f"Item {self.item_count} contains {len(item)} keys: {keys_list}")
            
            # If logging values, provide sample values for this item
            if self.log_values:
                for key, value in sorted(item.items()):
                    if hasattr(value, 'shape'):  # Numpy arrays
                        self.logger.log(self.log_level, f"  Key: {key} = Array shape: {value.shape}, dtype: {value.dtype}")
                    elif isinstance(value, (str, int, float, bool)) and len(str(value)) < 100:
                        self.logger.log(self.log_level, f"  Key: {key} = {value}")
                    elif isinstance(value, dict):
                        self.logger.log(self.log_level, f"  Key: {key} = Dict with {len(value)} keys")
                    elif isinstance(value, list):
                        self.logger.log(self.log_level, f"  Key: {key} = List with {len(value)} items")
                    else:
                        self.logger.log(self.log_level, f"  Key: {key} = Type: {type(value).__name__}")
        
        # Pass the item to the next worker(s) without modification
        self.done_with_item(item)

    def shutdown(self):
        """Output summary information during shutdown
        """
        # Log a summary of all keys encountered
        self.logger.info(f"LogAllKeys worker processed {self.item_count} items")
        self.logger.info(f"All keys encountered ({len(self.all_keys_encountered)}): {sorted(self.all_keys_encountered)}")
        
        # Log sample values for all keys if enabled
        if self.log_values:
            self.logger.info("Sample values for encountered keys:")
            for key, value in sorted(self.value_samples.items()):
                self.logger.info(f"  {key} = {value}")
                
        self.logger.info("LogAllKeys worker shutting down")