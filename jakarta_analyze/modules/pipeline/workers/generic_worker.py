# ============ Base imports ======================
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


class GenericWorker(PipelineWorker):
    """Exemplary class which illustrates how to make a worker
    
    This class serves as a template for implementing pipeline workers.
    """
    def initialize(self, config_param1=None, config_param2=None, **kwargs):
        """Initialize worker-specific parameters
        
        Args:
            config_param1: Example config parameter 1
            config_param2: Example config parameter 2
        """
        self.config_param1 = config_param1
        self.config_param2 = config_param2
        self.logger.info(f"Initialized with params: {config_param1}, {config_param2}")

    def startup(self):
        """Startup operations
        
        This method gets called for each worker PROCESS after it has been spawned.
        Use it to set environment variables or set up connections to the GPU, etc.
        """
        self.logger.info("Generic worker startup")
        # Set environment variables, GPU connections, etc. here

    def run(self, item=None, *args, **kwargs):
        """Process an item
        
        Args:
            item: Item to process
        """
        if item is None:
            self.logger.info("No item provided (this is a source worker)")
            return
            
        self.logger.info(f"Processing item: {item.get('frame_number', 'unknown')}")
        self.logger.debug("This statement will only show in the terminal when running in debug mode")
        
        # Example: Add a field to the item
        item["example_field"] = "example_value"
        
        # Pass the item to the next worker(s)
        self.done_with_item(item)

    def shutdown(self):
        """Shutdown operations
        
        This method gets called after the queue which this worker monitors has been shut down
        """
        self.logger.info("Generic worker shutdown")
    
    #==============================
    #= Support Functions/Classes ==
    #==============================
    
    def example_instance_method(self, arg1):
        """Example for an additional helper function with access to instance
        
        This method has a reference to the calling class (self)
        
        Args:
            arg1: Example argument
        """
        self.logger.debug(f"Called example_instance_method with {arg1}")
        return arg1

    @staticmethod
    def example_static_method(arg1):
        """Example for an additional helper function without access to instance
        
        This method does not have a reference to the calling class
        Call this method using GenericWorker.example_static_method
        
        Args:
            arg1: Example argument
        """
        return arg1 * 2