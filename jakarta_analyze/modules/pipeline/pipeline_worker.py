# ============ Base imports ======================
import os
import time
import traceback
import multiprocessing as mp
# ====== External package imports ================
# ====== Internal package imports ================
# ============== Logging  ========================
import logging
from jakarta_analyze.modules.utils.setup import IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from jakarta_analyze.modules.utils.config_loader import get_config
conf = get_config()
# ================================================


class PipelineWorker:
    """Abstract base class for all pipeline workers
    
    This class defines the interface and common functionality for pipeline workers.
    All worker classes should inherit from this class and implement the required methods.
    """
    
    def __init__(self, input_queue=None, output_queues=None, pipeline_config=None, 
                start_time=None, model_number=None, out_path=None, **kwargs):
        """Initialize the pipeline worker
        
        Args:
            input_queue: Queue from which to read items
            output_queues: List of queues to which to write items
            pipeline_config: Complete pipeline configuration
            start_time: Pipeline start time
            model_number: Model identifier
            out_path: Path for output files
            **kwargs: Additional keyword arguments specific to the worker
        """
        self.input_queue = input_queue
        self.output_queues = output_queues if output_queues is not None else []
        self.pipeline_config = pipeline_config if pipeline_config is not None else {}
        self.start_time = start_time if start_time is not None else time.time()
        self.model_number = model_number if model_number is not None else 'unknown'
        self.out_path = out_path if out_path is not None else 'output'
        self.logger = logger
        
        # Call worker-specific initialization
        try:
            self.initialize(**kwargs)
        except Exception as e:
            self.logger.exception(f"Error during worker initialization: {str(e)}")
            raise
    
    def initialize(self, **kwargs):
        """Initialize worker-specific parameters
        
        This method should be overridden by subclasses to initialize
        worker-specific parameters from kwargs.
        
        Args:
            **kwargs: Worker-specific keyword arguments
        """
        raise NotImplementedError("Subclasses must implement initialize()")
    
    def startup(self):
        """Startup operations
        
        This method is called once when the worker process starts.
        Override this method to perform any startup operations such as
        connecting to resources or loading models.
        """
        pass
    
    def run(self, item):
        """Process a single item
        
        This method should be overridden by subclasses to process a single item
        from the input queue. If this worker is a source worker (no input queue),
        this method may be called with item=None.
        
        Args:
            item: Item to process, may be None for source workers
        """
        raise NotImplementedError("Subclasses must implement run()")
        
    def shutdown(self):
        """Shutdown operations
        
        This method is called once when the worker process is shutting down.
        Override this method to perform any cleanup operations such as
        closing connections or saving state.
        """
        pass
    
    def done_with_item(self, item):
        """Send an item to all output queues
        
        Args:
            item: Item to send to output queues
        """
        for queue in self.output_queues:
            queue.put(item)
    
    def _run(self):
        """Main worker loop
        
        This method is the main loop that reads from the input queue,
        processes items, and writes to the output queues. It also handles
        exceptions and special commands like 'STOP'.
        """
        try:
            # Call startup method
            self.startup()
            
            # If there's no input queue, this is a source worker
            if self.input_queue is None:
                # Run until an exception occurs or the process is terminated
                while True:
                    try:
                        self.run(None)
                    except Exception as e:
                        self.logger.exception(f"Error in source worker: {str(e)}")
                        # Continue despite the error
                        time.sleep(1)  # Avoid tight loop in case of persistent errors
            else:
                # Process items from the input queue
                while True:
                    try:
                        item = self.input_queue.get()
                        
                        # Check for stop signal
                        if item == 'STOP':
                            # Forward stop signal to output queues
                            for queue in self.output_queues:
                                queue.put('STOP')
                            break
                        
                        # Process item
                        self.run(item)
                        
                    except Exception as e:
                        self.logger.exception(f"Error processing item: {str(e)}")
                        # Continue despite the error
        except Exception as e:
            self.logger.exception(f"Fatal error in worker: {str(e)}")
        finally:
            # Call shutdown method
            try:
                self.shutdown()
            except Exception as e:
                self.logger.exception(f"Error during worker shutdown: {str(e)}")


def run_with_exception_handling(func, *args, **kwargs):
    """Run a function with exception handling
    
    Args:
        func: Function to run
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Result of the function call or None if an exception occurred
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.exception(f"Error running {func.__name__}: {str(e)}")
        return None


def catch_all(func):
    """Decorator to catch all exceptions in a function
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Error in {func.__name__}: {str(e)}")
            return None
    return wrapper