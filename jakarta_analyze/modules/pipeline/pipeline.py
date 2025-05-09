# ============ Base imports ======================
import os
import time
import json
import yaml
import importlib
import multiprocessing as mp
from typing import Dict, List, Any
# ====== External package imports ================
# ====== Internal package imports ================
from jakarta_analyze.modules.pipeline.pipeline_worker import PipelineWorker, run_with_exception_handling
# ============== Logging  ========================
import logging
from jakarta_analyze.modules.utils.setup import IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from jakarta_analyze.modules.utils.config_loader import get_config
conf = get_config()
# ================================================


class Pipeline:
    """Main pipeline class that coordinates worker processes
    
    This class is responsible for setting up and managing the pipeline workers
    based on configuration. It handles the creation of worker processes, the
    interconnecting queues, and the pipeline lifecycle.
    """
    
    def __init__(self, config_file=None, model_number=None, out_path=None):
        """Initialize the pipeline with configuration
        
        Args:
            config_file (str): Path to pipeline configuration file
            model_number (str): Model identifier
            out_path (str): Path for output files
        """
        self.config = self._load_config(config_file)
        self.start_time = time.time()
        self.model_number = model_number if model_number is not None else 'default'
        self.out_path = out_path if out_path is not None else os.path.join('output', f"pipeline_{self.model_number}")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.out_path, exist_ok=True)
        
        # Initialize pipeline structures
        self.workers = {}
        self.processes = {}
        self.queues = {}
        
        logger.info(f"Pipeline initialized with model number {self.model_number}")
        logger.info(f"Output path: {self.out_path}")
        
    def _load_config(self, config_file):
        """Load pipeline configuration from file
        
        Args:
            config_file (str): Path to pipeline configuration file
            
        Returns:
            dict: Loaded configuration
        """
        if config_file is None:
            # Check environment variable
            config_file = os.environ.get("JAKARTA_PIPELINE_CONFIG_PATH")
            
            # If not set, use default
            if config_file is None:
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
                config_file = os.path.join(project_root, "config_examples/pipeline.yml")
        
        # Check if config file exists
        if not os.path.exists(config_file):
            logger.error(f"Pipeline configuration file not found: {config_file}")
            logger.error("Set JAKARTA_PIPELINE_CONFIG_PATH environment variable to specify pipeline config file location")
            return {}
            
        # Load config file based on extension
        try:
            with open(config_file, 'r') as f:
                if config_file.endswith(".json"):
                    config = json.load(f)
                elif config_file.endswith((".yml", ".yaml")):
                    config = yaml.safe_load(f)
                else:
                    logger.error(f"Unsupported config file format: {config_file}")
                    return {}
            
            # Adapt the configuration structure if needed
            # If config has a 'pipeline' key with 'tasks', transform to expected format
            if 'pipeline' in config and 'tasks' in config['pipeline']:
                # Convert tasks to workers
                workers = []
                for task in config['pipeline']['tasks']:
                    # Create a worker config from each task
                    worker_config = task.copy()
                    
                    # Map worker_type to type if it exists
                    if 'worker_type' in worker_config:
                        worker_config['type'] = worker_config.pop('worker_type')
                    
                    # Map task fields to worker fields if needed
                    if 'prev_task' in worker_config:
                        # Create a list of next workers for each previous task
                        prev_task = worker_config.pop('prev_task')
                        if prev_task:  # If not None or empty
                            worker_config['source'] = False
                            # We'll handle connections later
                        else:
                            worker_config['source'] = True
                    
                    # Add worker to list
                    workers.append(worker_config)
                
                # Now process the workers to set up next connections
                for i, worker in enumerate(workers):
                    if not worker.get('source', False):
                        # Find the previous worker by name
                        prev_name = config['pipeline']['tasks'][i]['prev_task']
                        for prev_worker in workers:
                            if prev_worker['name'] == prev_name:
                                if 'next' not in prev_worker:
                                    prev_worker['next'] = []
                                prev_worker['next'].append(worker['name'])
                
                # Create new config structure
                config = {
                    'workers': workers,
                    'name': config['pipeline'].get('name', 'default_pipeline'),
                    'options': config['pipeline'].get('options', {})
                }
                    
            logger.info(f"Loaded pipeline configuration from {config_file}")
            return config
        except Exception as e:
            logger.exception(f"Error loading pipeline configuration: {str(e)}")
            return {}
    
    def setup(self):
        """Set up the pipeline based on configuration
        
        Creates worker instances, queues, and prepares the pipeline for execution.
        
        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            # Define worker class registry mapping worker types to their module paths
            worker_registry = {
                'ReadFramesFromVidFilesInDir': 'jakarta_analyze.modules.pipeline.workers.read_frames_from_vid_files_in_dir.ReadFramesFromVidFilesInDir',
                'Yolo3Detect': 'jakarta_analyze.modules.pipeline.workers.yolo3_detect.Yolo3Detect',
                'LKSparseOpticalFlow': 'jakarta_analyze.modules.pipeline.workers.lk_sparse_optical_flow.LKSparseOpticalFlow',
                'MeanMotionDirection': 'jakarta_analyze.modules.pipeline.workers.mean_motion_direction.MeanMotionDirection',
                # 'WriteKeysToDatabaseTable': 'jakarta_analyze.modules.pipeline.workers.write_keys_to_database_table.WriteKeysToDatabaseTable',
                'WriteFramesToVidFiles': 'jakarta_analyze.modules.pipeline.workers.write_frames_to_vid_files.WriteFramesToVidFiles',
                'ComputeFrameStats': 'jakarta_analyze.modules.pipeline.workers.compute_frame_stats.ComputeFrameStats',
                'WriteKeysToFiles': 'jakarta_analyze.modules.pipeline.workers.write_keys_to_files.WriteKeysToFiles',
                'ReadFramesFromVid': 'jakarta_analyze.modules.pipeline.workers.read_frames_from_vid.ReadFramesFromVid',
                'ReadFramesFromVidFile': 'jakarta_analyze.modules.pipeline.workers.read_frames_from_vid_file.ReadFramesFromVidFile',
            }
            
            workers_config = self.config.get('workers', [])
            
            # First pass: create all queues
            for i, worker_config in enumerate(workers_config):
                worker_name = worker_config.get('name', f"worker_{i}")
                
                # Create input queue for this worker
                input_queue_name = f"q_in_{worker_name}"
                if worker_config.get('source', False):
                    # Source workers don't have an input queue
                    input_queue = None
                else:
                    # Create input queue with specified or default size
                    queue_size = worker_config.get('queue_size', 100)
                    input_queue = mp.Queue(maxsize=queue_size)
                    self.queues[input_queue_name] = input_queue
                
                # Create output queue for this worker
                output_queue_name = f"q_out_{worker_name}"
                queue_size = worker_config.get('queue_size', 100)
                output_queue = mp.Queue(maxsize=queue_size)
                self.queues[output_queue_name] = output_queue
                
            # Second pass: create worker instances and connect queues
            for i, worker_config in enumerate(workers_config):
                worker_name = worker_config.get('name', f"worker_{i}")
                
                # Get worker class
                worker_type = worker_config.get('type')
                if not worker_type:
                    logger.error(f"Worker type not specified for worker {worker_name}")
                    continue
                    
                # Try to get worker class from registry
                if worker_type in worker_registry:
                    module_path, class_name = worker_registry[worker_type].rsplit('.', 1)
                    try:
                        module = importlib.import_module(module_path)
                        worker_class = getattr(module, class_name)
                    except (ImportError, AttributeError) as e:
                        logger.error(f"Error importing worker class {worker_type} from registry: {str(e)}")
                        continue
                else:
                    # Fall back to dynamic import
                    try:
                        # Check if this is a fully qualified path
                        if '.' in worker_type:
                            module_path, class_name = worker_type.rsplit('.', 1)
                            module = importlib.import_module(module_path)
                            worker_class = getattr(module, class_name)
                        else:
                            logger.error(f"Worker type {worker_type} not found in registry and not a fully qualified path")
                            continue
                    except (ImportError, AttributeError) as e:
                        logger.error(f"Error importing worker class {worker_type}: {str(e)}")
                        continue
                
                # Get input and output queues
                input_queue_name = f"q_in_{worker_name}"
                input_queue = None if worker_config.get('source', False) else self.queues.get(input_queue_name)
                
                # Connect output queues based on configuration
                output_queues = []
                for next_worker in worker_config.get('next', []):
                    next_queue_name = f"q_in_{next_worker}"
                    if next_queue_name in self.queues:
                        output_queues.append(self.queues[next_queue_name])
                    else:
                        logger.warning(f"Output queue {next_queue_name} for worker {worker_name} not found")
                
                # Create worker instance with all required parameters
                # Copy all parameters from worker_config except meta parameters
                worker_kwargs = {k: v for k, v in worker_config.items() if k not in 
                                ['type', 'name', 'source', 'next', 'queue_size', 'prev_task']}
                
                # Add default parameters for specific worker types
                if worker_type == 'Yolo3Detect':
                    # Default parameters for Yolo3Detect if not provided
                    defaults = {
                        'annotation_font_scale': 0.75,
                        'annotate_frame_key': worker_kwargs.get('frame_key', 'frame'),
                        'config_path': os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'config.yml')
                    }
                    # Only add defaults if keys don't exist
                    for key, value in defaults.items():
                        if key not in worker_kwargs:
                            worker_kwargs[key] = value
                
                elif worker_type == 'LKSparseOpticalFlow':
                    # Default parameters for LKSparseOpticalFlow if not provided
                    defaults = {
                        'path_track_length': 10,  # Default track length
                        'new_point_detect_interval': 5 if worker_kwargs.get('new_point_detect_interval_per_second') is None else None,
                    }
                    # Only add defaults if keys don't exist
                    for key, value in defaults.items():
                        if key not in worker_kwargs:
                            worker_kwargs[key] = value
                
                elif worker_type == 'MeanMotionDirection':
                    # Default parameters for MeanMotionDirection if not provided
                    defaults = {
                        'points_key': 'tracked_points',
                        'flows_key': 'tracked_flows',
                        'boxes_key': 'boxes'
                    }
                    # Only add defaults if keys don't exist
                    for key, value in defaults.items():
                        if key not in worker_kwargs:
                            worker_kwargs[key] = value
                
                # Add other worker type defaults as needed
                
                worker_instance = worker_class(
                    name=worker_name,
                    input_queue=input_queue,
                    output_queues=output_queues,
                    pipeline_config=self.config,
                    start_time=self.start_time,
                    model_number=self.model_number,
                    out_path=self.out_path,
                    **worker_kwargs
                )
                
                # Store worker instance
                self.workers[worker_name] = worker_instance
                logger.info(f"Created worker: {worker_name} ({worker_type})")
            
            logger.info(f"Pipeline setup complete with {len(self.workers)} workers")
            return True
        except Exception as e:
            logger.exception(f"Error setting up pipeline: {str(e)}")
            return False
    
    def start(self):
        """Start the pipeline
        
        Creates and starts worker processes for all configured workers.
        
        Returns:
            bool: True if startup successful, False otherwise
        """
        try:
            # Start processes for all workers
            for worker_name, worker in self.workers.items():
                process = mp.Process(target=worker._run, name=worker_name)
                process.daemon = True
                process.start()
                self.processes[worker_name] = process
                logger.info(f"Started worker process: {worker_name} (PID: {process.pid})")
            
            logger.info(f"Pipeline started with {len(self.processes)} processes")
            return True
        except Exception as e:
            logger.exception(f"Error starting pipeline: {str(e)}")
            return False
    
    def stop(self):
        """Stop the pipeline
        
        Sends stop signals to all workers and waits for processes to terminate.
        
        Returns:
            bool: True if shutdown successful, False otherwise
        """
        try:
            # Send stop signal to source workers
            for worker_name, worker in self.workers.items():
                if worker.input_queue is None:  # This is a source worker
                    # Find its output queue and send stop signal
                    out_queue_name = f"q_out_{worker_name}"
                    for next_worker in self.config.get('workers', []):
                        if next_worker.get('name') == worker_name:
                            for next_name in next_worker.get('next', []):
                                queue_name = f"q_in_{next_name}"
                                if queue_name in self.queues:
                                    self.queues[queue_name].put('STOP')
                                    logger.info(f"Sent STOP signal to {worker_name} output queue {queue_name}")
            
            # Wait for processes to terminate
            timeout = 30  # seconds
            start_time = time.time()
            while time.time() - start_time < timeout:
                # Check if any processes are still alive
                alive_processes = [name for name, process in self.processes.items() if process.is_alive()]
                if not alive_processes:
                    break
                    
                logger.info(f"Waiting for processes to terminate: {', '.join(alive_processes)}")
                time.sleep(1)
            
            # Force terminate any remaining processes
            for name, process in self.processes.items():
                if process.is_alive():
                    logger.warning(f"Forcibly terminating process: {name}")
                    process.terminate()
                    
            logger.info("Pipeline stopped")
            return True
        except Exception as e:
            logger.exception(f"Error stopping pipeline: {str(e)}")
            return False
    
    def run(self):
        """Run the pipeline until it completes or is interrupted
        
        Sets up the pipeline, starts it, and monitors it until completion.
        
        Returns:
            bool: True if run successful, False otherwise
        """
        try:
            # Setup the pipeline
            if not self.setup():
                logger.error("Pipeline setup failed")
                return False
                
            # Start the pipeline
            if not self.start():
                logger.error("Pipeline start failed")
                return False
                
            # Monitor the pipeline
            try:
                while True:
                    # Check if any processes have terminated unexpectedly
                    for name, process in self.processes.items():
                        if not process.is_alive():
                            logger.error(f"Process {name} terminated unexpectedly")
                            # Stop the pipeline if a process dies
                            self.stop()
                            return False
                            
                    # Check if all source workers have completed
                    source_workers_done = True
                    for worker_name, worker in self.workers.items():
                        if worker.input_queue is None:  # This is a source worker
                            process = self.processes[worker_name]
                            if process.is_alive():
                                source_workers_done = False
                                break
                                
                    if source_workers_done:
                        logger.info("All source workers completed, stopping pipeline")
                        self.stop()
                        break
                        
                    # Sleep briefly to avoid tight loop
                    time.sleep(1)
                    
                return True
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, stopping pipeline")
                self.stop()
                return False
        except Exception as e:
            logger.exception(f"Error running pipeline: {str(e)}")
            self.stop()  # Try to stop cleanly even after error
            return False
            
    def __enter__(self):
        """Context manager entry
        
        Returns:
            Pipeline: Self
        """
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit
        
        Ensures pipeline is stopped when exiting context.
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        self.stop()
        
        
def main():
    """Run a pipeline from command line
    """
    import argparse
    parser = argparse.ArgumentParser(description="Run a Jakarta Analyze pipeline")
    parser.add_argument("--config", help="Path to pipeline configuration file")
    parser.add_argument("--model", help="Model number/identifier", default="default")
    parser.add_argument("--output", help="Output directory", default=None)
    args = parser.parse_args()
    
    # Set up logging
    from jakarta_analyze.modules.utils.setup import setup_logging
    setup_logging()
    
    # Create and run pipeline
    pipeline = Pipeline(config_file=args.config, model_number=args.model, out_path=args.output)
    success = pipeline.run()
    
    return 0 if success else 1
    

if __name__ == "__main__":
    exit(main())