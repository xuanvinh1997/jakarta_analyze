# ============ Base imports ======================
# ====== External package imports ================
# ====== Internal package imports ================
# Import worker classes
try:
    from jakarta_analyze.modules.pipeline.workers.read_frames_from_vid_file import ReadFramesFromVidFile
except ImportError:
    ReadFramesFromVidFile = None
    
try:
    from jakarta_analyze.modules.pipeline.workers.read_frames_from_vid_files_in_dir import ReadFramesFromVidFilesInDir
except ImportError:
    ReadFramesFromVidFilesInDir = None
    
try:
    from jakarta_analyze.modules.pipeline.workers.write_frames_to_vid_files import WriteFramesToVidFiles
except ImportError:
    WriteFramesToVidFiles = None
    
try:
    from jakarta_analyze.modules.pipeline.workers.yolo3_detect import Yolo3Detect
except ImportError:
    Yolo3Detect = None
    
try:
    from jakarta_analyze.modules.pipeline.workers.lk_sparse_optical_flow import LKSparseOpticalFlow
except ImportError:
    LKSparseOpticalFlow = None
    
try:
    from jakarta_analyze.modules.pipeline.workers.mean_motion_direction import MeanMotionDirection
except ImportError:
    MeanMotionDirection = None
    
try:
    from jakarta_analyze.modules.pipeline.workers.compute_frame_stats import ComputeFrameStats
except ImportError:
    ComputeFrameStats = None
    
try:
    from jakarta_analyze.modules.pipeline.workers.write_keys_to_files import WriteKeysToFiles
except ImportError:
    WriteKeysToFiles = None
    
try:
    from jakarta_analyze.modules.pipeline.workers.write_keys_to_database_table import WriteKeysToDatabaseTable
except ImportError:
    WriteKeysToDatabaseTable = None
    
try:
    from jakarta_analyze.modules.pipeline.workers.generic_worker import GenericWorker
except ImportError:
    GenericWorker = None
    
# ============== Logging  ========================
import logging
from jakarta_analyze.modules.utils.setup import IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# ================================================


# Register all workers in a dictionary for easy access
WORKER_REGISTRY = {
    'ReadFramesFromVidFile': ReadFramesFromVidFile,
    'ReadFramesFromVidFilesInDir': ReadFramesFromVidFilesInDir,
    'WriteFramesToVidFiles': WriteFramesToVidFiles,
    'Yolo3Detect': Yolo3Detect,
    'LKSparseOpticalFlow': LKSparseOpticalFlow,
    'MeanMotionDirection': MeanMotionDirection,
    'ComputeFrameStats': ComputeFrameStats,
    'WriteKeysToFiles': WriteKeysToFiles,
    'WriteKeysToDatabaseTable': WriteKeysToDatabaseTable,
    'GenericWorker': GenericWorker,
}

# Filter out None values (failed imports)
WORKER_REGISTRY = {k: v for k, v in WORKER_REGISTRY.items() if v is not None}


def get_worker_class(worker_type):
    """Get a worker class by name
    
    Args:
        worker_type (str): Worker class name
        
    Returns:
        class: Worker class or None if not found
    """
    if worker_type in WORKER_REGISTRY:
        return WORKER_REGISTRY[worker_type]
    else:
        logger.error(f"Worker type not found: {worker_type}")
        return None
        
        
def list_available_workers():
    """List all available worker classes
    
    Returns:
        list: List of available worker class names
    """
    return list(WORKER_REGISTRY.keys())


def print_available_workers():
    """Print all available worker classes to console
    """
    print("Available pipeline workers:")
    for name in list_available_workers():
        print(f"  - {name}")


if __name__ == "__main__":
    print_available_workers()