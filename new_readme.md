# Jakarta Traffic Analysis Tools

A comprehensive toolkit for analyzing traffic patterns in Jakarta using computer vision and machine learning techniques.

## Features

- **Video Processing Pipeline**: Analyze traffic videos with object detection and motion tracking
- **Metadata Extraction**: Extract and analyze metadata from video files
- **Visualization Tools**: Create visualizations of video analysis results
- **Model Management**: Download and update AI models from Ultralytics Hub
- **Database Integration**: Store and query analysis results in MongoDB

## Installation

### Prerequisites

- Python 3.12.3 (tested)
- MongoDB database
- FFmpeg (for video processing)
- CUDA-compatible GPU (recommended for object detection)

### Setup

1. Clone the repository
   ```bash
   git clone https://github.com/your-username/jakarta_analyze.git
   cd jakarta_analyze
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. Download the YOLO model
   ```bash
   # Using wget
   wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt -P models
   
   # OR using the built-in download command
   python -m jakarta_analyze download-model yolo11m
   ```

## Quick Start

### Download and Prepare Videos
Download traffic videos from the Jakarta dataset and prepare them for analysis:

```bash
# Download videos
python -m jakarta_analyze download-videos 
```

### Extract Video Metadata

Extract metadata such as frame statistics, packet information, and subtitles:

```bash
python -m jakarta_analyze extract-metadata -v /path/to/videos
```

### Visualize Video Metadata

Generate visualizations from the extracted metadata:

```bash
python -m jakarta_analyze visualize-metadata --format png
```

### Run the Analysis Pipeline

Process videos with object detection and motion tracking:

```bash
python -m jakarta_analyze pipeline -c pipeline.yml
```

## Configuration Files

The toolkit uses several configuration files:

### pipeline.yml

Defines the video analysis pipeline with object detection and motion tracking:

```yaml
pipeline:
  name: test_pipeline
  options:
    queue_monitor_delay_seconds: 10
    queue_monitor_meter_size: 10
  tasks:
    - worker_type: ReadFramesFromVidFilesInDir
      # ...configuration for reading video frames
    - worker_type: Yolo3Detect
      # ...configuration for object detection
    - worker_type: LKSparseOpticalFlow
      # ...configuration for motion tracking
    - worker_type: MeanMotionDirection
      # ...configuration for motion analysis
    - worker_type: WriteKeysToDatabaseTable
      # ...configuration for database writing
    - worker_type: WriteFramesToVidFiles
      # ...configuration for video output
```

### config.yml

Contains general configuration settings for the toolkit:

```yaml
visualization:
  path_track_length: 10
  annotation_font_scale: 1.0
  vid_containers: [".mkv"]
  output_extension: ".pdf"

validation:
  iou_threshold: 0.25
  labels_we_test: ['pedestrian','bicycle','car','motorbike','bus','train','truck']
  confidence_threshold: 0.1

dirs:
  raw_videos: downloaded_videos/
  frame_stats: outputs/frame_stats/
  packet_stats: outputs/packet_stats/
  subtitles: outputs/subtitles/
  visualizations: outputs/visualizations/
  output_videos: outputs/videos/
  models: models/
```

### database.yml

Database connection settings:

```yaml
database:
  mongo_uri: "mongodb://localhost:27017"
  dbname: jakarta_traffic
  timeout: 30000  # Timeout in milliseconds
  collections:
    videos: videos
    boxes: boxes
    motion: box_motion
```

## Command Line Interface

The toolkit provides several commands:

### download-model

Download AI models from Ultralytics Hub:

```bash
# List available models
python -m jakarta_analyze download-model --list

# Download a specific model
python -m jakarta_analyze download-model yolo11m

# Force re-download of a model
python -m jakarta_analyze download-model yolo11m --force

# Download to a specific directory
python -m jakarta_analyze download-model yolo11m -o /path/to/models
```

### extract-metadata

Extract metadata from video files:

```bash
# Extract from default video directory
python -m jakarta_analyze extract-metadata

# Extract from specific directory
python -m jakarta_analyze extract-metadata -v /path/to/videos

# Use a custom config file
python -m jakarta_analyze extract-metadata -c custom_config.yml
```

### visualize-metadata

Generate visualizations from extracted metadata:

```bash
# Generate default visualizations
python -m jakarta_analyze visualize-metadata

# Specify output format
python -m jakarta_analyze visualize-metadata --format png

# Use a custom output directory
python -m jakarta_analyze visualize-metadata -o /path/to/output
```

### pipeline

Run the complete traffic analysis pipeline:

```bash
# Run with default pipeline configuration
python -m jakarta_analyze pipeline -c pipeline.yml

# Specify custom video and output directories
python -m jakarta_analyze pipeline -c pipeline.yml -v /path/to/videos -o /path/to/output
```

## Pipeline Tasks

The analysis pipeline consists of several tasks:

1. **ReadFramesFromVidFilesInDir**: Reads video files from a directory
2. **Yolo3Detect**: Performs object detection using YOLOv8/YOLOv11 models
3. **LKSparseOpticalFlow**: Tracks motion using Lucas-Kanade optical flow
4. **MeanMotionDirection**: Calculates mean motion direction and magnitude
5. **WriteKeysToDatabaseTable**: Writes analysis results to the database
6. **WriteFramesToVidFiles**: Outputs annotated video files

## Database Schema

The toolkit uses MongoDB with the following collections:

### videos

Stores information about processed videos:

- `_id`: MongoDB document ID
- `file_name`: Name of the video file
- `file_path`: Full path to the video file
- `camera_id`: Camera identifier
- `timestamp`: Video capture timestamp
- `duration_seconds`: Video duration
- `frame_count`: Total number of frames
- `fps`: Frames per second
- `height`: Video height in pixels
- `width`: Video width in pixels

### boxes

Stores object detection results:

- `_id`: MongoDB document ID
- `box_id`: Unique identifier for detected object
- `video_id`: Reference to the video document
- `frame_number`: Frame number in the video
- `video_name`: Name of the video file
- `label`: Object class (car, pedestrian, etc.)
- `confidence`: Detection confidence score
- `x_min`, `y_min`, `x_max`, `y_max`: Bounding box coordinates
- `timestamp`: Time of database insertion

### box_motion

Stores motion tracking results:

- `_id`: MongoDB document ID
- `box_id`: Unique identifier for detected object
- `video_id`: Reference to the video document
- `frame_number`: Frame number in the video
- `video_name`: Name of the video file
- `point_count`: Number of tracked points
- `avg_motion_x`, `avg_motion_y`: Average motion vector components
- `motion_magnitude`: Motion magnitude in pixels
- `motion_direction`: Motion direction in degrees
- `timestamp`: Time of database insertion

## Visualization Types

The toolkit produces several types of visualizations:

1. **Frame Type Distribution**: Histogram of I/P/B-frame types
2. **Bitrate Analysis**: Plot of bitrate over time
3. **Frame Size Analysis**: Box plot of frame sizes by frame type
4. **Keyframe Interval Analysis**: Distribution and timeline of keyframe intervals
5. **Summary Report**: Comparative metrics across videos

## Advanced Usage

### Customizing the Pipeline

You can customize the pipeline by editing the `pipeline.yml` file:

1. Adjust object detection thresholds:
   ```yaml
   object_detect_threshold: 0.4
   non_maximal_box_suppression_threshold: 0.4
   ```

2. Modify optical flow parameters:
   ```yaml
   maxCorners: 500
   qualityLevel: 0.3
   minDistance: 7
   ```

3. Configure database writing:
   ```yaml
   columns:
     - video_name
     - frame_number
     - box_id
     # ... other columns
   ```

### Using Different Models

The toolkit supports various YOLO models:

- `yolov8n`: Smallest and fastest model
- `yolov8s`: Small model, good balance
- `yolov8m`: Medium-sized model
- `yolov8l`: Large model, more accurate
- `yolov8x`: Extra large model, most accurate
- `yolo11m`: Latest YOLOv11 medium model

### Using as a Python Package

You can import the package's modules directly in your Python code:

```python
import jakarta_analyze
from jakarta_analyze import get_config, setup, IndentLogger
import logging

# Initialize logging
logger = IndentLogger(logging.getLogger(''), {})
setup("my_script")

# Get configuration
conf = get_config()

# Use the configuration
logger.info(f"Working with configuration: {conf}")
```

### Configuration File Search Path

The toolkit looks for configuration files in the following order:

1. If `JAKARTA_CONFIG_PATH` environment variable is set, it uses that file directly
2. If not set, it checks these locations in order:
   - `config_examples/config.yml` in the project root
   - `config.yml` in the project root
   - If none found, returns an empty configuration

### Converting from Original Project Structure

If you have code that uses the original project structure, update your imports:

From:
```python
from src.modules.utils.setup import setup, IndentLogger
from src.modules.utils.config_loader import get_config
```

To:
```python
from jakarta_analyze.modules.utils.setup import setup, IndentLogger
from jakarta_analyze.modules.utils.config_loader import get_config
```

And remove any references to `JAKARTAPATH` and `PYTHONPATH` environment variables.

## Troubleshooting

### Database Connection Issues

If you encounter database connection issues:

1. Check that MongoDB is running
2. Verify the MongoDB connection URI in `database.yml`
3. Ensure the database has the required collections

### Video Processing Errors

For video processing errors:

1. Check that FFmpeg is installed and in your PATH
2. Verify video file formats are supported
3. Ensure the GPU is properly configured if using CUDA

### Missing Dependencies

If you encounter missing dependencies:

```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms of the LICENSE file included in the repository.

