#!/usr/bin/env python
# ============ Base imports ======================
import os
import sys
import shutil
from pathlib import Path
# ====== External package imports ================
import torch
import requests
# ====== Internal package imports ================
# ============== Logging  ========================
import logging
from jakarta_analyze.modules.utils.setup import setup, IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from jakarta_analyze.modules.utils.config_loader import get_config
conf = get_config()
# ================================================

# Default popular Ultralytics models
DEFAULT_MODELS = {
    'yolo3u': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov3u.pt',
    'yolov8n': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
    'yolov8s': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt', 
    'yolov8m': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
    'yolov8l': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt',
    'yolov8x': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt',
    'yolo11m': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11m.pt'
}


def list_available_models():
    """
    List all available models from Ultralytics Hub that can be downloaded.
    
    Returns:
        dict: Dictionary of model name to URL
    """
    try:
        # Use the built-in models list for now
        # This could be expanded to fetch from the API dynamically
        return DEFAULT_MODELS
    except Exception as e:
        logger.error(f"Failed to retrieve available models: {str(e)}")
        return DEFAULT_MODELS


def download_model(model_name, output_dir=None, force=False):
    """
    Download a model from Ultralytics Hub.
    
    Args:
        model_name (str): Name of the model to download (e.g., 'yolov8n')
        output_dir (str, optional): Directory to save the model to. If None, uses the 
                                   default models directory from config.
        force (bool): Whether to force download even if the model already exists.
        
    Returns:
        str: Path to the downloaded model file
    """
    # Get models directory from config or use the provided output_dir
    if output_dir is None:
        models_dir = conf.get('dirs', {}).get('models', 'models')
        # Create absolute path
        if not os.path.isabs(models_dir):
            # If it's a relative path, make it absolute from the project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            models_dir = os.path.join(project_root, models_dir)
    else:
        models_dir = output_dir
        
    # Create directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Check if model is available
    available_models = list_available_models()
    if model_name not in available_models:
        logger.error(f"Model '{model_name}' is not available. Choose from: {', '.join(available_models.keys())}")
        return None
    
    model_url = available_models[model_name]
    model_path = os.path.join(models_dir, f"{model_name}.pt")
    
    # Check if model already exists
    if os.path.exists(model_path) and not force:
        logger.info(f"Model '{model_name}' already exists at {model_path}")
        return model_path
    
    # Download the model
    try:
        logger.info(f"Downloading model '{model_name}' from {model_url}")
        
        # Use requests to download the file with progress reporting
        with requests.get(model_url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            # Create a temporary file for downloading
            temp_file = f"{model_path}.download"
            with open(temp_file, 'wb') as f:
                downloaded = 0
                last_log_percent = -1
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Log progress every 10%
                        if total_size > 0:
                            percent = int(downloaded * 100 / total_size)
                            if percent // 10 > last_log_percent // 10:
                                logger.info(f"Download progress: {percent}% ({downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB)")
                                last_log_percent = percent
            
            # Rename the temporary file to the final filename
            shutil.move(temp_file, model_path)
        
        logger.info(f"Model '{model_name}' downloaded successfully to {model_path}")
        return model_path
    
    except Exception as e:
        logger.error(f"Failed to download model '{model_name}': {str(e)}")
        
        # Clean up partial downloads
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        return None


def download_from_ultralytics_hub(model_id, api_key=None, output_dir=None):
    """
    Download a model from Ultralytics HUB using the API.
    
    Args:
        model_id (str): ID of the model in Ultralytics HUB
        api_key (str, optional): Ultralytics API key. If None, will look for ULTRALYTICS_API_KEY env var.
        output_dir (str, optional): Directory to save the model to.
        
    Returns:
        str: Path to the downloaded model
    """
    try:
        # Import Ultralytics
        import ultralytics
        from ultralytics import hub
        
        # Set API key if provided, otherwise it will use the ULTRALYTICS_API_KEY env var
        if api_key:
            os.environ["ULTRALYTICS_API_KEY"] = api_key
            
        # Ensure output directory exists
        if output_dir is None:
            models_dir = conf.get('dirs', {}).get('models', 'models')
            # Create absolute path
            if not os.path.isabs(models_dir):
                # If it's a relative path, make it absolute from the project root
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                output_dir = os.path.join(project_root, models_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Download the model
        logger.info(f"Downloading model {model_id} from Ultralytics HUB")
        model_path = hub.download(model=model_id, path=output_dir)
        
        logger.info(f"Model downloaded successfully to {model_path}")
        return model_path
        
    except ImportError:
        logger.error("Ultralytics package not installed. Install with: pip install ultralytics")
        return None
    except Exception as e:
        logger.error(f"Failed to download model from Ultralytics HUB: {str(e)}")
        return None


if __name__ == "__main__":
    setup("model_downloader")
    
    # Example usage
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        download_model(model_name)
    else:
        print("Available models:")
        for name in list_available_models():
            print(f"- {name}")
        print("\nUsage: python -m jakarta_analyze.modules.models.downloader <model_name>")