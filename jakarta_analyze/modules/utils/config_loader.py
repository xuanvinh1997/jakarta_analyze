# ============ Base imports ======================
import os
import sys
import json
import yaml
# ====== External package imports ================
# ====== Internal package imports ================
# ============== Logging  ========================
import logging
logger = logging.getLogger("")
# ================================================

# Global configuration object
_config = None

def get_config():
    """Get the configuration object
    
    Returns a singleton configuration object, loading it first if needed.
    
    Returns:
        dict: Configuration object
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config

def load_config(config_path=None):
    """Load configuration from file
    
    Args:
        config_path (str): Path to configuration file, if None uses JAKARTA_CONFIG_PATH env var
                         or default location in config_examples/config.yml
    
    Returns:
        dict: Configuration object
    """
    if config_path is None:
        # Check environment variable
        config_path = os.environ.get("JAKARTA_CONFIG_PATH")
        
        # If not set, use default
        if config_path is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            config_path = os.path.join(project_root, "config_examples/config.yml")
            
            # If default doesn't exist, try other locations
            if not os.path.exists(config_path):
                alt_paths = [
                    os.path.join(project_root, "config.yml"),
                    os.path.join(os.path.dirname(__file__), "../../../config.yml")
                ]
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        config_path = alt_path
                        break
    
    # Check if config file exists
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        logger.error("Set JAKARTA_CONFIG_PATH environment variable to specify config file location")
        return {}
        
    # Load config file based on extension
    try:
        with open(config_path, 'r') as f:
            if config_path.endswith(".json"):
                config = json.load(f)
            elif config_path.endswith((".yml", ".yaml")):
                config = yaml.safe_load(f)
            else:
                logger.error(f"Unsupported config file format: {config_path}")
                return {}
                
        logger.info(f"Loaded configuration from {config_path}")
        
        # Also load credentials if path exists
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        creds_path = os.path.join(project_root, "config_examples/creds.yml")
        if os.path.exists(creds_path):
            with open(creds_path, 'r') as f:
                creds = yaml.safe_load(f)
                # Merge credentials with main config
                config.update(creds)
            logger.info(f"Loaded credentials from {creds_path}")
                
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return {}


def set_config(config):
    """Set the global configuration object
    
    Args:
        config (dict): Configuration object
    """
    global _config
    _config = config


def main():
    config = get_config()
    print(config)

if __name__ == "__main__":
    main()