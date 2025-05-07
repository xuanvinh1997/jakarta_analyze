# ============ Base imports ======================
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
import time
import threading
# ====== External package imports ================
# ====== Internal package imports ================
from jakarta_analyze.modules.utils.config_loader import get_config
# ================================================


class IndentLogger:
    """Logger wrapper that adds indentation to log messages
    
    This class wraps a standard logger and provides indentation functionality
    to make log output more readable, especially for complex hierarchical processes.
    """
    
    def __init__(self, logger, indent_data, indent_level=None):
        """Initialize with logger and indentation data
        
        Args:
            logger: Logger object to wrap
            indent_data: Indentation data dictionary
            indent_level: Optional explicit indentation level
        """
        self.logger = logger
        self.indent_data = indent_data
        self.indent_level = indent_level

    def _get_indent(self):
        """Get appropriate indentation string
        
        Returns:
            str: String of spaces for indentation
        """
        if self.indent_level is not None:
            indent = self.indent_level
        else:
            indent = self.indent_data.get(threading.get_ident(), 0)
        return "  " * indent

    def log(self, level, msg, *args, **kwargs):
        """Log a message with proper indentation
        
        Args:
            level: Log level
            msg: Log message
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        if isinstance(msg, str):
            prefix = self._get_indent()
            msg = prefix + msg
        self.logger.log(level, msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        """Log a debug message
        
        Args:
            msg: Log message
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        self.log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """Log an info message
        
        Args:
            msg: Log message
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        self.log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """Log a warning message
        
        Args:
            msg: Log message
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        self.log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """Log an error message
        
        Args:
            msg: Log message
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        self.log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """Log a critical message
        
        Args:
            msg: Log message
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        self.log(logging.CRITICAL, msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        """Log an exception message
        
        Args:
            msg: Log message
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        kwargs['exc_info'] = 1
        self.log(logging.ERROR, msg, *args, **kwargs)


def setup(command_name=None):
    """Set up the Jakarta Analyze environment
    
    Args:
        command_name (str): Name of the command being executed
        
    Returns:
        logger: Configured logger
    """
    # Set up logging
    logger = setup_logging()
    
    if command_name:
        logger.info(f"Setting up environment for command: {command_name}")
    
    # Additional setup steps can be added here
    
    return logger


def setup_logging(log_path=None, log_level=None, log_to_console=True):
    """Set up logging configuration
    
    Args:
        log_path (str): Path to log file, if None uses config or defaults to logs directory
        log_level (str): Log level (DEBUG, INFO, etc.), if None uses config
        log_to_console (bool): Whether to log to console
        
    Returns:
        logger: Configured root logger
    """
    # Load configuration
    config = get_config()
    logging_config = config.get('logging', {})
    
    # Determine log level
    if log_level is None:
        log_level = logging_config.get('level', 'INFO')
    
    # Convert log level string to constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Add console handler if requested
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Determine log file path
    if log_path is None:
        log_path = logging_config.get('path')
        
        if log_path is None:
            # Default to logs directory
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            log_dir = os.path.join(project_root, "logs")
            
            # Create logs directory if it doesn't exist
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            # Generate log file name with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            log_path = os.path.join(log_dir, f"jakarta_analyze_{timestamp}.log")
    
    # Create log directory if needed
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Add file handler
    if log_path:
        max_bytes = logging_config.get('max_bytes', 10485760)  # Default 10MB
        backup_count = logging_config.get('backup_count', 5)
        
        file_handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Log setup information
    root_logger.info(f"Logging initialized with level {log_level}")
    if log_path:
        root_logger.info(f"Logging to file: {log_path}")
    
    # Create and return indent logger
    indent_data = {}
    return IndentLogger(root_logger, indent_data)


# used for testing this script
def main():
    logger = setup_logging()
    logger.info("Testing logging setup")


if __name__ == "__main__":
    main()