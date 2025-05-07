#!/usr/bin/env python
# ============ Base imports ======================
import traceback
import sys
# ====== External package imports ================
# ====== Internal package imports ================
# ============== Logging  ========================
# =========== Config File Loading ================
from jakarta_analyze.modules.utils.config_loader import get_config
conf = get_config()
# ================================================


def run_and_catch_exceptions(logger, func, *args, **kwargs):
    """Run the function and catch exceptions
    
    Execute the function with provided arguments and catch any exceptions 
    that occur, logging them appropriately.
    
    Args:
        logger: Logger object to log errors
        func: Function to execute
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Result from the function if successful, None otherwise
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Exception in {func.__name__}: {str(e)}")
        logger.error(traceback.format_exc())
        return None