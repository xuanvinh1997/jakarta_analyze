#!/usr/bin/env python
# ============ Base imports ======================
import os
import sys
import argparse
import yaml
import time
# ====== External package imports ================
# ====== Internal package imports ================
from jakarta_analyze.modules.pipeline.pipeline import Pipeline
from jakarta_analyze.modules.utils.misc import run_and_catch_exceptions
from jakarta_analyze.modules.utils.setup import setup, IndentLogger
# ============== Logging  ========================
import logging
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from jakarta_analyze.modules.utils.config_loader import get_config
conf = get_config()
# ================================================


def parse_args(args):
    """Parse command line arguments
    
    Args:
        args: Command line arguments
        
    Returns:
        Parsed arguments object
    """
    parser = argparse.ArgumentParser(description='Run the full video processing pipeline')
    
    parser.add_argument('--config', '-c', required=True,
                        help='Path to YAML pipeline configuration file')
    
    parser.add_argument('--videos-dir', '-v',
                        help='Directory containing videos to process (overrides config)')
    
    parser.add_argument('--output-dir', '-o',
                        help='Directory to save output (overrides config)')
    
    return parser.parse_args(args)


def main(args=None):
    """Main entry point for running the video processing pipeline
    
    Args:
        args: Command line arguments (optional)
        
    Returns:
        int: Exit code (0 on success, non-zero on error)
    """
    if args is None:
        args = sys.argv[1:]
        
    try:
        parsed_args = parse_args(args)
        
        # Check if config file exists
        if not os.path.isfile(parsed_args.config):
            logger.error(f"Config file not found: {parsed_args.config}")
            return 1
        
        # Load pipeline configuration to make any necessary modifications
        with open(parsed_args.config, 'r') as f:
            pipe_conf = yaml.safe_load(f)
        
        # Override videos directory if specified
        if parsed_args.videos_dir:
            for task in pipe_conf['pipeline']['tasks']:
                if task.get('worker_type') == 'ReadFramesFromVidFilesInDir' and 'vid_dir' in task:
                    logger.info(f"Overriding video directory: {parsed_args.videos_dir}")
                    task['vid_dir'] = parsed_args.videos_dir
        
        # Determine output directory
        if parsed_args.output_dir:
            output_dir = parsed_args.output_dir
            logger.info(f"Using command line output directory: {output_dir}")
        elif 'output_dir' in conf:
            output_dir = conf['output_dir']
            logger.info(f"Using config output directory: {output_dir}")
        else:
            # Use default output directory
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            output_dir = os.path.join(project_root, "outputs")
            logger.info(f"Using default output directory: {output_dir}")
            
        # Create output directory if it doesn't exist
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")

        # Write the modified configuration back to the file if needed
        if parsed_args.videos_dir:
            with open(parsed_args.config, 'w') as f:
                yaml.dump(pipe_conf, f)
            
        # Run the pipeline with the config file path
        logger.info("=" * 50)
        logger.info("Starting video processing pipeline")
        logger.info("=" * 50)
        print("Pipeline processing started. Check logs for detailed progress.")
        
        # Create completion marker file to indicate the pipeline is running
        completion_marker = os.path.join(output_dir, "pipeline_completion.txt")
        try:
            with open(completion_marker, 'w') as f:
                f.write("Pipeline started at: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
                f.write("Status: RUNNING\n")
        except Exception as e:
            logger.warning(f"Could not create pipeline status marker: {str(e)}")
        
        # Run the pipeline
        pl = Pipeline(config_file=parsed_args.config, out_path=output_dir)
        start_time = time.time()
        result = pl.run()
        end_time = time.time()
        
        # Calculate processing time
        processing_time = end_time - start_time
        hours, remainder = divmod(processing_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Update completion marker file
        try:
            with open(completion_marker, 'w') as f:
                f.write("Pipeline started at: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)) + "\n")
                f.write("Pipeline completed at: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)) + "\n")
                f.write(f"Processing time: {int(hours)}h {int(minutes)}m {int(seconds)}s\n")
                f.write(f"Status: {'COMPLETED' if result else 'FAILED'}\n")
        except Exception as e:
            logger.warning(f"Could not update pipeline status marker: {str(e)}")
        
        logger.info("=" * 50)
        if result:
            status = "COMPLETED SUCCESSFULLY"
        else:
            status = "COMPLETED WITH ERRORS"
        logger.info(f"Pipeline processing {status}")
        logger.info(f"Processing time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 50)
        
        # Print a clear message to the console
        print("\n" + "=" * 50)
        print(f"Pipeline processing {status}")
        print(f"Processing time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"Output directory: {output_dir}")
        print("=" * 50 + "\n")
        
        return 0 if result else 1
            
    except Exception as e:
        logger.error(f"Error running pipeline: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    setup("run_pipeline")
    sys.exit(main())