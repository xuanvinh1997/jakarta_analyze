# ============ Base imports ======================
import os
from collections.abc import Iterable
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


class WriteKeysToFiles(PipelineWorker):
    """Pipeline worker to write out specific items in the dictionary to csv files
    """
    def initialize(self, keys, filenames, keys_headers=None, name="", buffer_size=100, additional_data=None, field_separator=",", **kwargs):
        """Initialize with file and key information
        
        Args:
            keys (list): Keys to extract from items
            filenames (list): File names to write to
            keys_headers (list): Headers for each key (optional)
            name (str): Name for this worker
            buffer_size (int): Number of lines to buffer before writing
            additional_data (list): Additional data items to include
            field_separator (str): Separator for CSV fields
        """
        self.key_file_pairs = list(zip(keys, filenames))
        self.keys_headers = keys_headers if keys_headers else [None] * len(keys)
        self.name = name
        self.buffer = {}
        self.files = {}
        self.buffer_size = buffer_size
        self.additional_data = additional_data if additional_data else []
        self.field_separator = field_separator
        self.logger.info(f"Initialized with {len(self.key_file_pairs)} key-file pairs, buffer size: {buffer_size}")

    def startup(self):
        """Startup operations
        """
        self.logger.info(f"Starting up WriteKeysToFiles worker")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.out_path, exist_ok=True)

    def run(self, item):
        """Write specified keys from item to files
        
        Args:
            item: Item containing data to write to files
        """
        # Add a prefix to each line with frame and video info
        prefix = f"{item.get('video_info', {}).get('id', 'unknown')},{item.get('frame_number', -1)}"
        
        # Add additional data from item
        if self.additional_data:
            for data_key in self.additional_data:
                if data_key in item:
                    prefix += f",{item[data_key]}"
                else:
                    prefix += ","  # Empty field if data not found
        
        # Process each key and write to corresponding file
        for (key, filename), header in zip(self.key_file_pairs, self.keys_headers):
            # Skip keys not in the item
            if key not in item:
                continue
                
            if filename not in self.buffer:
                self.buffer[filename] = []
                
            # Convert item data to string format
            data_string = self.make_string(prefix, item[key])
            if data_string:
                self.buffer[filename].append(data_string)
                
            # Write to file if buffer is full
            if len(self.buffer[filename]) >= self.buffer_size:
                self._write_buffer_to_file(filename, header)
        
        # Pass the item to the next worker
        self.done_with_item(item)

    def shutdown(self):
        """Shutdown operations - write any remaining buffered data
        """
        self.close_files()
        self.logger.info("Shutting down WriteKeysToFiles worker")

    def close_files(self):
        """Write any remaining buffered data and close files
        """
        for filename in list(self.buffer.keys()):
            if self.buffer[filename]:
                self._write_buffer_to_file(filename, self.keys_headers[next((i for i, (k, f) in enumerate(self.key_file_pairs) if f == filename), 0)])
        
        # Close any open files
        for f in self.files.values():
            if not f.closed:
                f.close()

    def make_string(self, prefix, data):
        """Convert data to a string format for writing to a file
        
        Args:
            prefix (str): Prefix information (video ID, frame number)
            data: Data to convert to string
            
        Returns:
            str: Formatted string or None if data is empty
        """
        # If the data is not iterable or is a string, just convert to string
        if not isinstance(data, Iterable) or isinstance(data, (str, bytes)):
            return f"{prefix}{self.field_separator}{str(data)}\n"
        
        # Convert data to list to handle any iterable
        data_list = [item for item in data]
        if len(data_list) == 0:
            return None
        
        # Handle nested iterables
        if isinstance(data_list[0], Iterable) and not isinstance(data_list[0], (str, bytes)):
            nl = "\n"
            return nl.join([
                f"{prefix}{self.field_separator}{self.field_separator.join([str(item) for item in subdata])}" 
                for subdata in data_list
            ]) + nl
        else:
            # Handle flat iterables
            return f"{prefix}{self.field_separator}{self.field_separator.join([str(item) for item in data_list])}\n"

    def _write_buffer_to_file(self, filename, header=None):
        """Write buffered data to file
        
        Args:
            filename (str): Name of file to write to
            header (str): Header to use for the file if needed
        """
        filepath = os.path.join(self.out_path, filename)
        
        # Check if file exists to determine if header is needed
        file_exists = os.path.exists(filepath)
        
        # Open file in append mode
        try:
            with open(filepath, 'a') as f:
                # Write header if file is new and header is provided
                if not file_exists and header:
                    f.write(f"video_id,frame_number{','.join(self.additional_data)},{header}\n")
                
                # Write buffered data
                f.writelines(self.buffer[filename])
                
            # Clear buffer
            self.buffer[filename] = []
            self.logger.debug(f"Wrote data to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error writing to {filepath}: {str(e)}")