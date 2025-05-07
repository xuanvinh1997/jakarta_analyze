# ============ Base imports ======================
import os
import json
from collections.abc import Iterable
# ====== External package imports ================
import numpy as np
# ====== Internal package imports ================
from jakarta_analyze.modules.pipeline.pipeline_worker import PipelineWorker
from jakarta_analyze.modules.data.database_io import DatabaseIO
# ============== Logging  ========================
import logging
from jakarta_analyze.modules.utils.setup import IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from jakarta_analyze.modules.utils.config_loader import get_config
conf = get_config()
# ================================================


class WriteKeysToDatabaseTable(PipelineWorker):
    """Pipeline worker to write out specific items in the dictionary to database tables
    """
    def initialize(self, keys, schemas, tables, keys_headers=None, name="", buffer_size=100, additional_data=None, field_separator=",", **kwargs):
        """Initialize with database and key information
        
        Args:
            keys (list): Keys to extract from items
            schemas (list): Database schemas for tables
            tables (list): Database tables to write to
            keys_headers (list): Headers for each key (optional)
            name (str): Name for this worker
            buffer_size (int): Number of items to buffer before writing
            additional_data (list): Additional data items to include
            field_separator (str): Separator for fields
        """
        self.key_schema_table_triples = list(zip(keys, schemas, tables))
        self.keys_headers = keys_headers if keys_headers else [None] * len(keys)
        self.name = name
        self.buffer = {}
        self.buffer_size = buffer_size
        self.additional_data = additional_data if additional_data else []
        self.field_separator = field_separator
        self.dbio = None  # Will be initialized in startup
        self.logger.info(f"Initialized with {len(self.key_schema_table_triples)} key-schema-table triples, buffer size: {buffer_size}")

    def startup(self):
        """Startup operations - connect to database
        """
        self.logger.info(f"Starting up WriteKeysToDatabaseTable worker")
        self.dbio = DatabaseIO()
        
        # Initialize buffer for each table
        for _, schema, table in self.key_schema_table_triples:
            key = f"{schema}.{table}"
            if key not in self.buffer:
                self.buffer[key] = []

    def run(self, item):
        """Write specified keys from item to database tables
        
        Args:
            item: Item containing data to write to database
        """
        # Prepare common prefix data (video ID, frame number)
        video_id = item.get('video_info', {}).get('id', 'unknown')
        frame_number = item.get('frame_number', -1)
        
        # Add additional data from item
        additional_values = []
        if self.additional_data:
            for data_key in self.additional_data:
                if data_key in item:
                    additional_values.append(item[data_key])
                else:
                    additional_values.append(None)  # NULL if data not found
        
        # Process each key and prepare data for corresponding table
        for (key, schema, table), header in zip(self.key_schema_table_triples, self.keys_headers):
            # Skip keys not in the item
            if key not in item:
                continue
                
            db_key = f"{schema}.{table}"
            
            # Format data for database insertion
            data = self._format_data_for_db(item[key], video_id, frame_number, additional_values, header)
            if data:
                self.buffer[db_key].extend(data)
                
            # Write to database if buffer is full
            if len(self.buffer[db_key]) >= self.buffer_size:
                self._write_buffer_to_db(schema, table, db_key, header)
        
        # Pass the item to the next worker
        self.done_with_item(item)

    def shutdown(self):
        """Shutdown operations - write any remaining buffered data
        """
        self.write_buffers_to_db()
        self.logger.info("Shutting down WriteKeysToDatabaseTable worker")

    def write_buffers_to_db(self):
        """Write any remaining buffered data to database
        """
        for key, schema, table in self.key_schema_table_triples:
            db_key = f"{schema}.{table}"
            if db_key in self.buffer and self.buffer[db_key]:
                header_idx = next((i for i, (k, s, t) in enumerate(self.key_schema_table_triples) 
                                 if f"{s}.{t}" == db_key), 0)
                self._write_buffer_to_db(schema, table, db_key, self.keys_headers[header_idx])

    def _format_data_for_db(self, data, video_id, frame_number, additional_values, header):
        """Format data for database insertion
        
        Args:
            data: Data to format
            video_id (str): Video ID
            frame_number (int): Frame number
            additional_values (list): Additional values to include
            header (str): Header information
            
        Returns:
            list: Formatted data rows for database
        """
        formatted_data = []
        
        try:
            # Common prefix columns
            prefix_values = [video_id, frame_number] + additional_values
            
            # Handle different data types
            if data is None:
                # Handle None data case
                return []
                
            # If data is not iterable or is a string/bytes, make it a single row
            if not isinstance(data, Iterable) or isinstance(data, (str, bytes, dict)):
                if isinstance(data, dict):
                    # Convert dict to values
                    values = list(data.values())
                    formatted_data.append(prefix_values + values)
                else:
                    # Convert to string and make a single value row
                    formatted_data.append(prefix_values + [str(data)])
                return formatted_data
            
            # Convert data to list to handle any iterable
            try:
                data_list = list(data)
            except TypeError:
                # If conversion fails, treat as a single item
                formatted_data.append(prefix_values + [str(data)])
                return formatted_data
                
            if len(data_list) == 0:
                return []
            
            # Handle nested iterables
            if isinstance(data_list[0], (list, tuple, np.ndarray)):
                for subdata in data_list:
                    # Convert all items to strings and handle None values
                    subdata_values = []
                    for item in subdata:
                        if item is None:
                            subdata_values.append("NULL")
                        elif isinstance(item, (np.floating, float)):
                            subdata_values.append(str(round(item, 6)))  # Round floats to avoid precision issues
                        else:
                            subdata_values.append(str(item))
                    formatted_data.append(prefix_values + subdata_values)
            elif isinstance(data_list[0], dict):
                # Handle list of dictionaries
                for item_dict in data_list:
                    # Extract values from dict in consistent order
                    dict_values = list(item_dict.values())
                    formatted_data.append(prefix_values + dict_values)
            else:
                # Handle flat iterables - treat as a single row
                flat_values = []
                for item in data_list:
                    if item is None:
                        flat_values.append("NULL")
                    elif isinstance(item, (np.floating, float)):
                        flat_values.append(str(round(item, 6)))
                    else:
                        flat_values.append(str(item))
                formatted_data.append(prefix_values + flat_values)
                
            return formatted_data
        except Exception as e:
            self.logger.error(f"Error formatting data for database: {str(e)}")
            return []

    def _write_buffer_to_db(self, schema, table, db_key, header=None):
        """Write buffered data to database
        
        Args:
            schema (str): Database schema
            table (str): Database table
            db_key (str): Key for accessing buffer
            header (str): Column headers
        """
        if not self.buffer[db_key]:
            return
            
        try:
            # Prepare columns - use 'video_name' instead of 'video_id' to match schema
            columns = ["video_name", "frame_number"]
            
            # Add additional data columns
            if self.additional_data:
                columns.extend(self.additional_data)
                
            # Add header columns if provided
            if header:
                header_columns = [h.strip() for h in header.split(self.field_separator)]
                columns.extend(header_columns)
            
            # Validate data rows against column count
            sample_row = self.buffer[db_key][0] if self.buffer[db_key] else []
            if len(sample_row) != len(columns):
                self.logger.warning(f"Column count mismatch for {schema}.{table}: {len(columns)} columns defined but data has {len(sample_row)} values")
                
                # Adjust columns or data to match
                if len(sample_row) > len(columns):
                    # Add generic column names for extra data
                    for i in range(len(columns), len(sample_row)):
                        columns.append(f"col_{i}")
                    self.logger.info(f"Added generic column names to match data width: now {len(columns)} columns")
                elif len(sample_row) < len(columns):
                    # Trim columns to match data
                    columns = columns[:len(sample_row)]
                    self.logger.warning(f"Trimmed columns to match data width: now {len(columns)} columns")
            
            # Map column names to match the database schema if needed
            if schema == 'results':
                # Replace any columns that need to be mapped to match schema
                for i, col in enumerate(columns):
                    if col == 'video_id':
                        columns[i] = 'video_name'
                    # Add more mappings as needed for other columns
            
            # Insert data into database
            result = self.dbio.insert_many_into_table(schema, table, columns, self.buffer[db_key])
            
            # Clear buffer after successful write
            if result:
                count = len(self.buffer[db_key])
                self.buffer[db_key] = []
                self.logger.debug(f"Successfully wrote {count} rows to {schema}.{table}")
            else:
                self.logger.error(f"Failed to write {len(self.buffer[db_key])} rows to {schema}.{table}")
            
        except Exception as e:
            self.logger.error(f"Error writing to database {schema}.{table}: {str(e)}")
            # Don't clear the buffer on error, to potentially retry later