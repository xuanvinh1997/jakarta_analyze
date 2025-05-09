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
    """Pipeline worker to write out specific items in the dictionary to database tables or MongoDB collections
    """
    def initialize(self, keys, schemas, tables, keys_headers=None, name="", buffer_size=100, 
                  additional_data=None, field_separator=",", columns=None, **kwargs):
        """Initialize with database and key information
        
        Args:
            keys (list): Keys to extract from items
            schemas (list): Database schemas or MongoDB database names
            tables (list): Database tables or MongoDB collection names
            keys_headers (list): Headers for each key (optional)
            name (str): Name for this worker
            buffer_size (int): Number of items to buffer before writing
            additional_data (list): Additional data items to include
            field_separator (str): Separator for fields
            columns (list): Explicit column names for each table (override auto-detection)
        """
        self.key_schema_table_triples = list(zip(keys, schemas, tables))
        self.keys_headers = keys_headers if keys_headers else [None] * len(keys)
        self.name = name
        self.buffer = {}
        self.buffer_size = buffer_size
        self.additional_data = additional_data if additional_data else []
        self.field_separator = field_separator
        self.columns_map = {} if columns is None else {f"{schemas[0]}.{tables[0]}": columns}
        self.dbio = None  # Will be initialized in startup
        self.logger.info(f"Initialized with {len(self.key_schema_table_triples)} key-collection triples, buffer size: {buffer_size}")
        if self.columns_map:
            self.logger.info(f"Using explicit column mapping: {self.columns_map}")

    def startup(self):
        """Startup operations - connect to database
        """
        self.logger.info(f"Starting up WriteKeysToDatabaseTable worker")
        self.dbio = DatabaseIO()
        
        # Initialize buffer for each collection
        for _, schema, table in self.key_schema_table_triples:
            # For MongoDB compatibility, use collection name directly when appropriate
            if schema.lower() == 'mongodb':
                key = table  # Use just the collection name as the key
            else:
                key = f"{schema}.{table}"  # Backward compatibility for SQL DBs
                
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
                
            # For MongoDB compatibility, use collection name directly when appropriate
            if schema.lower() == 'mongodb':
                db_key = table  # Use just the collection name as the key
            else:
                db_key = f"{schema}.{table}"  # Backward compatibility for SQL DBs
            
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
            # For MongoDB compatibility, use collection name directly when appropriate
            if schema.lower() == 'mongodb':
                db_key = table  # Use just the collection name as the key
            else:
                db_key = f"{schema}.{table}"  # Backward compatibility for SQL DBs
                
            if db_key in self.buffer and self.buffer[db_key]:
                # Find the matching header for this schema/table combination
                header_idx = next((i for i, (k, s, t) in enumerate(self.key_schema_table_triples) 
                                if s == schema and t == table), 0)
                header = self.keys_headers[header_idx] if header_idx < len(self.keys_headers) else None
                self._write_buffer_to_db(schema, table, db_key, header)

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
                for i, subdata in enumerate(data_list):
                    # Add box_id if it's not already in the data
                    if "boxes" in header:
                        # For box data, ensure box_id is present or add it
                        box_id = i if (len(additional_values) == 0 or additional_values[0] is None) else additional_values[0]
                        
                    # Convert all items to proper types based on context
                    subdata_values = []
                    for idx, item in enumerate(subdata):
                        if item is None:
                            subdata_values.append(None)
                        elif isinstance(item, (np.floating, float)):
                            subdata_values.append(float(round(item, 6)))  # Round floats to avoid precision issues
                        elif isinstance(item, (np.integer, int)):
                            subdata_values.append(int(item))
                        else:
                            subdata_values.append(str(item))
                    
                    # Handle special case for boxes data to ensure coordinates are properly formatted
                    if len(subdata_values) >= 5 and "boxes" in header:
                        # Add explicit box_id if it's not already in additional values
                        box_values = [box_id] if "box_id" not in prefix_values else []
                        
                        # For box data, validate class name and confidence are sensible
                        class_name = str(subdata_values[0]) if subdata_values[0] is not None else "unknown"
                        confidence = float(subdata_values[1]) if subdata_values[1] is not None else 0.0
                        
                        # Ensure coordinates are all float values
                        # Format: x1, y1, x2, y2
                        if len(subdata_values) >= 6:
                            try:
                                x1 = float(subdata_values[2])
                                y1 = float(subdata_values[3])
                                x2 = float(subdata_values[4])
                                y2 = float(subdata_values[5])
                                
                                # Ensure the right coordinate ordering (min should be less than max)
                                if x1 > x2:
                                    x1, x2 = x2, x1
                                if y1 > y2:
                                    y1, y2 = y2, y1
                                
                                formatted_data.append(prefix_values + box_values + [class_name, confidence, x1, y1, x2, y2])
                            except (ValueError, TypeError):
                                self.logger.warning(f"Invalid box coordinates: {subdata_values[2:6]}, skipping")
                        else:
                            self.logger.warning(f"Box data has insufficient coordinates: {subdata_values}, skipping")
                    else:
                        # Regular data, just append
                        formatted_data.append(prefix_values + subdata_values)
            elif isinstance(data_list[0], dict):
                # Handle list of dictionaries
                for i, item_dict in enumerate(data_list):
                    # Extract values from dict in consistent order
                    dict_values = list(item_dict.values())
                    
                    # Special case for box data from dictionaries
                    if "boxes" in header:
                        # Add box_id if not present
                        box_id = i if (len(additional_values) == 0 or additional_values[0] is None) else additional_values[0]
                        if "box_id" not in item_dict:
                            formatted_data.append(prefix_values + [box_id] + dict_values)
                        else:
                            formatted_data.append(prefix_values + dict_values)
                    else:
                        formatted_data.append(prefix_values + dict_values)
            else:
                # Handle flat iterables - treat as a single row
                flat_values = []
                for item in data_list:
                    if item is None:
                        flat_values.append(None)
                    elif isinstance(item, (np.floating, float)):
                        flat_values.append(float(round(item, 6)))
                    elif isinstance(item, (np.integer, int)):
                        flat_values.append(int(item))
                    else:
                        flat_values.append(str(item))
                formatted_data.append(prefix_values + flat_values)
                
            return formatted_data
        except Exception as e:
            import traceback
            self.logger.error(traceback.format_exc())
            raise e  # Re-raise the exception to be handled by the caller

    def _write_buffer_to_db(self, schema, table, db_key, header=None):
        """Write buffered data to database
        
        Args:
            schema (str): Database schema or MongoDB database name
            table (str): Database table or MongoDB collection name
            db_key (str): Key for accessing buffer
            header (str): Column headers
        """
        if not self.buffer[db_key]:
            return
            
        try:
            # Check if we have explicit column mapping for this table/collection
            if db_key in self.columns_map:
                columns = self.columns_map[db_key]
                self.logger.info(f"Using explicit column mapping for {db_key}: {columns}")
            else:
                # Standard column determination - using video_id instead of video_name for MongoDB compatibility
                columns = ["video_id", "frame_number"]
            
                # Add additional data columns
                if self.additional_data:
                    columns.extend(self.additional_data)
                    
                # Add header columns if provided
                if header:
                    header_columns = [h.strip() for h in header.split(self.field_separator)]
                    columns.extend(header_columns)
            
            # Special handling for specific collections
            processed_data = []
            
            # Apply collection-specific transformations
            if schema.lower() == 'mongodb' and table == 'boxes':
                # Transform data for boxes collection
                for row in self.buffer[db_key]:
                    if len(row) < 9:  # At minimum we need video_id, frame_number, box_id, class, conf, x1, y1, x2, y2
                        self.logger.warning(f"Skipping row with insufficient data: {row}")
                        continue
                        
                    # Validate box_id (should be position 2 if it was added by _format_data_for_db)
                    if row[2] is None:
                        # If box_id is still None, use the row index
                        row[2] = len(processed_data)
                    
                    # Ensure class name is a string, confidence is a float, and coordinates are floats
                    try:
                        # Positions 3-8 should be class_name, confidence, x1, y1, x2, y2
                        row[3] = str(row[3]) if row[3] is not None else "unknown"
                        row[4] = float(row[4]) if row[4] is not None else 0.0
                        
                        # Ensure coordinates are floats
                        for i in range(5, 9):
                            if i < len(row):
                                row[i] = float(row[i]) if row[i] is not None else 0.0
                                
                        processed_data.append(row)
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Error processing box data {row}: {str(e)}")
            elif schema.lower() == 'mongodb' and table == 'box_motion':
                # Transform data for box_motion collection
                for row in self.buffer[db_key]:
                    if len(row) < 8:  # At minimum we need: video_id, frame_number, box_id, num_points, mean_dx, mean_dy, magnitude, angle
                        self.logger.warning(f"Skipping motion row with insufficient data: {row}")
                        continue
                        
                    # Validate box_id (should be position 2)
                    if row[2] is None:
                        # If box_id is still None, use the row index
                        row[2] = len(processed_data)
                    
                    try:
                        # Ensure numeric fields are actually numeric
                        for i in range(3, min(len(row), 8)):
                            if row[i] is not None:
                                try:
                                    row[i] = float(row[i])
                                except (ValueError, TypeError):
                                    row[i] = 0.0
                                    
                        processed_data.append(row)
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Error processing motion data {row}: {str(e)}")
            # Also keep backward compatibility for old schema names
            elif (schema == 'results' and table == 'boxes') or (schema == 'results' and table == 'box_motion'):
                # Legacy schema handling
                for row in self.buffer[db_key]:
                    if len(row) < 7:  # At minimum we need some basic data
                        self.logger.warning(f"Skipping row with insufficient data: {row}")
                        continue
                        
                    # Process numeric fields
                    try:
                        for i in range(3, len(row)):
                            if row[i] is not None and isinstance(row[i], str) and row[i].replace('.', '', 1).isdigit():
                                row[i] = float(row[i]) if '.' in row[i] else int(row[i])
                                
                        processed_data.append(row)
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Error processing data {row}: {str(e)}")
            else:
                # Default case for other tables/collections
                processed_data = self.buffer[db_key]
            
            # Log the final column mapping
            self.logger.info(f"Final columns for {db_key}: {columns}")
            if processed_data and len(processed_data) > 0:
                self.logger.debug(f"Sample row data (processed): {processed_data[0]}")
                
            # Insert data into database
            result = self.dbio.insert_many_into_table(schema, table, columns, processed_data)
            
            # Clear buffer after successful write
            if result:
                count = len(processed_data)
                self.buffer[db_key] = []
                self.logger.info(f"Successfully wrote {count} rows to {db_key}")
            else:
                self.logger.error(f"Failed to write {len(processed_data)} rows to {db_key}")
            
        except Exception as e:
            self.logger.error(f"Error writing to database {db_key}: {str(e)}")
            # Don't clear the buffer on error, to potentially retry later