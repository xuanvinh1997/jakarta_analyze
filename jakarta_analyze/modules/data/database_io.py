# ============ Base imports ======================
import os
import psycopg2
import psycopg2.extras
import json
from contextlib import contextmanager
# ====== External package imports ================
import logging
# ====== Internal package imports ================
# ============== Logging  ========================
from jakarta_analyze.modules.utils.setup import IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from jakarta_analyze.modules.utils.config_loader import get_config
conf = get_config()
# ================================================
logger.info("Loading Database IO module: %s", conf)

class DatabaseIO:
    """Class for database operations
    
    This class handles connections to the PostgreSQL database and provides
    methods for common operations like querying and inserting data.
    """
    
    def __init__(self):
        """Initialize database connection parameters from config
        """
        # Get database configuration
        try:
            db_config = conf.get('database', {})
            self.host = db_config.get('host', 'localhost')
            self.port = db_config.get('port', 5432)
            self.dbname = db_config.get('dbname', 'jakarta_traffic')
            self.user = db_config.get('user', 'postgres')
            self.password = db_config.get('password', '')
            
            # Build connection string and ensure password is included properly
            if self.password:
                self.connection_string = f"host={self.host} port={self.port} dbname={self.dbname} user={self.user} password={self.password}"
            else:
                # If no password is specified, remove the password parameter
                self.connection_string = f"host={self.host} port={self.port} dbname={self.dbname} user={self.user}"
                
            logger.debug(f"Database connection initialized for {self.dbname} at {self.host}:{self.port}")
            logger.debug(f"Using connection string: host={self.host} port={self.port} dbname={self.dbname} user={self.user}")
            
            # Automatically create schemas and tables if they don't exist
            self.create_tables_if_not_exist()
            
        except Exception as e:
            logger.error(f"Error initializing database connection: {str(e)}")
            self.connection_string = ""
    
    @contextmanager
    def get_connection(self):
        """Get a database connection
        
        Context manager for obtaining a database connection and handling cleanup.
        
        Yields:
            connection: Database connection object
        """
        connection = None
        try:
            logger.info(f"Connecting to database: {self.connection_string}")
            connection = psycopg2.connect(self.connection_string)
            yield connection
        except Exception as e:
            logger.error(f"Database connection error: {str(e)}")
            raise
        finally:
            if connection:
                connection.close()
    
    @contextmanager
    def get_cursor(self, cursor_factory=None):
        """Get a database cursor
        
        Context manager for obtaining a database cursor and handling cleanup.
        
        Args:
            cursor_factory: Optional cursor factory to use
            
        Yields:
            cursor: Database cursor object
        """
        with self.get_connection() as connection:
            cursor = connection.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
                connection.commit()
            except Exception as e:
                connection.rollback()
                logger.error(f"Database cursor error: {str(e)}")
                raise
            finally:
                cursor.close()
    
    def execute_query(self, query, params=None, fetch=True):
        """Execute a database query
        
        Args:
            query (str): SQL query to execute
            params (tuple): Parameters for the query
            fetch (bool): Whether to fetch results
            
        Returns:
            list: Query results if fetch is True, None otherwise
        """
        try:
            with self.get_cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute(query, params)
                if fetch:
                    return cursor.fetchall()
                return None
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            logger.error(f"Query: {query}")
            if params:
                logger.error(f"Params: {params}")
            return None
    
    def insert_many_into_table(self, schema, table, columns, values):
        """Insert multiple rows into a table
        
        Args:
            schema (str): Database schema
            table (str): Table name
            columns (list): Column names
            values (list): List of value tuples to insert
            
        Returns:
            bool: True on success, False on failure
        """
        if not values:
            return True
            
        try:
            # Build the INSERT statement
            columns_str = ", ".join(columns)
            placeholders = ", ".join(["%s"] * len(columns))
            query = f"INSERT INTO {schema}.{table} ({columns_str}) VALUES ({placeholders})"
            
            # Execute the query with multiple rows
            with self.get_cursor() as cursor:
                psycopg2.extras.execute_batch(cursor, query, values)
                
            logger.debug(f"Inserted {len(values)} rows into {schema}.{table}")
            return True
        except Exception as e:
            logger.error(f"Error inserting data into {schema}.{table}: {str(e)}")
            return False
    
    def get_video_info(self, file_pattern):
        """Get video information from the database
        
        Args:
            file_pattern (str): File name or pattern to search for
            
        Returns:
            dict: Video information or None if not found
        """
        try:
            query = """
            SELECT id, file_name, file_path, height, width, fps, duration
            FROM video.videos
            WHERE file_name LIKE %s
            LIMIT 1
            """
            
            results = self.execute_query(query, (file_pattern,))
            
            if results and len(results) > 0:
                row = results[0]
                return {
                    "id": row["id"],
                    "file_name": row["file_name"],
                    "file_path": row["file_path"],
                    "height": row["height"],
                    "width": row["width"],
                    "fps": row["fps"],
                    "duration": row["duration"]
                }
            return None
        except Exception as e:
            logger.error(f"Error retrieving video info for {file_pattern}: {str(e)}")
            return None
            
    def create_tables_if_not_exist(self):
        """Create database tables if they don't exist
        
        Creates the necessary schema and tables for the pipeline.
        
        Returns:
            bool: True on success, False on failure
        """
        try:
            # Create schemas
            schemas = ["video", "results", "metadata"]
            with self.get_cursor() as cursor:
                for schema in schemas:
                    cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
            
            # Create videos table
            video_table_query = """
            CREATE TABLE IF NOT EXISTS video.videos (
                id SERIAL PRIMARY KEY,
                file_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                height INTEGER,
                width INTEGER,
                fps FLOAT,
                duration FLOAT,
                created_at TIMESTAMP DEFAULT NOW()
            )
            """
            
            # Create boxes table
            boxes_table_query = """
            CREATE TABLE IF NOT EXISTS results.boxes (
                id SERIAL PRIMARY KEY,
                video_id TEXT NOT NULL,
                frame_number INTEGER NOT NULL,
                box_id INTEGER,
                x1 INTEGER,
                y1 INTEGER,
                x2 INTEGER,
                y2 INTEGER,
                confidence FLOAT,
                class_id INTEGER,
                class_name TEXT
            )
            """
            
            # Create box_motion table
            motion_table_query = """
            CREATE TABLE IF NOT EXISTS results.box_motion (
                id SERIAL PRIMARY KEY,
                video_id TEXT NOT NULL,
                frame_number INTEGER NOT NULL,
                box_id INTEGER,
                num_points INTEGER,
                mean_dx FLOAT,
                mean_dy FLOAT,
                magnitude FLOAT,
                angle_radians FLOAT,
                angle_degrees FLOAT
            )
            """
            
            # Execute table creation queries
            with self.get_cursor() as cursor:
                cursor.execute(video_table_query)
                cursor.execute(boxes_table_query)
                cursor.execute(motion_table_query)
                
            logger.info("Database tables created successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating database tables: {str(e)}")
            return False

    def register_video_file(self, file_path):
        """Register a new video file in the database
        
        Args:
            file_path (str): Full path to the video file
            
        Returns:
            dict: Video information dictionary or None on failure
        """
        try:
            import cv2
            
            # Extract file name from path
            file_name = os.path.basename(file_path)
            
            # Open the video to get properties
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                logger.error(f"Could not open video file: {file_path}")
                return None
                
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Release the video capture
            cap.release()
            
            # Insert into database
            query = """
            INSERT INTO video.videos (file_name, file_path, height, width, fps, duration)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """
            params = (file_name, file_path, height, width, fps, duration)
            
            results = self.execute_query(query, params)
            
            if results and len(results) > 0:
                video_id = results[0]["id"]
                logger.info(f"Registered video file in database: {file_name} with ID {video_id}")
                
                # Return the video info dictionary
                return {
                    "id": video_id,
                    "file_name": file_name,
                    "file_path": file_path,
                    "height": height,
                    "width": width,
                    "fps": fps,
                    "duration": duration
                }
            return None
        except Exception as e:
            logger.error(f"Error registering video file {file_path}: {str(e)}")
            return None