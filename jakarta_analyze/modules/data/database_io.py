# ============ Base imports ======================
import os
import json
from contextlib import contextmanager
from datetime import datetime
# ====== External package imports ================
import logging
import pymongo
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
# ====== Internal package imports ================
# ============== Logging  ========================
from jakarta_analyze.modules.utils.setup import IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from jakarta_analyze.modules.utils.config_loader import get_config
conf = get_config()
# ================================================
logger.info("Loading MongoDB Database IO module")

class DatabaseIO:
    """Class for MongoDB database operations
    
    This class handles connections to the MongoDB database and provides
    methods for common operations like querying and inserting data.
    """
    
    def __init__(self):
        """Initialize database connection parameters from config
        """
        # Get database configuration
        try:
            db_config = conf.get('database', {})
            self.mongo_uri = db_config.get('mongo_uri', 'mongodb://localhost:27017')
            self.dbname = db_config.get('dbname', 'jakarta_traffic')
            self.collections_config = db_config.get('collections', {
                'videos': 'videos',
                'boxes': 'boxes',
                'motion': 'box_motion'
            })
            
            # Store timeout setting
            self.timeout = db_config.get('timeout', 30000)  # MongoDB timeout in milliseconds
            
            logger.debug(f"MongoDB connection initialized for {self.dbname} at {self.mongo_uri}")
            
            # Create initial connection to verify it works
            self._test_connection()
            
            # Create indices for better query performance
            self._create_indices()
            
        except Exception as e:
            logger.error(f"Error initializing MongoDB connection: {str(e)}")
    
    def _test_connection(self):
        """Test the MongoDB connection
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Create a client and ping the server
            with MongoClient(self.mongo_uri, serverSelectionTimeoutMS=self.timeout) as client:
                client.admin.command('ping')
                logger.info(f"Successfully connected to MongoDB at {self.mongo_uri}")
                return True
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"MongoDB connection failed: {str(e)}")
            return False
    
    def _get_collection(self, collection_name):
        """Get a MongoDB collection
        
        Args:
            collection_name (str): Name of the collection to get
            
        Returns:
            Collection: MongoDB collection object
        """
        client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=self.timeout)
        db = client[self.dbname]
        return db[collection_name]
    
    def _create_indices(self):
        """Create indices for better query performance
        
        Creates indices on commonly queried fields in collections
        """
        try:
            # Create indices for videos collection
            videos_collection = self._get_collection(self.collections_config['videos'])
            videos_collection.create_index('file_name', unique=True)
            
            # Create indices for boxes collection
            boxes_collection = self._get_collection(self.collections_config['boxes'])
            boxes_collection.create_index([('video_id', pymongo.ASCENDING), ('frame_number', pymongo.ASCENDING)])
            boxes_collection.create_index('box_id')
            
            # Create indices for motion collection
            motion_collection = self._get_collection(self.collections_config['motion'])
            motion_collection.create_index([('video_id', pymongo.ASCENDING), ('frame_number', pymongo.ASCENDING)])
            motion_collection.create_index('box_id')
            
            logger.info("MongoDB indices created successfully")
        except Exception as e:
            logger.error(f"Error creating MongoDB indices: {str(e)}")
    
    def insert_many_into_table(self, schema, table, columns, values):
        """Insert multiple rows into a collection
        
        Args:
            schema (str): Database schema (ignored in MongoDB, kept for compatibility)
            table (str): Collection name
            columns (list): Column names
            values (list): List of value tuples to insert
            
        Returns:
            bool: True on success, False on failure
        """
        if not values:
            return True
            
        try:
            # Map the collection name from the config
            collection_name = self.collections_config.get(table, table)
            collection = self._get_collection(collection_name)
            
            # Convert tuples of values to documents using column names
            documents = []
            for value_tuple in values:
                document = {}
                for i, column in enumerate(columns):
                    if i < len(value_tuple):
                        document[column] = value_tuple[i]
                
                # Add timestamp
                document['timestamp'] = datetime.now()
                documents.append(document)
            
            # Insert the documents
            result = collection.insert_many(documents)
            
            logger.debug(f"Inserted {len(result.inserted_ids)} documents into collection {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error inserting data into collection {table}: {str(e)}")
            return False
    
    def get_video_info(self, file_pattern):
        """Get video information from the database
        
        Args:
            file_pattern (str): File name or pattern to search for
            
        Returns:
            dict: Video information or None if not found
        """
        try:
            collection_name = self.collections_config['videos']
            collection = self._get_collection(collection_name)
            
            # Create a regex pattern for file name search
            import re
            pattern = re.compile(f".*{file_pattern}.*", re.IGNORECASE)
            
            # Find the video document
            video_doc = collection.find_one({'file_name': pattern})
            
            if video_doc:
                # Convert MongoDB _id to string for JSON serialization
                if '_id' in video_doc:
                    video_doc['_id'] = str(video_doc['_id'])
                return video_doc
            return None
        except Exception as e:
            logger.error(f"Error retrieving video info for {file_pattern}: {str(e)}")
            return None
    
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
            
            # Prepare the video document
            video_doc = {
                'file_name': file_name,
                'file_path': file_path,
                'height': height,
                'width': width,
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration,
                'timestamp': datetime.now()
            }
            
            # Insert into database
            collection_name = self.collections_config['videos']
            collection = self._get_collection(collection_name)
            
            # Use upsert to update if exists or insert if not
            result = collection.update_one(
                {'file_name': file_name},
                {'$set': video_doc},
                upsert=True
            )
            
            if result.acknowledged:
                # Get the inserted document
                inserted_doc = collection.find_one({'file_name': file_name})
                if inserted_doc:
                    # Convert MongoDB _id to string for compatibility
                    inserted_doc['id'] = str(inserted_doc['_id'])
                    logger.info(f"Registered video file in MongoDB: {file_name} with ID {inserted_doc['id']}")
                    return inserted_doc
            
            return None
        except Exception as e:
            logger.error(f"Error registering video file {file_path}: {str(e)}")
            return None

    def get_results_boxes(self, model_no=None, file_name=None):
        """Get boxes from the database
        
        Args:
            model_no (str): Model number filter
            file_name (str): Video file name filter
            
        Returns:
            tuple: (List of box documents, List of column names)
        """
        try:
            collection_name = self.collections_config['boxes']
            collection = self._get_collection(collection_name)
            
            # Build query filters
            query = {}
            if model_no:
                query['model_number'] = model_no
            if file_name:
                query['video_id'] = file_name
                
            # Get all matching documents
            boxes = list(collection.find(query))
            
            # Extract column names from first document or use defaults
            columns = list(boxes[0].keys()) if boxes else ['_id', 'video_id', 'frame_number', 'box_id', 
                                                        'x1', 'y1', 'x2', 'y2', 'confidence', 
                                                        'class_id', 'class_name']
            
            return boxes, columns
        except Exception as e:
            logger.error(f"Error retrieving boxes from database: {str(e)}")
            return [], []
    
    def get_results_motion(self, model_no=None, file_name=None):
        """Get motion data from the database
        
        Args:
            model_no (str): Model number filter
            file_name (str): Video file name filter
            
        Returns:
            tuple: (List of motion documents, List of column names)
        """
        try:
            # Get boxes first
            boxes, _ = self.get_results_boxes(model_no, file_name)
            
            # Get motion data
            motion_collection = self._get_collection(self.collections_config['motion'])
            
            # Build lookup of box_ids
            box_ids = [box['box_id'] for box in boxes]
            
            # Query motion data for these box_ids
            motion_docs = list(motion_collection.find({'box_id': {'$in': box_ids}}))
            
            # Join motion data with boxes
            results = []
            for box in boxes:
                box_data = box.copy()
                # Find matching motion data
                motion = next((m for m in motion_docs if m['box_id'] == box['box_id']), {})
                
                # Add motion fields to box data
                box_data['mean_delta_x'] = motion.get('mean_dx', 0)
                box_data['mean_delta_y'] = motion.get('mean_dy', 0)
                box_data['magnitude'] = motion.get('magnitude', 0)
                
                results.append(box_data)
            
            # Extract column names from first document or use defaults
            columns = list(results[0].keys()) if results else []
            
            return results, columns
        except Exception as e:
            logger.error(f"Error retrieving motion data from database: {str(e)}")
            return [], []