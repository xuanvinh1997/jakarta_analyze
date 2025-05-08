# ============ Base imports ======================
import os
import argparse
import json
from datetime import datetime
# ====== External package imports ================
from pymongo import MongoClient, ASCENDING
# ====== Internal package imports ================
# ============== Logging  ========================
import logging
from jakarta_analyze.modules.utils.setup import IndentLogger, setup
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from jakarta_analyze.modules.utils.config_loader import get_config
conf = get_config()
# ================================================

def setup_mongodb(drop_existing=False):
    """Set up MongoDB collections for the Jakarta Traffic Analysis project
    
    Args:
        drop_existing (bool): Whether to drop existing collections
    
    Returns:
        bool: True if setup successful, False otherwise
    """
    try:
        # Get MongoDB connection parameters from config
        db_config = conf.get('database', {})
        mongo_uri = db_config.get('mongo_uri', 'mongodb://localhost:27017')
        dbname = db_config.get('dbname', 'jakarta_traffic')
        
        # Connect to MongoDB
        logger.info(f"Connecting to MongoDB at {mongo_uri}")
        client = MongoClient(mongo_uri)
        
        # Test connection
        client.admin.command('ping')
        logger.info("MongoDB connection successful")
        
        # Get database
        db = client[dbname]
        
        # Get collection names from config
        collections_config = db_config.get('collections', {
            'videos': 'videos',
            'boxes': 'boxes',
            'motion': 'box_motion'
        })
        
        # Drop existing collections if requested
        if drop_existing:
            for collection_name in collections_config.values():
                if collection_name in db.list_collection_names():
                    logger.info(f"Dropping existing collection: {collection_name}")
                    db.drop_collection(collection_name)
        
        # Create collections and indices
        # Videos collection
        videos_collection = db[collections_config['videos']]
        videos_collection.create_index([('file_name', ASCENDING)], unique=True)
        logger.info(f"Created videos collection: {collections_config['videos']}")
        
        # Boxes collection
        boxes_collection = db[collections_config['boxes']]
        boxes_collection.create_index([('video_id', ASCENDING), ('frame_number', ASCENDING)])
        boxes_collection.create_index([('box_id', ASCENDING)])
        logger.info(f"Created boxes collection: {collections_config['boxes']}")
        
        # Motion collection
        motion_collection = db[collections_config['motion']]
        motion_collection.create_index([('video_id', ASCENDING), ('frame_number', ASCENDING)])
        motion_collection.create_index([('box_id', ASCENDING)])
        logger.info(f"Created motion collection: {collections_config['motion']}")
        
        # Insert a test document to verify write permissions
        test_doc = {
            'type': 'test',
            'created_at': datetime.now(),
            'message': 'MongoDB setup successful'
        }
        videos_collection.insert_one(test_doc)
        videos_collection.delete_one({'type': 'test'})
        
        logger.info("MongoDB setup completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up MongoDB: {str(e)}")
        return False

def main():
    """Main function to run MongoDB setup
    """
    parser = argparse.ArgumentParser(description="Set up MongoDB for Jakarta Traffic Analysis")
    parser.add_argument("--drop", action="store_true", help="Drop existing collections")
    args = parser.parse_args()
    
    setup("setup_mongodb")
    success = setup_mongodb(drop_existing=args.drop)
    
    if success:
        logger.info("MongoDB setup completed successfully")
    else:
        logger.error("MongoDB setup failed")

if __name__ == "__main__":
    main()