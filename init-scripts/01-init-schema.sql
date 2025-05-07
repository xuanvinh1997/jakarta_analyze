-- Create the results schema
CREATE SCHEMA IF NOT EXISTS results;

-- Create boxes table for storing object detection results
CREATE TABLE IF NOT EXISTS results.boxes (
    id SERIAL PRIMARY KEY,
    box_id TEXT,
    frame_number INTEGER,
    video_name TEXT,
    label TEXT,
    confidence FLOAT,
    x_min INTEGER,
    y_min INTEGER,
    x_max INTEGER,
    y_max INTEGER,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Create box_motion table for storing motion tracking results
CREATE TABLE IF NOT EXISTS results.box_motion (
    id SERIAL PRIMARY KEY,
    box_id TEXT,
    frame_number INTEGER,
    video_name TEXT,
    point_count INTEGER,
    avg_motion_x FLOAT,
    avg_motion_y FLOAT,
    motion_magnitude FLOAT,
    motion_direction FLOAT,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_boxes_box_id ON results.boxes(box_id);
CREATE INDEX IF NOT EXISTS idx_boxes_video_frame ON results.boxes(video_name, frame_number);
CREATE INDEX IF NOT EXISTS idx_box_motion_box_id ON results.box_motion(box_id);
CREATE INDEX IF NOT EXISTS idx_box_motion_video_frame ON results.box_motion(video_name, frame_number);