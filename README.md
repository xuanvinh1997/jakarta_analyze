# Improving Traffic Safety through (Traffic) Video Analysis

This repository that contains the code necessary to replicate the work done by fellows at the University of Chicago's <a href="https://dssg.uchicago.edu/">Data Science for Social Good (DSSG) </a> 2018 summer fellowship. In partnership with Jakarta Smart City (JSC) and United Nations Global Pulse (UNGP), DSSG developed a pipeline that can be used to analyze traffic videos taken from Jakarta's vast network of CCTV cameras. The pipeline is able to ingest a video, perform a range of computer vision techniques (object detection, object classification, optical flow), and output the results to a PostgreSQL database. This repository allows users to download all of the files necessary to build and launch the pipeline, and customize it as necessary. We also include some other tools related to extracting video metadata, randomly sampling segments from clean videos, and evaluating the performance of various model components.  

## Table of Contents

1. [Introduction](https://github.com/dssg/jakarta_smart_city_traffic_safety#introduction)
2. [Setup](https://github.com/dssg/jakarta_smart_city_traffic_safety#setup)
3. [Non-Pipeline Functionality](https://github.com/dssg/jakarta_smart_city_traffic_safety#non-pipeline-functionality)
4. [Pipeline Functionality](https://github.com/dssg/jakarta_smart_city_traffic_safety#pipeline-functionality)
5. [Testing the Pipeline](https://github.com/dssg/jakarta_smart_city_traffic_safety#testing-the-pipeline)
6. [Workflows](https://github.com/dssg/jakarta_smart_city_traffic_safety#workflows)
7. [Contributors](https://github.com/dssg/jakarta_smart_city_traffic_safety#contributors)

## Introduction

#### Data Science for Social Good at the University of Chicago
![DSSG](https://dssg.uchicago.edu/wp-content/uploads/2017/12/dssglogoucsquare.png)

The Data Science for Social Good Fellowship is a University of Chicago summer program to train aspiring data scientists to work on data mining, machine learning, big data, and data science projects with social impact. Working closely with governments and nonprofits, fellows take on real-world problems in education, health, energy, public safety, transportation, economic development, international development, and more.

For three months they learn, hone, and apply their data science, analytical, and coding skills, collaborate in a fast-paced atmosphere, and learn from mentors coming from industry and academia.

#### Partners: Jakarta Smart City and Pulse Lab Jakarta

<p float="left">
  <img src ="resources/images/jakarta_smart_city_logo.png" alt="alt text" width="288" height="99" />
  <img src="resources/images/Pulse_Lab_Jakarta_logo_with_pulsars.png" alt="alt text" width = "318" height="81" />
</p> 

Jakarta Smart City (JSC) is a government initiative to develop a multi-use, crowdsourced, big data platform to close the digital divide and facilitate data transparency and citizen communication with government officials in Jakarta. The initiative includes a website, [https://smartcity.jakarta.go.id](smartcity.jakarta.go.id), as well as eight (8) citizen complaint channels, such as Qlue, LAPOR!, Balai Warga, popular social media, SMS platform, and e-mail, and CROP Jakarta for civil servants and officials. 
smartcity.jakarta.go.id uses the Google Maps engine and data from the traffic application Waze. 

United Nations Global Pulse is an organization in the United Nations system that focuses on harnessing big data, artificial intelligence, and other emerging technologies for sustainable development and humanitarian action. It is headquartered in New York City, with additional Pulse Labs in Jakarta, Indonesia and Kampala, Uganda. Global Pulse aims to bring together a wide range of stakeholders (academia, agencies, local governments, etc.) to use Big Data in support of the UN’s broader development goals.  

#### Code Base

This repository is a collection of Jupyter Notebooks, Python scripts and a streaming processing system that we call the pipeline. The various scripts are used for downloading and handling video files, extracting their metadata, producing descriptive statistics, and evaluating the results of the streaming processing system. The pipeline itself uses tasks and workers to process videos frame-by-frame. By using Python Multiprocessing, we can distribute tasks across multiple CPU processes, which enables parallelization and active monitoring of the workflow. At a basic level, the pipeline takes in a video, separates it into individual frames, and then pushes each frame through a variety of tasks using a series of asynchonous queues. The user can customize the pipeline by specifying various hyperparameters for each task, and new tasks/workers can be created and incorporated easily.

#### The Pipeline

Conceptually, the stream processing system is organized into tasks, workers, and queues:
* Task: An operation to be performed on an item of interest (in our case, video frames). Operations should be more-or-less atomic, meaning applying the operation to a particular item is independent of the operation to other items. 
* Worker: What actually performs tasks on items of interest.
* Queue: A First in, First out asynchronous queue which contain items. Workers pull from and place into these queues as defined in the pipeline definition file. 

![Pipeline](resources/images/pipeline_concept.png)

Programmatically, tasks are implicitly defined by a Worker which performs that task. In other words, to define a new task, one should write a new class which subclasses the PipelineWorker class found in `project_root/src/modules/pipeline/workers/pipeline_worker.py`

## Setup

Below we detail the hardware, software, computing environment, and external services necessary to run the code in this repo.

#### Hardware

To make full use of python multiprocessing, a multicore processor is a good idea. Because videos and model weights data take a large amount memory, we recommend at least 16GB or 32GB of RAM. 
This project makes use of General Purpose GPU computing, so a GPU is highly recommended. The Tensorflow and PyTorch models can be run without GPU support, but this drastically decreases computation speed. Instructions on how to operate each framework with no GPU can be found in their respective documentation.


#### Software

* This project requires [Python 3.6](https://www.python.org/downloads/) or later, with packages as specified in `requirements.txt`. If you have pip installed, packages can be installed by running `pip install -r requirements.txt`.
* [CUDA](https://developer.nvidia.com/cuda-zone) for using GPUs. We used version V9.0.176.
* [Tensorflow](https://www.tensorflow.org/) for running pre-trained models in our pipeline. We used version 1.8.0.
* [PyTorch](https://pytorch.org/) for running pre-trained models in our pipeline. We used version 0.4.0.
* [OpenCV](https://opencv.org/) for drawing annotated boxes and points on video frames and for running the Lucas-Kanade Motio Detection Algorithm. We used version 3.4.2.
* [ffmpeg](https://www.ffmpeg.org/) for reading, writing, and analyzing streams of video files. We used version 2.8.14.
* [PostgreSQL](https://www.postgresql.org/) for storing non-video data for inputs and outputs to the model. We used version 9.5.10. You should have the command line executable, psql.
* [Computer Vision Annotation Tool (CVAT)](https://github.com/opencv/cvat): We used version 0.1.1. Instructions on installing and running CVAT can be found on their website.
* [Docker](https://www.docker.com/) for running a docker container that contains CVAT. We used version 17.12.1-ce, build 7390fc6.

#### Other Services

* Some code in this repository interacts with [Amazon AWS S3](https://aws.amazon.com/s3/) buckets, though this functionality is not central to any of the main functions of this repository.



#### Environment Variables
The following environment variables should be set:

* `JAKARTAPATH` should point to the root directory of this repository. This allows loading of configuration files.
* `PYTHONPATH` should also include the root directory of this repository. This allows loading from Python modules referenced relative to the project root.

#### Configuration Files

Many system specifications such as model parameters and file paths are contained in `YAML` configuration files found in `project_root/config`. During setup, our system reads all the files with `.yml` or `.yaml` extensions in that directory and combines them into a single configuration object. This allows flexibility in organizing config files.

We recommend using four separate config files: 
* `config.yml` contains general configuration options, including validation parameters.
* `creds.yml` contains credentials necessary for accessing the PostgreSQL database and Amazon AWS services.
* `paths.yml` contains relevant file paths for input and output files.
* `pipeline.yml` defines that pipeline that should be run.

Examples of configuration files can be found in `project_root/config_examples`. In this README, we refer to configuration files using carats to indicate nested parameters, such as `conf>dirs>logs`, which refers to the directory containing logs. 

## Non-Pipeline Functionality

In addition to the core pipeline that processes videos and outputs a database, we also implemented a range of functionality that can be used to download new videos and analyze them. These functions are primarily useful for retrieving new videos, analyzing their quality, and conducting validation tests. They are indispensible for a general workflow that incorporates the pipeline into a general framework for analyzing traffic safety risk. 

#### Configuring the Database

Once you have a PostgreSQL database set up, you can automatically create some of the required schemas and tables by running:

`python src/main/configure_database.py`

These include tables which contain information about videos, and cameras, as well as results from the pipeline.

#### Extracting Video Metadata and Uploading to the Database

Various metadata can be extracted from videos, including extracting their subtitles, frame statistics, and packet statistics, by running:

`python src/main/extract_video_metadata.py`

This will extract metadata from all files in `conf>dirs>raw_videos` and place output in `conf>dirs>subtitles`, `conf>dirs>frame_stats`, and `conf>dirs>packet_stats`. Because subtitles exist in the `.mkv` video files we have but not the `.mp4` files downloaded from the website, this currently only operates on `.mkv` files. This script could be modified to ignore the subtitle extraction for `.mp4` file, however. 

Video metadata can then be upload to the database by running:

`python src/main/upload_video_metadata_to_database.py`

Note: This step should be completed before the pipeline can be run.

#### Uploading files to an S3 Bucket

For long-term storage, raw videos can be uploaded to an S3 bucket by running:

`python src/main/upload_videos_to_s3.py`

#### Downloading New Videos

New videos can be downloaded from Jakarta's Open Data Portal. We provide a Python script that automates this process. There are hundreds of CCTV cameras posted around the city, and users can watch both live streams or streams going back approximately 48 hours. The script allows a user to specify which cameras they would like to download video from, as well as the amount of video they would like to download. The script retrieves the current timestamp, and searches for videos in the previous 48 hours.

To run this script:

`python src/main/download_videos_from_web_portal.py`

The resulting files are in `.mp4` format and placed in the directory specified by`conf>dirs>downloaded_videos`. Note these videos will not have subtitles.

#### Assessing video quality

When starting to work with the videos, it might be important to assess their quality. One simple check one might want to start with is to plot how consistent the frame rate of the videos is. We found that the CCTV video often has dropped frames, which lead to a drop in the average frame rate. An example of how to perform this check can be found in [`resources/video_quality_tests/average_fps.ipynb`](https://github.com/dssg/jakarta_smart_city_traffic_safety/tree/master/resources/video_quality_tests/average_fps.ipynb), using the frame statistics extracted above.

In videos with framerates below 25 fps, we noted that whenever frames were dropped, they were glued together with a time between frames that was either the time with no frames (if that was on the order of a few seconds), or 0.04s as if the video framerate was 25 fps. Therefore we provide functions that will detect segments with no dropped frames, by detecting if the spacing between frames is smaller or larger than expected. This provided completely clean segments for at least two of the cameras used in this work. The function can be found in [`resources/video_quality_tests/frame_spacings.ipynb`](https://github.com/dssg/jakarta_smart_city_traffic_safety/tree/master/resources/video_quality_tests/frame_spacings.ipynb).

#### Sampling and Chunking Videos

You may want to randomly sample videos from different cameras, times, and locations in order to evaluate the pipeline's performance under a variety of circumstances. There are multiple scripts which can be used for this.

We opted to run validation only on clean segments of video. The following script identifies clean segments using the subtitle files found in `conf>dirs>subtitles` and places the resulting list of clean segments in `conf>dirs>video_samples`:

`python src/main/find_contiguous_segments.py`

Once clean segments have been identified, we identify a random sample from those segments by running:

`python src/main/sample_from_contiguous_segments.py`

The output file is placed in `conf>dirs>video_samples`. This file is also used to read information about the segments when validating. To extract the video segments described in this files, you can run:

`python src/main/extract_video_samples_from_videos.py`

which will produce sample files contained in `conf>dirs>video_samples`.


#### Video Annotation

Labeling is an important part of any machine learning application. Because this pipeline is centered around object detection, classification, and motion determination, there are several outputs that benefit from validation tools. We use the Computer Vision Annotation Tool (CVAT), which enables a user to validate all of these outputs. With the tool, users can add bounding boxes, class labels, and trajectories to each object in a video. The results of this labeling are placed in a table, and provide a ground truth to select models and assess their performance.

To download and setup CVAT, see the original documentation <a href = "https://github.com/opencv/cvat/blob/e8b2c4033022902a7be856583fe98b5fe7e0cb4b/cvat/apps/documentation/user_guide.md">here.</a>

This also has information on how to set up Annotation Jobs. In our work, we created a separate job for each video segment. Also, here is the string we used for creating video labels:

```car ~checkbox=stopped_on_road:false ~checkbox=going_the_wrong_way:false pedestrian ~checkbox=on_road:false bicycle ~checkbox=going_the_wrong_way:false motorbike ~checkbox=stopped_on_road:false ~checkbox=going_the_wrong_way:false ~checkbox=on_sidewalk:false @checkbox=more_than_two_people:false bus ~checkbox=stopped_on_road:false ~checkbox=going_the_wrong_way:false truck ~checkbox=stopped_on_road:false ~checkbox=going_the_wrong_way:false @checkbox=heavy_truck:false minibus_van_angkot ~checkbox=stopped_on_road:false ~checkbox=going_the_wrong_way:false train tuktuk_motortrike_bajaj ~checkbox=stopped_on_road:false ~checkbox=going_the_wrong_way:false food_cart_or_street_vendor ~checkbox=stopped_on_road:false ~checkbox=going_the_wrong_way:false other_interesting_event @text=please_describe:""```

Running the `docker-compose` command as documented in the CVAT repo creates a cvat docker container called `cvat` which houses the contents of the annotation tool, as well as two auxiliary containers: `cvat_db` and `cvat_redis`.

We additionally provide a quickstart guide for how to label videos in CVAT <a href = "https://github.com/dssg/jakarta_smart_city_traffic_safety/blob/master/resources/video%20annotation/video_annotation_guide.md">here</a>.

#### Moving CVAT Annotations from the CVAT Docker Container into the PostgreSQL Database

CVAT Annotations are contained in a separate database within the CVAT Docker container. Here is how to extract them and store them in your database.

While inside the `cvat_db` docker container, run the following command to generate a database dump:

`pg_dump -h localhost -U postgres -d cvat -n public --no-owner -f cvat.dump`

To retrieve the file from the docker container run the following from outside the container:

`docker cp cvat_db:/cvat.dump .`

Then you can upload the data to your persistent PostgreSQL database using:

`cat cvat.dump | psql -f -`

To perform validation, we create derived tables from the CVAT output, by running:

`psql -f src/scripts/validation_calcs.sql`

The labeled information is now ready for validation!

#### Validating the model

After the pipeline has been run with some parameters, many plots useful for model validation can be generated by running

`python src/main/precision_recall.py`

The plots will be placed in the folder `conf>dirs>pr_curves` in the config files. The script will generate plots for each separate video chunk, aggregate videos from the same camera (all, all day videos, and all night videos), then all day and all night videos, and finally all videos together. 

You can also change which models will be tested by changing the `conf>validation>model_numbers` list in the config files. Other parameters that should be changed as the models are further examined are:
* The IOU threshold above which two boxes will be matched (`conf>validation>iou_threshold`),
* How much of the top of the image that will be disregarded when validating (`conf>validation>disregard_region`),
* What is the minimum motion detected for angle comparison (`conf>validation>minimum_motion`).

#### Finding events of interest

After running the pipeline on some videos, you might want to find when specific types of events happen in the footage, such as when cars are found coming the wrong way down a specific stretch of road. This can be done either using the results of the semantic segmentation or by defining the regions of interest by hand. An example of how to find such events can be found in [`resources/find_events.ipynb`](https://github.com/dssg/jakarta_smart_city_traffic_safety/blob/master/resources/find_events.ipynb).

## Pipeline Functionality

The pipeline contains several worker processes that perform the various tasks needed to analyze raw videos. In this section, we list the workers that are built into the pipeline, and describe their functionality. This list of modules is not exhaustive, and users can easily plug in new workers as necessary. These workers are listed in workers_list.py. Currently, the workers include the following:

* Write Frames to Video Files
* Read Frames from Video Files
* YOLO3 Object Detection
* Lucas-Kanade Sparse Optical Flow
* Compute Frame Statistics
* Write Keys to Flat Files
* Write Keys to Database
* Within Frame Validator
* Mean Motion Direction (for each object in a frame)

#### Write Frames to Video Files

This module is contained in <a href = "https://github.com/dssg/jakarta_smart_city_traffic_safety/blob/master/src/modules/pipeline/workers/write_frames_to_vid_files.py">write_frames_to_vid_files.py.</a> The worker takes in a series of frames, and outputs a video. The user may specify how many frames they would like to concatenate. This worker will generally be called at a point following one of the other tasks. For instance, this may be called after running the object detector, so that the user can see the results of the bounding boxes and classifications.

#### Read Frames From Video Files

This module is contained in <a href = "https://github.com/dssg/jakarta_smart_city_traffic_safety/blob/master/src/modules/pipeline/workers/read_frames_from_vid_files_in_dir.py">read_frames_from_vid_files_in_dir.py. </a> This module breaks up a video into frames, that can then be passed to the rest of the workers. Generally, this worker should come at the beginning of the pipeline, as the output of this worker is necessary as the inputs for the rest of the workers.

#### YOLO3 Object Detection

This module is contained in <a href = "https://github.com/dssg/jakarta_smart_city_traffic_safety/blob/master/src/modules/pipeline/workers/yolo3_detect.py">yolo3_detect.py.</a>  It provides the core method we deploy for object detection and classification, and is derived from <a href = "https://pjreddie.com/media/files/papers/YOLOv3.pdf">YOLOv3: An Incremental Approach</a> by Joseph Redmon and Ali Farhadi. The main advantage of YOLO is that it runs quickly, and was trained on a fairly extensive dataset. One drawback to its application in Jakarta is that there are objects that are specific to Jakarta and do not appear in the YOLO training set. We provide tools to help overcome this issue by allowing the user to collect labeled data for these more rare events, and therefore retrain YOLO to improve its performance in specific contexts. This worker outputs a dictionary containing frame number, bounding box dimensions, and an object's predicted classification.

To run this worker, you need the YOLO weights found [here](https://pjreddie.com/media/files/yolov3.weights).

#### Lucas-Kanade Sparse Optical Flow

This module is contained in <a href = "https://github.com/dssg/jakarta_smart_city_traffic_safety/blob/master/src/modules/pipeline/workers/lk_sparse_optical_flow.py">lk_sparse_optical_flow.py.</a> This module implements the Lucas-Kanade algorithm to calculate the optical flow for detected corners in objects. The Lucas-Kanade algorithm solves a linear system in the neighborhood of a point to calculate the "flow" from one frame to the next. The output from this method returns a list of arrays containing the vectors for optical flows of the various detected points.

#### Compute Frame Statistics

This module is contained in <a href = "https://github.com/dssg/jakarta_smart_city_traffic_safety/blob/master/src/modules/pipeline/workers/compute_frame_stats.py">compute_frame_stats.py.</a> It takes the boxes and predicted classes output by the YOLO3 module. It allows the user to return values that count the number of each type of object in a frame, as well as the associated confidence scores.

#### Write Keys to Flat Files

This module is contained in <a href ="https://github.com/dssg/jakarta_smart_city_traffic_safety/blob/master/src/modules/pipeline/workers/write_keys_to_files.py">write_keys_to_files.py</a>. It takes as its input the outputs from previous steps, and returns a csv with the relevant outputs.

#### Write Keys to Database

This module is contained in <a href ="https://github.com/dssg/jakarta_smart_city_traffic_safety/blob/master/src/modules/pipeline/workers/write_keys_to_database_table.py">write_keys_to_database_table.py</a>. This module works similarly to the flat files one, except it outputs its results to a postgres database instead of a csv.

#### Mean Motion Direction

This module is contained in <a href ="https://github.com/dssg/jakarta_smart_city_traffic_safety/blob/master/src/modules/pipeline/workers/write_keys_to_files.py">mean_motion_direction.py</a>. It takes the output from the LK Sparse Optical Flow and the boxes from YOLO3 as its inputs. It matches the optical flow points to their corresponding boxes, and returns an average displacement vector for that box. It also returns the magnitude of the displacement, and its angle. These two measures can be used for validation purposes.

#### Semantic Segmenter

We used the <a href ="https://github.com/mapillary/inplace_abn#mapillary-vistas-pre-trained-model">WideResNet38 + DeepLab3 pre-trained algorithm</a> to classify each pixel into particular classes (road, sidewalk, etc.). This helps us identify different image regions so we can then say things like "the motorcycle is on the sidewalk". Semantic segmentation is an expensive process to run; it takes some time to classify each pixel and then turn it into a mask. In our case, we have static cameras and regions such as road and sidewalks (which we are interested in) do not change as often so we will perform this process seldomly and store the masks into the database. 

## Testing the pipeline

Running `python src/main/pipeline_test.py` will run the pipeline.
Each time the pipeline is run, it is assigned a globally unique model number, and information about that model is stored in the database under `results.models`. Results are placed in `conf>dirs>output`, under a subdirectory matching the model number. This is equivalent to `python main/pipeline_test.py -l info`, since `info` is the default logging level. other options are `debug`, `warning` and `error`.

Note: Running the pipeline requires that the database contain metadata for all videos run through the pipeline.

## Workflows

#### Generate Predictions for Videos in the Pipeline:

1. Configure the database
1. Acquire .mkv traffic videos with timecode subtitles
2. Generate video metadata for those videos and upload the metadata to the database
3. Run the videos through the pipeline, configured to upload results to the database
4. Choose some videos to annotate with CVAT and validate the pipeline's results
5. Tweak pipeline and modules

#### Annotate Video Samples
1. Sample and chunk videos to generate short video segments for annotation
2. Set up CVAT in a docker container and create a separate job for each video segment
3. Annotate all videos in CVAT (perhaps have someone else review your annotations)
4. Move annotations from the CVAT docker container to the PostgreSQL database

#### Generate Semantic Segmentation
1. Getting the median image for foreground subtraction. An example of how do so can be found in [resources/semantic_seg/median_cameras.ipynb](https://github.com/dssg/jakarta_smart_city_traffic_safety/blob/master/resources/semantic_seg/median_cameras.ipynb). We have found good results by sampling one frame per second for 100-300 seconds from relatively calm times such as weekend mornings.
2. In order to perform Semantic Segmentation on the images you need to firstly run the [requirements.txt](https://github.com/dssg/jakarta_smart_city_traffic_safety/blob/master/resources/semantic_seg/requirements.txt) file to install all dependencies using pip including pytorch. Place the images into the /resources/semantic_seg/input/ folder, [download weights](https://github.com/dssg/jakarta_smart_city_traffic_safety/tree/master/resources/semantic_seg#mapillary-vistas-pre-trained-model) and then run `python test_vistas.py`.


#### Validate the Pipeline
1. Run the scripts to assess the performance of the models used.
2. Look at images containing predicted and annotated boxes in order to better understand where models are working and failing. An example notebook to do this is contained in [`resources/validation_image_comparison.ipynb`](https://github.com/dssg/jakarta_smart_city_traffic_safety/blob/master/resources/validation_image_comparison.ipynb).
3. Change parameters and test some more!

The particular methods included in each of these steps are detailed above.

## Logging

Most main executable scripts in this repository produce log files for auditing purposes. Log files always contain debug level information and are named according to the user running the script and the script name. Running the pipeline produces a log file containing the model number.

## Fine Tuning

In our work, due to the paucity of labeled traffic footage from Jakarta roads, we used a pretrained model to perform object detection. Naturally, we might desire to fine tune such models for classes that are specific to Jakarta. Currently, fine tuning of object classification models is not implemented in this repository, but many resources exist that explain the process, such as [here](http://wiki.fast.ai/index.php/Fine_tuning) or [here](http://blog.revolutionanalytics.com/2016/08/deep-learning-part-2.html).

## Contributors

João Caldeira, Alex Fout, Aniket Kesari, Raesetje Sefala, Joe Walsh (Technical Mentor), Katy Dupre (Project Manager).
