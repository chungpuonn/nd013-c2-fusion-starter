
# SDCND : Sensor Fusion and Tracking
This is the project for the second course in the  [Udacity Self-Driving Car Engineer Nanodegree Program](https://www.udacity.com/course/c-plus-plus-nanodegree--nd213) : Sensor Fusion and Tracking. 

In this project, you'll fuse measurements from LiDAR and camera and track vehicles over time. You will be using real-world data from the Waymo Open Dataset, detect objects in 3D point clouds and apply an extended Kalman filter for sensor fusion and tracking.

<img src="img/img_title_1.jpeg"/>

The project consists of two major parts: 
1. **Object detection**: In this part, a deep-learning approach is used to detect vehicles in LiDAR data based on a birds-eye view perspective of the 3D point-cloud. Also, a series of performance measures is used to evaluate the performance of the detection approach. 
2. **Object tracking** : In this part, an extended Kalman filter is used to track vehicles over time, based on the lidar detections fused with camera detections. Data association and track management are implemented as well.

The following diagram contains an outline of the data flow and of the individual steps that make up the algorithm. 

<img src="img/img_title_2_new.png"/>

Also, the project code contains various tasks, which are detailed step-by-step in the code. More information on the algorithm and on the tasks can be found in the Udacity classroom. 

## Project File Structure

📦project<br>
 ┣ 📂dataset --> contains the Waymo Open Dataset sequences <br>
 ┃<br>
 ┣ 📂misc<br>
 ┃ ┣ evaluation.py --> plot functions for tracking visualization and RMSE calculation<br>
 ┃ ┣ helpers.py --> misc. helper functions, e.g. for loading / saving binary files<br>
 ┃ ┗ objdet_tools.py --> object detection functions without student tasks<br>
 ┃ ┗ params.py --> parameter file for the tracking part<br>
 ┃ <br>
 ┣ 📂results --> binary files with pre-computed intermediate results<br>
 ┃ <br>
 ┣ 📂student <br>
 ┃ ┣ association.py --> data association logic for assigning measurements to tracks incl. student tasks <br>
 ┃ ┣ filter.py --> extended Kalman filter implementation incl. student tasks <br>
 ┃ ┣ measurements.py --> sensor and measurement classes for camera and lidar incl. student tasks <br>
 ┃ ┣ objdet_detect.py --> model-based object detection incl. student tasks <br>
 ┃ ┣ objdet_eval.py --> performance assessment for object detection incl. student tasks <br>
 ┃ ┣ objdet_pcl.py --> point-cloud functions, e.g. for birds-eye view incl. student tasks <br>
 ┃ ┗ trackmanagement.py --> track and track management classes incl. student tasks  <br>
 ┃ <br>
 ┣ 📂tools --> external tools<br>
 ┃ ┣ 📂objdet_models --> models for object detection<br>
 ┃ ┃ ┃<br>
 ┃ ┃ ┣ 📂darknet<br>
 ┃ ┃ ┃ ┣ 📂config<br>
 ┃ ┃ ┃ ┣ 📂models --> darknet / yolo model class and tools<br>
 ┃ ┃ ┃ ┣ 📂pretrained --> copy pre-trained model file here<br>
 ┃ ┃ ┃ ┃ ┗ complex_yolov4_mse_loss.pth<br>
 ┃ ┃ ┃ ┣ 📂utils --> various helper functions<br>
 ┃ ┃ ┃<br>
 ┃ ┃ ┗ 📂resnet<br>
 ┃ ┃ ┃ ┣ 📂models --> fpn_resnet model class and tools<br>
 ┃ ┃ ┃ ┣ 📂pretrained --> copy pre-trained model file here <br>
 ┃ ┃ ┃ ┃ ┗ fpn_resnet_18_epoch_300.pth <br>
 ┃ ┃ ┃ ┣ 📂utils --> various helper functions<br>
 ┃ ┃ ┃<br>
 ┃ ┗ 📂waymo_reader --> functions for light-weight loading of Waymo sequences<br>
 ┃<br>
 ┣ basic_loop.py<br>
 ┣ loop_over_dataset.py<br>



## Installation Instructions for Running Locally
### Cloning the Project
In order to create a local copy of the project, please click on "Code" and then "Download ZIP". Alternatively, you may of-course use GitHub Desktop or Git Bash for this purpose. 

### Python
The project has been written using Python 3.7. Please make sure that your local installation is equal or above this version. 

### Package Requirements
All dependencies required for the project have been listed in the file `requirements.txt`. You may either install them one-by-one using pip or you can use the following command to install them all at once: 
`pip3 install -r requirements.txt` 

### Waymo Open Dataset Reader
The Waymo Open Dataset Reader is a very convenient toolbox that allows you to access sequences from the Waymo Open Dataset without the need of installing all of the heavy-weight dependencies that come along with the official toolbox. The installation instructions can be found in `tools/waymo_reader/README.md`. 

### Waymo Open Dataset Files
This project makes use of three different sequences to illustrate the concepts of object detection and tracking. These are: 
- Sequence 1 : `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`
- Sequence 2 : `training_segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord`
- Sequence 3 : `training_segment-10963653239323173269_1924_000_1944_000_with_camera_labels.tfrecord`

To download these files, you will have to register with Waymo Open Dataset first: [Open Dataset – Waymo](https://waymo.com/open/terms), if you have not already, making sure to note "Udacity" as your institution.

Once you have done so, please [click here](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files) to access the Google Cloud Container that holds all the sequences. Once you have been cleared for access by Waymo (which might take up to 48 hours), you can download the individual sequences. 

The sequences listed above can be found in the folder "training". Please download them and put the `tfrecord`-files into the `dataset` folder of this project.


### Pre-Trained Models
The object detection methods used in this project use pre-trained models which have been provided by the original authors. They can be downloaded [here](https://drive.google.com/file/d/1Pqx7sShlqKSGmvshTYbNDcUEYyZwfn3A/view?usp=sharing) (darknet) and [here](https://drive.google.com/file/d/1RcEfUIF1pzDZco8PJkZ10OL-wLL2usEj/view?usp=sharing) (fpn_resnet). Once downloaded, please copy the model files into the paths `/tools/objdet_models/darknet/pretrained` and `/tools/objdet_models/fpn_resnet/pretrained` respectively.

### Using Pre-Computed Results

In the main file `loop_over_dataset.py`, you can choose which steps of the algorithm should be executed. If you want to call a specific function, you simply need to add the corresponding string literal to one of the following lists: 

- `exec_data` : controls the execution of steps related to sensor data. 
  - `pcl_from_rangeimage` transforms the Waymo Open Data range image into a 3D point-cloud
  - `load_image` returns the image of the front camera

- `exec_detection` : controls which steps of model-based 3D object detection are performed
  - `bev_from_pcl` transforms the point-cloud into a fixed-size birds-eye view perspective
  - `detect_objects` executes the actual detection and returns a set of objects (only vehicles) 
  - `validate_object_labels` decides which ground-truth labels should be considered (e.g. based on difficulty or visibility)
  - `measure_detection_performance` contains methods to evaluate detection performance for a single frame

In case you do not include a specific step into the list, pre-computed binary files will be loaded instead. This enables you to run the algorithm and look at the results even without having implemented anything yet. The pre-computed results for the mid-term project need to be loaded using [this](https://drive.google.com/drive/folders/1-s46dKSrtx8rrNwnObGbly2nO3i4D7r7?usp=sharing) link. Please use the folder `darknet` first. Unzip the file within and put its content into the folder `results`.

- `exec_tracking` : controls the execution of the object tracking algorithm

- `exec_visualization` : controls the visualization of results
  - `show_range_image` displays two LiDAR range image channels (range and intensity)
  - `show_labels_in_image` projects ground-truth boxes into the front camera image
  - `show_objects_and_labels_in_bev` projects detected objects and label boxes into the birds-eye view
  - `show_objects_in_bev_labels_in_camera` displays a stacked view with labels inside the camera image on top and the birds-eye view with detected objects on the bottom
  - `show_tracks` displays the tracking results
  - `show_detection_performance` displays the performance evaluation based on all detected 
  - `make_tracking_movie` renders an output movie of the object tracking results

Even without solving any of the tasks, the project code can be executed. 

The final project uses pre-computed lidar detections in order for all students to have the same input data. If you use the workspace, the data is prepared there already. Otherwise, [download the pre-computed lidar detections](https://drive.google.com/drive/folders/1IkqFGYTF6Fh_d8J3UjQOSNJ2V42UDZpO?usp=sharing) (~1 GB), unzip them and put them in the folder `results`.

## External Dependencies
Parts of this project are based on the following repositories: 
- [Simple Waymo Open Dataset Reader](https://github.com/gdlg/simple-waymo-open-dataset-reader)
- [Super Fast and Accurate 3D Object Detection based on 3D LiDAR Point Clouds](https://github.com/maudzung/SFA3D)
- [Complex-YOLO: Real-time 3D Object Detection on Point Clouds](https://github.com/maudzung/Complex-YOLOv4-Pytorch)


## License
[License](LICENSE.md)
<br><br/>

## Result
### Section 1: Compute Lidar Point-Cloud from Range Image 

a) Visualize range image channels 

In the Waymo Open dataset, lidar data is stored as a range image. Therefore, the first task is about extracting two of the data channels within the range image, which are "range" and "intensity", and converting the floating-point data to an 8-bit integer value range. After that, the OpenCV library is used to stack the range and intensity image vertically and the visualization is as shown as the Figure 1 follow. 
<img src="report/step_1/range_image_screenshot_05.12.2023_labelled.png"/>
Figure 1: Range and Intensity Image (Frame 1)
<br><br/>


b) Visualize lidar point-cloud 

The second task involves writing code within the function “show_pcl” located in the file “student/objdet_pcl.py”. The goal of this task is to use the Open3D library to display the lidar point-cloud in a 3d viewer in order to develop a feel for the nature of lidar point-clouds. A detailed description of all required steps can be found in the code. The viewer is used to locating and closely inspecting point-clouds on vehicles as demonstrated in Figure 2. 
<img src="report/step_1/example_0_labelled.png"/>
Figure 2: Visualization of lidar point-cloud in sensor coordinate space on Open3D (Frame 1)
<br><br/>
<img src="report/step_1/example_1.png"/>
Figure 2: Example 1 - Car
<img src="report/step_1/example_2.png"/>
Figure 3: Example 2 - Car
<img src="report/step_1/example_3.png"/>
Figure 4: Example 3 - Truck
<img src="report/step_1/example_4.png"/>
Figure 5: Example 4 - Car
<img src="report/step_1/example_5.png"/>
Figure 6: Example 5 - Car
<img src="report/step_1/example_6.png"/>
Figure 7: Example 6 - Car
<img src="report/step_1/example_7.png"/>
Figure 8: Example 7 - Car
<img src="report/step_1/example_8.png"/>
Figure 9: Example 8 - Truck
<img src="report/step_1/example_9.png"/>
Figure 10: Example 9 - Van
<img src="report/step_1/example_10.png"/>
Figure 11: Example 10 - Bus
<img src="report/step_1/example_11.png"/>
Figure 12: Example 11 - Truck towing a trailer
<br><br/>

Stable vehicle features (e.g. rear-bumper, taillights) are identified on most vehicles. The rear window of vehicles 2 and 3 can be recognized by the unoccupied region in the lidar point-cloud and the darker region in the intensity image. The wheels of vehicles can also be easily spotted in the lidar point-cloud, though it could be harder to spot in the range-intensity image. 
<br><br/>

### Section 2: Create Birds-Eye View from Lidar PCL 
a) Convert sensor coordinates to BEV-map coordinates 

The third task involves writing code within the function “bev_from_pcl” located in the file “student/objdet_pcl.py”. The goal of this task is to perform the first step in creating a birds-eye view (BEV) perspective of the lidar point-cloud. Figure 13 depicts the clipped lidar point-cloud in the sensor coordinate space based on the pre-defined configuration,  where 0≤x≤50, −25≤y≤25, −1≤z≤3 
 


Based on the (x,y)-coordinates in sensor space, the respective coordinates within the BEV coordinate space are computed so that in subsequent tasks, the actual BEV map can be filled with lidar data from the point-cloud. The “bev_height” and “bev_width” are set as 608 as contract to the original height and width of 50 in sensor-space. As illustrated in Figure 4, It can be clearly seen that after the conversion from sensor space to BEV space, the overall shape of point-cloud seems to be compressed in the z-axis direction, though the actual z-values remain the same as before.  

Figure 13: Visualization of clipped lidar point-cloud in sensor coordinate space (Frame 1)             |  Figure 14: Visualization of clipped lidar point-cloud in BEV coordinate space (Frame 1) 
:-------------------------:|:-------------------------:
![](report/step_2/clipped_lidar_pcl_based_on_config.png)  |  ![](report/step_2/clipped_lidar_pcl_bev_space.png)


<img src="report/step_2/clipped_lidar_pcl_bev_space_top_view.png"/>
Figure 15: Visualization of clipped lidar point-cloud in BEV coordinate space - top view (Frame 1)

<br><br/>
b) Compute intensity layer of the BEV map 

The goal of the fourth task is to fill the "intensity" channel of the BEV map with data from the point-cloud. In order to do so, all points with the same (x,y)-coordinates within the BEV map are identified and then the intensity value of the top-most lidar point is assigned to the respective BEV pixel. Moreover, the resulting intensity image is normalized using percentiles, in order to make sure that the influence of outlier values (very bright and very dark regions) is sufficiently mitigated and objects of interest (e.g. vehicles) are clearly separated from the background. 

As shown in Figure 16(a), the red arrow depicts the x-axis, green arrow depicts the y-axis, and the blue arrow depicts the z-axis. However, OpenCV has a distinct axe’s convention as shown in Figure 16(b). Due to the differences between the axe's convention of Open3D and OpenCV, it causes the wrong result of intensity map. xxx 

Figure 16: (a) Open3D axe's convention;              |  (b) OpenCV axe’s convention
:-------------------------:|:-------------------------:
![](report/step_2/open3d_axes_convention.png)  |  ![](report/step_2/opencv_axes_convention_resized.png)


Figure 17: (a) Intensity map before correction;              |  (b) Intensity map after correction
:-------------------------:|:-------------------------:
![](report/step_2/intensity_map_before_correction.png)  |  ![](report/step_2/intensity_map_after_correction.png)

<br><br/> 
As illustrated in Figure 8, it has been observed that the zoom-in region of vehicle 1 resemble the shape of trunk, with the middle unobstructed pixels could be the rear window glass where the lidar light pass through without reflecting back to its receiver.

<br><br/> 
<img src="report/step_2/zoom_in_pixels_intensity_value_of_vehicle_1.png"/>

Figure 18: Intensity Map with pixels value - zoom-in region of vehicle 1


c) Compute height layer of the BEV map

The goal of this task is to fill the "height" channel of the BEV map with data from the point-cloud. In order to do so, the sorted and pruned point-cloud “lidar_pcl_top” is used from the previous task and normalized the height in each BEV map pixel by the difference between max. and min. height which is defined in the configs structure. As depicted in Figure 19(b), It can be observed that the height values are decreased from the higher rows of the map (could be the rear roof) to the lower rows of the map (could be the trunk). Besides that, the barrier along the right side which is higher than most vehicles have a higher height pixel value as compared to the rest of the map region. 
 
Figure 19: (a) Original;              |  (b) Zoom-in pixel height values of the region of vehicle 1;
:-------------------------:|:-------------------------:
![](report/step_2/height_map_original.png)  |  ![](report/step_2/height_map_zoom_in_pixels.png)

<br><br/>
### Section 3: Model-based Object Detection in BEV Image 
a) Add a second model

The model-based detection of objects in lidar point-clouds using deep-learning is a heavily researched area with new approaches appearing in the literature and on GitHub every few weeks. On the website Papers With Code and on GitHub, several repositories with code for object detection can be found, such as Complex-YOLO: Real-time 3D Object Detection on Point Clouds and Super Fast and Accurate 3D Object Detection based on 3D LiDAR Point Clouds. 

The goal of this task is to illustrate how a new model can be integrated into an existing framework. The task consists of the following steps: 

1. Cloned the repo [Super Fast and Accurate 3D Object Detection based on 3D LiDAR Point Clouds](https://github.com/maudzung/SFA3D) 

2. Familiarized with the code in SFA3D->test.py with the goal of understanding the steps involved for performing inference with a pre-trained model 

3. Extracted the relevant parameters from SFA3D->test.py->parse_test_configs() and add them to the configs structure in load_configs_model. 

4. Instantiated the model for fpn_resnet in create_model. 

5. Decoded the output and performed post-processing in detect_objects, after model inference has been performed. 

6. Visualized the results by setting the flag show_objects_in_bev_labels_in_camera 

In this project, it’s only focussing on the detection of vehicles, even though the Waymo Open dataset contains labels for other road users as well. 

<br><br/>
Labels and detected objects 
(a) Sequence 1;             |  (b) Sequence 2; | (c) Sequence 3;
:-------------------------:|:-------------------------:|:-------------------------:
![](report/step_3/vehicle_detection_dataset_1.gif)  |  ![](report/step_3/vehicle_detection_dataset_2.gif) | ![](report/step_3/vehicle_detection_dataset_3.gif)

<br><br/>
### Section 4 : Performance Evaluation for Object Detection

The first goal of this task is to find pairings between ground-truth labels and detections, so that we can determine whether an object has been (a) missed (false negative), (b) successfully detected (true positive) or (c) has been falsely reported (false positive). Based on the labels within the Waymo Open Dataset, this task is to compute the geometrical overlap between the bounding boxes of labels and detected objects and determine the percentage of this overlap in relation to the area of the bounding boxes. A default method in the literature to arrive at this value is called intersection over union, which is what has been implemented in this task. Based on the pairings between ground-truth labels and detected objects, the next goal is to determine the number of false positives and false negatives for the current frame. After all frames have been processed, an overall performance measure will be computed based on the results produced in this task. After processing all the frames of a sequence, the performance of the object detection algorithm shall now be evaluated. To do so in a meaningful way, the two standard measures "precision" and "recall" will be used, which are based on the accumulated number of positives and negatives from all frames. 
<br><br/> 
<img src="report/step_4/performance.png"/>
Figure 20: Performance of the object detection algorithm for the Sequence 1 - actual result
<br><br/>

To make sure that the code produces plausible results, the flag “configs_det.use_labels_as_objects” should be set to “True” in a second run. The resulting performance measures for this setting should be the following: 

precision = 1.0, recall = 1.0 
<br><br/> 
<img src="report/step_4/performance_true.png"/>
Figure 21: Performance of the object detection algorithm for the Sequence 1 - testing
<br><br/>