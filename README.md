# wm_dataset_preparation
This ROS package let you easily prepare a custom dataset for YOLO. 
This will automate the object bounding box selection by removing the background.
Then it will augment the dataset using bluring, reversing and background modifications. 
The main steps to produce a dataset will be :
1. Capturing various pictures of the objects by using a greenscreen
     1. Using a greenscreen will help removing the background surrounding the object
     2. A motorized platform will be use to automate the capture process
 2. Once every object done, we will apply data augmentation on the previous pictures
     1. Since the background is removed from the original picture, we will use various pictures to generate different background
     2. The object will be place randomly in the picture and then crop
     3. Finally, horizontal flip and blurring will be randomly applied on the picture
 3. YOLO file generation
     1. All the picture and the bounding box files will be generate according to the YOLO folder scheme
     2. The configuration file will also be generate

## Instructions
### Capture the object
1. catkin_make
2. roslaunch openni2_launch openni2.launch
3. rosrun wm_dataset_preparation extract_object
4. rqt, select dynamic reconfigure in plugins->configuration
5. Ajust the HSV values if needed
6. rostopic pub /dataset/save_image std_msgs/Bool "data: false"
7. The object image with the substracted background will be saved in the directory images
8. TODO : add parameters 
### Generate object images with random background
1. rosrun wm_dataset_preparation generate_image.py
2. TODO : add parameters 
### YOLO generation
1. TODO

## Dependencies
* ros kinetic
* python
* opencv 2
