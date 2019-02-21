# wm_dataset_preparation
This ROS package let you easily prepare a custom dataset for YOLO. 
This will automate the object bounding box selection by removing the background.
Then it will augment the dataset using bluring, reversing and background modifications. 

## Instructions
* catkin_make
* roslaunch openni2_launch openni2.launch
* rosrun wm_dataset_preparation extract_object

## Dependencies
* python
* opencv 2
