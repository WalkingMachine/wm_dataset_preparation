#!/usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('wm_dataset_preparation')
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from wm_dataset_preparation.cfg import object_extractionConfig
from shutil import copyfile
import time

import rospy
from std_msgs.msg import Empty


def callback(data):
    # self.save = True
    for i in range(0, 10):
        time.sleep(1)
        copyfile("/home/jeffrey/dataset_ws/src/wm_dataset_preparation/result.png",
                 "/home/jeffrey/dataset_ws/src/wm_dataset_preparation/transparent/result" + str(i) + ".png")



def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('save_image', anonymous=True)

    rospy.Subscriber("dataset/save_image", Empty, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    listener()
