#!/usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('wm_dataset_preparation')
import sys, os
import rospy
import cv2
import numpy as np
import random
from std_msgs.msg import String
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from wm_dataset_preparation.cfg import object_extractionConfig



class image_generator:

    def __init__(self):
        self.script_dir = sys.path[0]
        self.image_path = os.path.join(self.script_dir, '../images/')
        self.object_name = 'orange.png'
        


    def generate_new_image(self):
        self.background = cv2.imread(self.image_path+"table.jpg")
        self.background  = cv2.cvtColor(self.background , cv2.COLOR_RGB2RGBA)
        self.object = cv2.imread(self.image_path+"objects/"+self.object_name, cv2.IMREAD_UNCHANGED)
        out_image = self.background.copy()
        
        
        height_object = len(self.object)
        width_object = len(self.object[0])
        height_bg = len(self.background)
        width_bg = len(self.background[0])
        # Random position in image
        # Size of image - size of object
        random_y = random.randint(0,height_bg - height_object)
        random_x = random.randint(0,width_bg - width_object)
        
        for line in range(0, height_object):
            for elem in range(0, width_object):
                if self.object[line,elem][3] != 0:
                    out_image[line+random_y,elem+random_x][0] = self.object[line,elem][0]
                    out_image[line+random_y,elem+random_x][1] = self.object[line,elem][1]
                    out_image[line+random_y,elem+random_x][2] = self.object[line,elem][2]

        cv2.imwrite("result_w_bg.png", out_image)
        cv2.imwrite("result_w_bg_crop.png", out_image[random_y:random_y+height_object, random_x:random_x+width_object])


def main(args):
  ic = image_generator()
  ic.generate_new_image()
  

if __name__ == '__main__':
    main(sys.argv)