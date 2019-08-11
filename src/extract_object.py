#!/usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('wm_dataset_preparation')
import sys
import rospy
import cv2
import numpy as np
import os
from std_msgs.msg import String
from std_msgs.msg import Bool, Int32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from wm_dataset_preparation.cfg import object_extractionConfig
from shutil import copyfile
import time

class image_converter:

  def __init__(self):
    self.image_pub = rospy.Publisher("image_topic_2",Image)
    self.srv = Server(object_extractionConfig, self.srv_callback)
    self.bridge = CvBridge()
    self.save_image_sub = rospy.Subscriber("/dataset/save_image",Int32,self.save_image)

    self.image_sub = rospy.Subscriber("/camera/rgb/image_raw/slow",Image,self.callback)
    self.fgbg = cv2.createBackgroundSubtractorMOG2()
    self.blur = np.ones((5, 5), np.float32) / 25
    self.tapis_low = np.array([0,3,0])
    self.tapis_high = np.array([152,247,255])
    self.first = True
    self.counter = 0
    self.save = True

  def srv_callback(self, config, level):
    self.tapis_low = np.array([config["H_low"],config["S_low"],config["V_low"]])
    self.tapis_high = np.array([config["H_high"],config["S_high"],config["V_high"]])
    return config

  def save_image(self,data):
    #self.save = True
    img_dir = os.listdir("/home/jeffrey/dataset_ws/src/wm_dataset_preparation/images/transparent/")
    for i in range(0,int(data.data)):
        time.sleep(1)
        copyfile("/home/jeffrey/dataset_ws/src/wm_dataset_preparation/result.png", "/home/jeffrey/dataset_ws/src/wm_dataset_preparation/images/transparent/result"+str(i+len(img_dir))+".png")

  def callback(self,data):
    if self.counter < 10:
      self.counter += 1
    else:
      try:
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      except CvBridgeError as e:
        print(e)

      if self.first:
          self.r = cv2.selectROI(cv_image)
          self.first = False

      cv_image = cv_image[int(self.r[1]):int(self.r[1] + self.r[3]), int(self.r[0]):int(self.r[0] + self.r[2])]

      # BACKGROUND SUBSTRACTION
      output= cv_image
      output_no_contour = cv_image.copy()
      cv_image = cv2.filter2D(cv_image,-1,self.blur)

      # CONVERT TO HSV
      hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
      # BLUR TO GET RID OF NOISE
      cv_image = cv2.filter2D(cv_image, -1, self.blur)

      mask = cv2.inRange(hsv, self.tapis_low, self.tapis_high)
      mask2 = mask
      #mask = cv2.medianBlur(mask, 5)
      #mask = cv2.medianBlur(mask, 5)
      mask3 = mask
      mask3 = cv2.bitwise_not(mask3)
      invert = mask3

      #lower_green = np.array([70, 50, 50])
      #upper_green = np.array([85, 255, 255])
      im2, contours, hierarchy = cv2.findContours(mask3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

      if len(contours) != 0:
          # draw in blue the contours that were founded
          #cv2.drawContours(output, contours, -1, 255, 3)

          # find the biggest area
          c = max(contours, key=cv2.contourArea)
          cv2.drawContours(output, c, -1, 255, 3)
          x, y, w, h = cv2.boundingRect(c)
          # draw the book contour (in green)
          cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

          inverted_image = invert[y:y+h, x:x+w]

          out_crop = output_no_contour[y:y+h, x:x+w]
          out_crop = cv2.cvtColor(out_crop, cv2.COLOR_RGB2RGBA)

          for line in range(0, len(inverted_image)):
            for elem in range(0, len(inverted_image[0])):
              if inverted_image[line,elem] != 255:
                out_crop[line,elem] = 0
          cv2.imwrite("/home/jeffrey/dataset_ws/src/wm_dataset_preparation/result.png", out_crop)
          cv2.imshow("invert_crop", invert[y:y+h, x:x+w])
          cv2.imshow("output_crop", out_crop)

          #if self.save:
            #inverted_image = invert[y:y+h, x:x+w]

            
            #out_image = output[y:y+h, x:x+w]
            #out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2RGBA)
            #for line in range(0, len(inverted_image)):
            #  for elem in range(0, len(inverted_image[0])):
            #    if inverted_image[line,elem] != 255:
            #      out_image[line,elem] = 0
            #cv2.imwrite("result.png", out_crop)
            #self.save = False

      
      
      #  *_, alpha = cv2.split(src)
      #  gray_layer = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
      #  dst = cv2.merge((gray_layer, gray_layer, gray_layer, alpha))
      #  cv2.imwrite("result.png", dst)

      #cv2.imshow("Image window", output)
      #cv2.imshow("Image window", output)
      cv2.waitKey(3)

      try:
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(output, "bgr8"))
      except CvBridgeError as e:
        print(e)

def main(args):
  rospy.init_node('image_converter', anonymous=True)
  ic = image_converter()
  
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)