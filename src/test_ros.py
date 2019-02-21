#!/usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('wm_dataset_preparation')
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


#cv2.namedWindow('image')

def nothing(x):
    pass

class image_converter:




  def __init__(self):
    self.image_pub = rospy.Publisher("image_topic_2",Image)

    self.bridge = CvBridge()
    #self.image_sub = rospy.Subscriber("camera/rgb/image_raw",Image,self.callback)
    #self.image_sub = rospy.Subscriber("usb_cam/image_raw", Image, self.callback)

    self.fgbg = cv2.createBackgroundSubtractorMOG2()
    self.blur = np.ones((5, 5), np.float32) / 25

    cv2.namedWindow('camera')

    # create trackbars for color change
    #cv2.createTrackbar('R', 'image', 0, 255, nothing)
    #cv2.createTrackbar('G', 'image', 0, 255, nothing)
    #cv2.createTrackbar('B', 'image', 0, 255, nothing)

    # Creating track bar

    default_tapis_low = [0,3,0]
    default_tapis_high = [152,247,255]

    #cv2.createTrackbar('H_low', 'camera', default_tapis_low[0], 179, nothing)
    #cv2.createTrackbar('S_low', 'camera', default_tapis_low[1], 255, nothing)
    #cv2.createTrackbar('V_low', 'camera', default_tapis_low[2], 255, nothing)#

    #cv2.createTrackbar('H_high', 'camera', default_tapis_high[0], 179, nothing)
    #cv2.createTrackbar('S_high', 'camera', default_tapis_high[1], 255, nothing)
    #cv2.createTrackbar('V_high', 'camera', default_tapis_high[2], 255, nothing)

    # create switch for ON/OFF functionality
    self.switch = '0 : OFF \n1 : ON'
    #cv2.createTrackbar(self.switch, 'camera', 0, 1, nothing)

    self.first = True
    #switch = '0 : OFF \n1 : ON'
    #cv2.createTrackbar(switch, 'image', 0, 1, nothing)


  def add_blobs(self,crop_frame):
      frame = cv2.GaussianBlur(crop_frame, (3, 3), 0)
      # Convert BGR to HSV
      hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
      # define range of green color in HSV
      lower_green = np.array([70, 50, 50])
      upper_green = np.array([85, 255, 255])
      # Threshold the HSV image to get only blue colors
      mask = cv2.inRange(hsv, lower_green, upper_green)
      mask = cv2.erode(mask, None, iterations=1)
      mask = cv2.dilate(mask, None, iterations=1)
      # Bitwise-AND mask and original image
      res = cv2.bitwise_and(frame, frame, mask=mask)
      detector = cv2.SimpleBlobDetector_create()
      # Detect blobs.
      reversemask = 255 - mask
      keypoints = detector.detect(reversemask)
      if keypoints:
          print
          "found blobs"
          if len(keypoints) > 4:
              keypoints.sort(key=(lambda s: s.size))
              keypoints = keypoints[0:3]
          # Draw detected blobs as red circles.
          # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
          im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255),
                                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
      else:
          print
          "no blobs"
          im_with_keypoints = crop_frame

      return im_with_keypoints  # , max_blob_dist, blob_center, keypoint_in_orders


  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    mask = np.zeros(cv_image.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    x1 = 10
    x2 = 400
    y1 = 10
    y2 = 400

    rect = (x1,x2,y1,y2)

    # BACKGROUND SUBSTRACTION
    cv_image = cv2.filter2D(cv_image,-1,self.blur)
    #cv_image = self.fgbg.apply(cv_image)
    #cv_image = cv2.medianBlur(cv_image, 5)


    # COLOR DETECTION
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    cv_image = cv2.filter2D(cv_image, -1, self.blur)

    lower_green = np.array([70, 50, 50])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.medianBlur(mask, 5)


    cv_image = cv2.bitwise_and(cv_image, cv_image, mask=mask)
    imgray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    im2, contours, hierarchy = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        # draw in blue the contours that were founded
        cv2.drawContours(cv_image, contours, -1, 255, 3)

        # find the biggest area
        c = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(c)
        # draw the book contour (in green)
        cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #cv2.drawContours(cv_image, contours, -1, (0, 255, 0), 3)

    #cv2.grabCut(cv_image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    #mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    #cv_image = cv_image * mask2[:, :, np.newaxis]

    #cv_image = self.add_blobs(cv_image)
    #cv2.rectangle(cv_image, (x1,y1), (x2,y2), (255,0,0), 2)


    cv2.imshow('image', imgray)
    cv2.waitKey(1)

    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
      print(e)

  def callback_no_topic(self):


   # data = rospy.wait_for_message("/camera/rgb/image_raw", Image)
    data = rospy.wait_for_message("usb_cam/image_raw", Image)
    try:
      cv_image_full = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    if self.first:
        self.r = cv2.selectROI(cv_image_full)
        self.first = False

    if cv2.getTrackbarPos(self.switch,'camera') == 1:
        self.r = cv2.selectROI(cv_image_full)

    cv_image = cv_image_full[int(self.r[1]):int(self.r[1] + self.r[3]), int(self.r[0]):int(self.r[0] + self.r[2])]

    # BACKGROUND SUBSTRACTION
    output= cv_image
    cv_image = cv2.filter2D(cv_image,-1,self.blur)

    # CONVERT TO HSV
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    # BLUR TO GET RID OF NOISE
    cv_image = cv2.filter2D(cv_image, -1, self.blur)

    # get info from track bar and appy to result
    #h_low = cv2.getTrackbarPos('H_low', 'camera')
    #s_low = cv2.getTrackbarPos('S_low', 'camera')
    #v_low = cv2.getTrackbarPos('V_low', 'camera')
    #h_high = cv2.getTrackbarPos('H_high', 'camera')
    #s_high = cv2.getTrackbarPos('S_high', 'camera')
    #v_high = cv2.getTrackbarPos('V_high', 'camera')

    lower_green = np.array([h_low, s_low, v_low])
    upper_green = np.array([h_high, s_high, v_high])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask2 = mask
    #mask = cv2.medianBlur(mask, 5)
    #mask = cv2.medianBlur(mask, 5)
    mask3 = mask
    mask3 = cv2.bitwise_not(mask3)
    invert = mask3

    #cv_image3 = cv_image
    #cv_image2 = cv2.bitwise_and(cv_image, cv_image, mask=mask)
    #cv_image = cv_image2


    #imgray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)


    # TEST EXTRAIRE OBJET
    im2, contours, hierarchy = cv2.findContours(mask3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    #im2, contours, hierarchy = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        # draw in blue the contours that were founded
        #cv2.drawContours(output, contours, -1, 255, 3)

        # find the biggest area
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(output, c, -1, 255, 3)
        x, y, w, h = cv2.boundingRect(c)
        # draw the book contour (in green)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #cv2.drawContours(cv_image, contours, -1, (0, 255, 0), 3)

    #cv2.grabCut(cv_image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    #mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    #cv_image = cv_image * mask2[:, :, np.newaxis]

    #cv_image = self.add_blobs(cv_image)
    #cv2.rectangle(cv_image, (x1,y1), (x2,y2), (255,0,0), 2)

    cv2.imshow('inRange', mask2)
    cv2.imshow('invert', invert)
    cv2.imshow('camera', output)
    cv2.waitKey(1)

    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
      print(e)

def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  while not rospy.is_shutdown():
      try:
        ic.callback_no_topic()
      except KeyboardInterrupt:
        print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
