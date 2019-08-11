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

  # Callback pour configurer l'interval HSV
  def srv_callback(self, config, level):
    self.tapis_low = np.array([config["H_low"],config["S_low"],config["V_low"]])
    self.tapis_high = np.array([config["H_high"],config["S_high"],config["V_high"]])
    return config

  # Fonction pour enregistrer l'image actuelle avec le fond extrait
  # TODO: utiliser un chemin relatif
  def save_image(self,data):
    img_dir = os.listdir("/home/jeffrey/dataset_ws/src/wm_dataset_preparation/images/transparent/")
    for i in range(0,int(data.data)):
        time.sleep(1)
        copyfile("/home/jeffrey/dataset_ws/src/wm_dataset_preparation/result.png", "/home/jeffrey/dataset_ws/src/wm_dataset_preparation/images/transparent/result"+str(i+len(img_dir))+".png")

  # Callback de l'image de la caméra
  def callback(self,data):
    # Passer les premières images sinon il y a un problème avec les couleurs
    if self.counter < 10:
      self.counter += 1
    else:
      try:
        # Conversion vers bgr8
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      except CvBridgeError as e:
        print(e)

      if self.first:
          # Sélection de la région d'intérêt
          self.r = cv2.selectROI(cv_image)
          self.first = False
          
      # Découpe de l'image selon la région d'intért
      cv_image = cv_image[int(self.r[1]):int(self.r[1] + self.r[3]), int(self.r[0]):int(self.r[0] + self.r[2])]

      # Substraction du fond de couleur
      output= cv_image
      output_no_contour = cv_image.copy()
      cv_image = cv2.filter2D(cv_image,-1,self.blur)

      # Conversion vers HSV
      hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
      # Blur pour enlever du bruit
      cv_image = cv2.filter2D(cv_image, -1, self.blur)
      
      # Appliquer le masque HSV
      mask = cv2.inRange(hsv, self.tapis_low, self.tapis_high)
      mask2 = mask # Sauvegarde du masque pour visualisation seulement
      mask3 = mask 
      # Inversion du masque binaire
      mask3 = cv2.bitwise_not(mask3)
      invert = mask3
      
      # Algorithme de recherche de contours
      im2, contours, hierarchy = cv2.findContours(mask3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

      # Si un contour est trouvé
      if len(contours) != 0:

          # Trouver le plus gros contour de la liste
          c = max(contours, key=cv2.contourArea)
          cv2.drawContours(output, c, -1, 255, 3)
          x, y, w, h = cv2.boundingRect(c)
          
          # Dessiner le rectangle identifiant l'objet
          cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

          inverted_image = invert[y:y+h, x:x+w]
          out_crop = output_no_contour[y:y+h, x:x+w]
          out_crop = cv2.cvtColor(out_crop, cv2.COLOR_RGB2RGBA)

          for line in range(0, len(inverted_image)):
            for elem in range(0, len(inverted_image[0])):
              if inverted_image[line,elem] != 255:
                out_crop[line,elem] = 0
                
          # Enregistrer dans un fichier temporaire
          cv2.imwrite("/home/jeffrey/dataset_ws/src/wm_dataset_preparation/result.png", out_crop)
          cv2.imshow("invert_crop", invert[y:y+h, x:x+w])
          cv2.imshow("output_crop", out_crop)

      cv2.waitKey(3)

      try:
        # Envoie de l'image sur un topic ROS pour visualisation
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
