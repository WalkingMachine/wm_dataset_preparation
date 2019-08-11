#!/usr/bin/env python

# Author        : Jeffrey Cousineau
# Date          : 11 août 2019
# Description   : Script qui génére le fichier indiquant à YOLO chaque images utilisées
#                 pour l'entrainement ainsi que les tests
#                 => train.txt et test.txt

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


# ARG 1 : Class name in folder images/transparent
# BOUNDING  : XMin, XMax, YMin, YMax


class image_generator:

    def __init__(self):
        # Liste des classes utilisées 
        # TODO : utiliser un fichier de configuration
        self.classes = ["biscuit","frosty fruits","snakes","cloth","dishwasher tab","sponge","trash bags","beer","chocolate milk","coke","juice","lemonade","tea bag","water","carrot","cereals","noodles","onion","vegemite","apple","kiwi","lemon","orange","pear","cheetos","doritos","shapes chicken","shapes pizza","twisties","bowl","shot glass"]

    def generate_new_image(self):
        # Dossier contenant les images générées par generate_image.py
        # TODO : utiliser un fichier de configuration
        objects_folder = "/home/jeffrey/dataset_ws/src/wm_dataset_preparation/images/objects_w_bg"
        train_txt = open(objects_folder + "/train_robocup2019.txt", "w+")
        test_txt = open(objects_folder + "/test_robocup2019.txt", "w+")

        # Boucle pour toutes les classes définies
        for i in self.classes:
            image_path = objects_folder + "/" + i
            img_list = os.listdir(image_path)
            img_list_no_txt = []
            
            # Récupère la liste de toutes les images de la classe
            for j in img_list:
                if j.endswith(".JPEG"):
                    img_list_no_txt.append(j)
            
            # Mélange la liste d'image
            random.shuffle(img_list_no_txt)
            count = 0
            
            # Choisi 30% d'images pour les tests et 70% pour l'entrainement
            # TODO : déterminer les pourcentage dans un fichier de configuration
            for img in img_list_no_txt:
                if count > len(img_list_no_txt)*0.3:
                    train_txt.write("data/" + str(i) + "/" + img+'\n')
                else:
                    test_txt.write("data/" + str(i) + "/" + img+'\n')
                count += 1

        train_txt.close()
        test_txt.close()


def main(args):
    ic = image_generator()
    ic.generate_new_image()


if __name__ == '__main__':
    main(sys.argv)
