#!/usr/bin/env python
from __future__ import print_function


# Author        : Jeffrey Cousineau
# Date          : 11 août 2019
# Description   : Script permettant de prendre les images des différentes classes d'objet
#                 avec un fond transparent et de les transposer sur un fond aléatoire afin 
#                 de générer plusieurs images. Ce la génère également le fichier contenant 
#                 les coordonées de l'objet dans l'image

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
        # Liste des classes utilisées pour la génération d'images
        # TODO : utiliser un fichier de configuration
        self.classes = ["biscuit","frosty fruits","snakes","cloth","dishwasher tab","sponge","trash bags","beer","chocolate milk","coke","juice","lemonade","tea bag","water","carrot","cereals","noodles","onion","vegemite","apple","kiwi","lemon","orange","pear","cheetos","doritos","shapes chicken","shapes pizza","twisties","bowl","shot glass"]

        self.script_dir = sys.path[0]
        self.image_path = os.path.join(self.script_dir, '../images/')

    # Fonction qui génère les nouvelles images ainsi que le fichier de coordonnées
    def generate_new_image(self):
        count = 0
        self.class_id = 0
        
        # Boucle pour toutes les classes  d'objet
        for i in self.classes:
            self.object_name = i
            
            # Chemin vers le dossier comprenant les différents fonds
            background_path = "/home/jeffrey/dataset_ws/src/wm_dataset_preparation/images/background"
            # Chemin vers le dossier final où seront enregistrées les images
            final_folder = "/home/jeffrey/dataset_ws/src/wm_dataset_preparation/images/objects_w_bg/"+self.object_name
            bg_list = os.listdir(background_path)
            try:
                os.mkdir(final_folder)
            except:
                print("Folder already exists")
            image_path = "/home/jeffrey/dataset_ws/src/wm_dataset_preparation/images/transparent/"+self.object_name
            img_list = os.listdir(image_path)
            
            # Génère 3 images différentes pour chaque image d'origine
            for i in range(0,2):
                for img in img_list:
                    f = open(final_folder + "/" + str(count)+".txt", "w+")
                    print(str(self.class_id) + " " + self.object_name + " "+str(count))
                    
                    # Lecture de l'image de fond et conversion vers RGBA
                    self.background = cv2.imread(background_path+"/"+bg_list[random.randint(0,len(bg_list)-1)])
                    self.background  = cv2.cvtColor(self.background , cv2.COLOR_RGB2RGBA)
                    self.object = cv2.imread(image_path+"/"+img, cv2.IMREAD_UNCHANGED)
                    out_image = self.background.copy()

                    try:
                        height_object = len(self.object)
                    except:
                        print(img)
                    width_object = len(self.object[0])
                    height_bg = len(self.background)
                    width_bg = len(self.background[0])
                    
                    # Génération d'une position aléatoire
                    # Grandeur de l'image - grandeur de l'objet
                    random_y = random.randint(0,height_bg - height_object)
                    random_x = random.randint(0,width_bg - width_object)
                    
                    # Transposition de l'image de l'objet sur le fond
                    for line in range(0, height_object):
                        for elem in range(0, width_object):
                            if self.object[line,elem][3] != 0:
                                out_image[line+random_y,elem+random_x][0] = self.object[line,elem][0]
                                out_image[line+random_y,elem+random_x][1] = self.object[line,elem][1]
                                out_image[line+random_y,elem+random_x][2] = self.object[line,elem][2]
                    
                    # Enregistrement de l'image finale
                    cv2.imwrite(final_folder+"/"+str(count)+".JPEG", out_image)
                    count += 1

                    # Génération du fichier de coordonnées (bounding boxes)
                    x_min = float(random_x) / float(width_bg)
                    x_max = (float(random_x) + float(width_object)) / float(width_bg)
                    final_width = x_max-x_min

                    y_min = float(random_y) / float(height_bg)
                    y_max = (float(random_y) + float(height_object)) / float(height_bg)
                    final_height = y_max - y_min
                    
                    # Enregistrement du fichier 
                    f.write(str(self.class_id) + " " + str(x_min+final_width/2) + " " + str(y_min+final_height/2) + " " + str(final_width) + " " + str(final_height))
                    f.close()
            self.class_id += 1

def main(args):
  ic = image_generator()
  ic.generate_new_image()

if __name__ == '__main__':
    main(sys.argv)
