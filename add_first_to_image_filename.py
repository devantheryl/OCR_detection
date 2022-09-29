# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 13:01:46 2022

@author: LDE
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 10:01:15 2022

@author: LDE
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 13:34:21 2022

@author: LDE
"""

import os
from matplotlib import pyplot as plt

os.chdir("C:/Users/LDE/Prog/OCR_detection")


import OCR_detection as ocr
import get_number
import cv2 as cv
from os import walk
import numpy as np
from sklearn.metrics import accuracy_score
from PIL import Image, ImageOps, ImageEnhance
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds
import seaborn as sns
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
import time 


"""
LOAD THE DEEP LEARNING MODEL
"""

checkpoint_path = "model/training_real_number_only_2/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

latest = tf.train.latest_checkpoint(checkpoint_dir)

# Create a new model instance
model = Sequential()
model.add(ResNet50(include_top = False, pooling = 'avg', weights = 'imagenet'))
model.add(Dense(512,activation = 'relu'))
model.add(Dense(10,activation = 'softmax'))


# Load the previously saved weights
model.load_weights(latest)



"""
GET ALL THE FILE IN A FOLDER
"""

folder = "C:/Users/LDE/Prog/OCR_detection/Tests_Analyse/production_26.09.22_22-015716_old/"

f = []
for (dirpath, dirnames, filenames) in walk(folder):
    f = [file for file in filenames if ".png" in file]
    break


"""
GO TROUGH ALL THE FILES
"""

img_index = 0
out_directory = "dataset/production_22.09.22/"

batch_number = np.array([2,2,0,1,5,7,1,6])
err_img = []
for filename in f:
    print(filename)
    start = time.time()
    
    
    
    #GET ALL RELEVANT INFORMATION FROM IMAGE
    img_to_save = cv.imread(folder + filename)
    img = cv.rotate(cv.imread(folder + filename), cv.ROTATE_180)
    rectangles,imgs_cropped, img_resized, imgs_th, POIs_total_th, POIs_total_img_resized, POIs_total_img = ocr.find_numbers_positions(img)
    
    if len(POIs_total_img):
        
        #predict the classes
        
        
        classes, probas = get_number.get_number_from_image_POI(model,POIs_total_th)
        proba_score = [probas[i,classe] for i, classe in enumerate(classes)]
        
        
        
        for _,r in rectangles.items():
            img_resized = cv.rectangle(img_resized, (r[0],r[1]), (r[0]+r[2],r[1]+r[3]),(255,0,0),2)
      
        
        
        if len(classes) >=3:
            
            
            classes_prob_first = str(classes[1:3]).strip("[]")
            batch_number_ref_first = str(batch_number[1:3]).strip("[]")
            
       
            classes_prob_second = str(classes[-3:]).strip("[]")
            batch_number_ref_second = str(batch_number[-3:]).strip("[]")
           
            
            directory = "Tests_Analyse/production_26.09.22_22-015716_new/"
            file = filename.split(".")[0]
            
            writed = False
            if classes_prob_first in batch_number_ref_first:
                #cv.imwrite(directory + file + "_True" + ".png", img_to_save)
                writed = True
            if classes_prob_second in batch_number_ref_second:
                #cv.imwrite(directory + file + "_False" + ".png", img_to_save)
                writed = True
                
            if writed == False:
                err_img.append(filename)
        else:
            err_img.append(filename)
    else:
        err_img.append(filename)
                    
            
            
            
            


    
    
    
    
    
    
    
    
    