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

folder = "C:/Users/LDE/Prog/OCR_detection/Tests_Analyse/production_26.09.22_22-015716/"

f = []
for (dirpath, dirnames, filenames) in walk(folder):
    f.extend(filenames)
    break


"""
GO TROUGH ALL THE FILES
"""
problem_filename = ["3731.png", "3732.png","3733.png", "3734.png", "3735.png", "3736.png", "3737.png","3738.png","3739.png","3740.png",
                    "3741.png","3743.png","5359.png","5983.png"]

datas = pd.DataFrame(data = None, columns=("number", "proba", "stats0", "stats1", "stats2", "stats3"))
img_index = 0
out_directory = "dataset/production_22.09.22/"

batch_number = np.array([2,2,0,1,5,7,1,6])

for filename in f:
    print(filename)
    start = time.time()
    
    #GET ALL RELEVANT INFORMATION FROM IMAGE
    img = cv.rotate(cv.imread(folder + filename), cv.ROTATE_180)
    rectangles,imgs_cropped, img_resized, imgs_th, POIs_total_th, POIs_total_img_resized, POIs_total_img = ocr.find_numbers_positions(img)
    
    if len(POIs_total_img):
        
        #predict the classes
        
        
        classes, probas = get_number.get_number_from_image_POI(model,POIs_total_th)
        
        
        
        proba_score = [probas[i,classe] for i, classe in enumerate(classes)]
        
        
        for _,r in rectangles.items():
            img_resized = cv.rectangle(img_resized, (r[0],r[1]), (r[0]+r[2],r[1]+r[3]),(255,0,0),2)
      
        
        
        if len(classes) >=3:
            
            classes_str_first = str(classes[:3]).strip("[]")
            classes_str_second = str(classes[-5:]).strip("[]")
            batch_number_str = str(batch_number).strip("[]")
            if classes_str_first in batch_number_str or classes_str_second in batch_number_str :
                batch_number_ok = True
            else:
                batch_number_ok = False
            
                    
            if batch_number_ok == False:
                fig, axs = plt.subplots(2)
                
                axs[0].imshow(img_resized,'gray')
                axs[1].text(0,0, filename +"\n"+ str(classes))
        
                plt.show()
            
            
            
            
            for i,(_, poi) in enumerate(POIs_total_th.items()):
                
               
                poi_blurred = cv.GaussianBlur(poi,(3,7),0)
                _,poi_blurred_th = cv.threshold(poi_blurred,240,255,cv.THRESH_BINARY)
                
                                
       
                text = filename
                
                
                """
                fig, axs = plt.subplots(2)
                axs[0].imshow(poi_blurred_th,'gray')
                axs[1].text(0,0, text)
                """
        

            
        else:
            
            fig, axs = plt.subplots(2)
            
            axs[0].imshow(img_resized,'gray')
            axs[1].text(0,0, filename+ "  quality sucks")
    
            plt.show()   
            
    else:
        fig, axs = plt.subplots(2)
            
        axs[0].imshow(img_resized,'gray')
        axs[1].text(0,0, filename + "no batch number")

        plt.show()   
        

    
    
#datas.to_csv(out_directory + "datas.csv", sep=";")       
    

print(problem_filename)
    
    
    
    
    
    