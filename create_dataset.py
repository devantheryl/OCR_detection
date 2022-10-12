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
import segment_vials as sv
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


checkpoint_path = "model/training_real_number_only_4_128_10/cp-0095.ckpt"


# Load the previously saved weights
model = keras.models.load_model(checkpoint_path)


"""
GET ALL THE FILE IN A FOLDER
"""
folder = "C:/Users/LDE/Prog/OCR_detection/Tests_Analyse/production_26.09.22_22-015716/"

f = []
for (dirpath, dirnames, filenames) in walk(folder):
    f = [file for file in filenames if ".png" in file]
    break


"""
GO TROUGH ALL THE FILES
"""
write_out = False
img_number = 10000
out_directory = "dataset/production_22.09.22/"
output_dir = "dataset/"

batch_number = np.array([2,2,0,1,5,7,1,6])


problem_file = []

not_passed = []
passed = []
plot = True
prod_type = 1

for filename in f:
    print(filename)
    start = time.time()
    
    if "-" in filename:
        first = True if filename.split("_")[1].split("-")[0] == "True" else False
    else:
        first = True if filename.split("_")[1].split(".")[0] == "True" else False
    
    
    #GET ALL RELEVANT INFORMATION FROM IMAGE
    img = cv.rotate(cv.imread(folder + filename), cv.ROTATE_180)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    #prod_type = 0 if img full resolution
    #prod_type = 1 if img truncated 
    status = ocr.analyse_img(img_gray, first, model, batch_number, plot, prod_type)
    
    if status == "ok":
        passed.append(filename)
    else:
        not_passed.append(filename)
        
        if plot:
            fig, axs = plt.subplots(2)
                
            axs[0].imshow(img_gray,'gray')
            axs[1].text(0,0, filename +"\n" + status)
    
            plt.show()



print(passed)
print(not_passed)

print("ok : ", len(passed)/len(f)*100)
print("rejet : ",len(not_passed)/len(f)*100)

    
    
    
    
    
    
    
    
    