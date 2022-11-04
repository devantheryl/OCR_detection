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
import random
from random import sample


"""
LOAD THE DEEP LEARNING MODEL
"""
checkpoint_path = "model/training_real_number_only_4_128_10/cp-0095.ckpt"
# Load the previously saved weights
model = keras.models.load_model(checkpoint_path)



ref_folder = "C:/Users/LDE/Prog/OCR_detection/dataset_resized_noTh/1/"
ref_filename = "C:/Users/LDE/Prog/OCR_detection/number_ref_new/ref_1.png"

ref = cv.imread(ref_filename,0)

img0 = cv.imread(ref_folder +"27_True.png - Copie - Copie.png")
img0_gray = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
img0_gray_flat = img0_gray.flatten()

img0_th = cv.adaptiveThreshold(img0_gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,29,2)
img0_th_not = np.abs(img0_th - 255)


plt.imshow(img0_th_not,'gray')
plt.show()

white_part = np.argwhere(img0_th_not)
percent = 0.2
total_white_pixel_number = len(white_part)

#code to darkify x percent of the white, NOISE
def get_random_noise(white_part, img, percent,img_gray_flat):

    random_dark_part_NOISE = sample(list(white_part),int(total_white_pixel_number*percent))
    
    img0_dark_NOISE = img.copy()
    for x,y in random_dark_part_NOISE:
        img0_dark_NOISE[x,y] = max(img_gray_flat)
    
    plt.imshow(img0_dark_NOISE,'gray')
    plt.show()
    
    return img0_dark_NOISE

def get_random_row(white_part, img, nbr_row_merged, percent,img_gray_flat):
#code to darkify x percent of the white, ROW

    img0_dark_ROW = img.copy()
    
    nbr_white_pixel_per_merged_row = {}
    for i in range(0,70,nbr_row_merged):
        nbr_white_pixel_per_merged_row[i] = 0
        for j in range(nbr_row_merged):
            nbr_white_pixel_per_merged_row[i] += len(np.where(white_part[:,0] == i+j)[0])
            
            
    sum_of_chooser_row = 0
    keeped_rows = []
    while sum_of_chooser_row < total_white_pixel_number*percent:
        key = random.choice(list(nbr_white_pixel_per_merged_row.keys()))
        val = nbr_white_pixel_per_merged_row[key]
        
        sum_of_chooser_row += val
        keeped_rows.append(key)
        
        del nbr_white_pixel_per_merged_row[key]
        
    
    for keeped_row in keeped_rows:
        
        for j in range(nbr_row_merged):
            row =  np.where(white_part[:,0] == keeped_row+j)[0]
            for pos in row:
                x,y = white_part[pos]
                img0_dark_ROW[x,y] = max(img_gray_flat)
            
        
    plt.imshow(img0_dark_ROW,'gray')
    plt.show()

    return img0_dark_ROW



img0_dark_NOISE = get_random_noise(white_part,img0_gray.copy(),percent,img0_gray_flat)
img0_dark_ROW = get_random_row(white_part,img0_gray.copy(),2,percent,img0_gray_flat)

number,proba = get_number.get_number_from_image_POI(model, {"noise":img0_dark_NOISE, "row":img0_dark_ROW})
print(number)
print(proba)


cv.imwrite("C:/Users/LDE/Prog/OCR_detection/Defautheque/"  + "test.png", img0_dark_ROW)

img0_th = cv.adaptiveThreshold(img0_dark_ROW,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,29,2)
img0_th_not = np.abs(img0_th - 255)
img0_th_not[ref == 0] = 0


plt.imshow(img0_th_not,'gray')
plt.show()



