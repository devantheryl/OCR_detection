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





#code to darkify x percent of the white, NOISE
def get_random_noise(white_part, img, percent,img_gray_flat, plot = False):

    total_white_pixel_number = len(white_part)    
    random_dark_part_NOISE = sample(list(white_part),int(total_white_pixel_number*percent))
    
    img0_dark_NOISE = img.copy()
    for x,y in random_dark_part_NOISE:
        img0_dark_NOISE[x,y] = max(img_gray_flat)
    
    if plot:
        plt.imshow(img0_dark_NOISE,'gray')
        plt.show()
    
    return img0_dark_NOISE

def get_random_row(white_part, img, nbr_row_merged, percent,img_gray_flat, plot = False):
#code to darkify x percent of the white, ROW

    img0_dark_ROW = img.copy()
    total_white_pixel_number = len(white_part)
    
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
            
    if plot:
        plt.imshow(img0_dark_ROW,'gray')
        plt.show()

    return img0_dark_ROW

def get_random_blob(white_part, img, percent,img_gray_flat, plot = False):
    
    img0_dark_BLOB = img.copy()
    total_white_pixel_number = len(white_part)
    
    nbr_pixel = int(percent*total_white_pixel_number)
    
    offset = np.random.randint(total_white_pixel_number-nbr_pixel)
    for i in range(nbr_pixel):
        x,y = white_part[offset+i]
        img0_dark_BLOB[x,y] = max(img_gray_flat)
    
    if plot:
        plt.imshow(img0_dark_BLOB,'gray')
        plt.show()
    
    return img0_dark_BLOB
    
    
    
    
prob_folder = "C:/Users/LDE/Prog/OCR_detection/dataset_resized_noTh/"  
prob_filenames = {}
prob_filenames["0"] = ["133_True.png.png", "231_True-no_batch_number.png.png", "1527_True.png.png", 
                      "1923_True.png.png", "12182_True.png.png"]
prob_filenames["1"] = ["0624_False-quality_problem.png.png", "02270_False-quality_problem.png.png", "02608_False-quality_problem.png.png",
                      "02787_False-quality_problem.png.png", "3245_False-quality_problem.png.png"]
prob_filenames["2"] = ["121_True-no_batch_number.png.png", "1305_True-no_batch_number.png.png", "01684_True.png.png",
                      "03705_True.png.png", "04421_True.png.png"]
prob_filenames["3"] = ["38_False-batch_number False.png", "314_False-batch_number False.png", "34971_False-batch_number False.png",
                      "310537_False-batch_number False.png", "3320_False-batch_number False.png"]
prob_filenames["4"] = ["4img2_4-2.png", "3img1_4-2.png", "3img2_4-3.png", "1img2_4-3.png", "2img2_4-3.png"]
prob_filenames["5"] = ["368_False.png.png", "06512_False.png.png", "36346_False.png.png",
                     "06024_False.png.png", "0526_False.png.png"]
prob_filenames["6"] = ["0994_False.png.png", "1958_False.png.png", "1820_False.png.png",
                      "1328_False.png.png", "12113_False.png.png"]
prob_filenames["7"] = ["44_False-batch_number False.png", "35078_False-batch_number False.png.png", "33494_False-batch_number False.png.png",
                      "23496_False-batch_number False.png.png", "21635_False.png.png"]
prob_filenames["8"] = ["29620_False-batch_number False.png", "28690_False-batch_number False.png", "2957_False-quality_problem.png.png",
                      "46585_False-quality_problem.png.png","43498_False-quality_problem.png.png"]
prob_filenames["9"] = ["28380_False-quality_problem.png", "29552_False-quality_problem.png", "26682_False-quality_problem.png",
                      "21566_False-batch_number False.png", "211042_False-batch_number False.png"]
ref_folder = "C:/Users/LDE/Prog/OCR_detection/number_ref_new/"



number_of_defaut_per_percent = 100
for i in range(number_of_defaut_per_percent):
    print(i)
    for percent in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        for key, value in prob_filenames.items():
        
            ref = cv.imread(ref_folder + "ref_" + key + ".png" ,0)
            
            for prob_filename in value:
                img = cv.imread(prob_folder +key + "/" + prob_filename)
                img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                img_gray_flat = img_gray.flatten()
                
                img_th = cv.adaptiveThreshold(img_gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,29,2)
                img_th_not = np.abs(img_th - 255)
                
                
                
                white_part = np.argwhere(img_th_not)
                
                
                
                img_dark_NOISE = get_random_noise(white_part,img_gray.copy(),percent,img_gray_flat)
                img_dark_ROW = get_random_row(white_part,img_gray.copy(),2,percent,img_gray_flat)
                img_dark_BLOB = get_random_blob(white_part,img_gray.copy(),percent,img_gray_flat)
                
                """
                number,proba = get_number.get_number_from_image_POI(model, {"noise":img_dark_NOISE, "row":img_dark_ROW, "blob" : img_dark_BLOB})
                print(number)
                print(proba)
                """
                
                defautheque_folder = "C:/Users/LDE/Prog/OCR_detection/Defautheque/"
                cv.imwrite(defautheque_folder + "blob/" + str(int(percent*100)) + "/"  + key + "/" + str(i) + "_" + prob_filename , img_dark_BLOB)
                cv.imwrite(defautheque_folder + "row/" + str(int(percent*100)) + "/"  + key + "/" + str(i) + "_" + prob_filename , img_dark_ROW)
                cv.imwrite(defautheque_folder + "noise/" + str(int(percent*100)) + "/"  + key +"/" + str(i) + "_" + prob_filename , img_dark_NOISE)
                
                img_th = cv.adaptiveThreshold(img_dark_ROW,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,29,2)
                img_th_not = np.abs(img_th - 255)
                img_th_not[ref == 0] = 0
                



