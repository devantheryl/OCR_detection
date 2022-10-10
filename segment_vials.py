# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 10:22:42 2022

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


    
def get_POI_intensity(img_gray, crop_entry = True):
    
    #THRESHOLD
    if crop_entry:
        img_cropped = img_gray[300:1000,340:1400]
    else:
        img_cropped = img_gray
    
    
    img_cropped_not = np.abs(img_cropped - 255)
    _, img_cropped_th = cv.threshold(img_cropped,254,255,cv.THRESH_BINARY)
    img_cropped_not_summed = np.sum(img_cropped_th,axis = 1)/max(np.sum(img_cropped_th,axis = 1)) * np.shape(img_cropped_th)[0]
    #img_cropped_not_summed[-1] = np.shape(img_cropped_th)[0]
    
    
    controle_line_img_cropped = np.full(np.shape(img_cropped_not_summed)[0],100)
    idx_cropped = np.argwhere(np.diff(np.sign(controle_line_img_cropped - img_cropped_not_summed))).flatten()
    x_cropped = np.array(range(np.shape(img_cropped_th)[0]))
    
    max_x1 = 0
    max_x2 = 0
    for i in range(1,len(idx_cropped)):
       if abs(idx_cropped[i] - idx_cropped[i-1])> abs(max_x1 - max_x2):
           
           #print(np.sum(img_cropped__not_summed[idx_cropped[i-1]:idx_cropped[i]]))
           if np.sum(img_cropped_not_summed[idx_cropped[i-1]:idx_cropped[i]]) < 5000:
               max_x1 = idx_cropped[i-1]
               max_x2 = idx_cropped[i]
           else:
               max_x1 = idx_cropped[0]
               max_x2 = 550
           
    if max_x2 ==0:
        max_x2 = 550
        
    
    img_cropped_cropped = img_cropped[max_x1:max_x2,:]
    
    
    
    blurred = cv.GaussianBlur(img_cropped_cropped,(9,21),0)
    th = cv.adaptiveThreshold(blurred,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,121,16)

    edged = cv.Canny(th, 50, 100,L2gradient = True, apertureSize = 3)
    
    th_not = np.abs(th - 255)
    row_of_zero = np.zeros((1,np.shape(th_not)[1]))
    th_not = np.concatenate((row_of_zero,th_not,row_of_zero), axis = 0)
    
    edged_summed = np.sum(th_not,axis = 1)/max(np.sum(th_not,axis = 1)) * np.shape(edged)[0]
    edged_summed[0] = 0
    
    counter = 0
    maybe_merge = False
    first_point = 0

    for i in range(1,len(edged_summed)):
        diff = edged_summed[i] - edged_summed[i-1]
        if diff < 0 and abs(diff) > 20:
            first_point = i
            maybe_merge = True
            counter = 0
        if maybe_merge:
            counter += 1
        if counter > 10:
            counter = 0
            maybe_merge = False
        if (diff > 0 and abs(diff) > 20) and maybe_merge:
            edged_summed[first_point: i+1] += 10
            counter = 0
            maybe_merge = False
            
            
        
        
    
    x = np.array(range(np.shape(th_not)[0]))
    
    controle_line_x = np.full(np.shape(th_not)[0], 1)
    
    idx  = np.argwhere(np.diff(np.sign(controle_line_x - edged_summed))).flatten()

    idx_OI = []
    max_POI_x1=0
    max_POI_x2=0
    for i in range(1,len(idx)):
       if abs(idx[i] - idx[i-1])> 100:
           if np.sum(edged_summed[idx[i-1]:idx[i]]) > 5000 and abs(idx[i] - idx[i-1]) > abs(max_POI_x1-max_POI_x2) :
               max_POI_x1 = idx[i-1]
               max_POI_x2 = idx[i]
    
    

    POI_th_not = th_not[max_POI_x1:max_POI_x2+1,:]
    POI_th_not_summed = np.sum(POI_th_not,axis = 0)/max(np.sum(POI_th_not,axis = 0)) * np.shape(POI_th_not)[0] * 0.75
    
    controle_line_y = np.full(np.shape(POI_th_not)[1], 1)
    idy  = np.argwhere(np.diff(np.sign(controle_line_y - POI_th_not_summed))).flatten()
    
    POIs = {}
    POIs_th = {}
    for i in range(1,len(idy)):
       if abs(idy[i] - idy[i-1])> 70:
           
           if np.sum(POI_th_not_summed[idy[i-1]:idy[i]]) > 3000:
               POI_y1 = idy[i-1]
               POI_y2 = idy[i]
               POI_base = img_cropped_cropped[max_POI_x1:max_POI_x2+1, POI_y1:POI_y2+1]
               
               """
               plt.imshow(POI_base)
               plt.show()
               """
               
               if POI_base.shape[0] > 180:
                   
                   POI_prob = POI_th_not[:, POI_y1:POI_y2+1]
                   row_of_zero = np.zeros((2,np.shape(POI_prob)[1]))
                   POI_prob = np.concatenate((row_of_zero,POI_prob,row_of_zero), axis = 0)
                   POI_summed = np.sum(POI_prob,axis = 1)/max(np.sum(POI_prob,axis = 1)) * np.shape(POI_prob)[1]
                   POI_controle_line_x = np.full(np.shape(POI_prob)[0], 1)
                   POI_idx  = np.argwhere(np.diff(np.sign(POI_controle_line_x - POI_summed))).flatten()
                   test = 0
                   
                   """
                   POI_x_ = np.array(range(np.shape(POI_prob)[0]))
                   plt.imshow(POI_prob,'gray')
                   plt.plot(POI_summed,POI_x_)
                   plt.plot(POI_controle_line_x, POI_x_, 'r-')
                   plt.plot(POI_controle_line_x[POI_idx],POI_x_[POI_idx], 'ro')
                   plt.show()
                   """
                   
                   max_prob_x1 = 0
                   max_prob_x2 = 0
                   for j in range(1,len(POI_idx)):
                       if abs(POI_idx[j]-POI_idx[j-1]) > abs(max_prob_x2-max_prob_x1):
                           max_prob_x1 = POI_idx[j-1]
                           max_prob_x2 = POI_idx[j]
                   
                   POI = POI_base[max_prob_x1:max_prob_x2,:]
               
               else:
                   POI = POI_base
               
               
               
               scale_percent = 40 # percent of original size
               width = int(POI.shape[1] * scale_percent / 100)
               height = int(POI.shape[0] * scale_percent / 100)
               dim = (width, height)
               POI = cv.resize(POI,dim, interpolation = cv.INTER_AREA)
               
               
               POI_th = cv.adaptiveThreshold(POI,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,61,18)
               POI_th_blurred = cv.GaussianBlur(POI_th,(3,7),0)
               _,POI_th_blurred_th = cv.threshold(POI_th_blurred,240,255,cv.THRESH_BINARY)
               
               
               POIs[idy[i-1]] = POI
               POIs_th[idy[i-1]] = POI_th_blurred_th
    
    y = np.array(range(np.shape(POI_th_not)[1]))
    
    
    plt.imshow(img_cropped,'gray')
    plt.show()
    
    
    plt.imshow(img_cropped_th,'gray')
    plt.plot(img_cropped_not_summed,x_cropped)
    plt.plot(controle_line_img_cropped, x_cropped, 'r-')
    plt.plot(controle_line_img_cropped[idx_cropped],x_cropped[idx_cropped], 'ro')
    plt.show()
    
    plt.imshow(blurred,'gray')
    plt.show()
    
    plt.imshow(th,'gray')
    plt.show() 

    
    
    
    plt.imshow(th_not,'gray')
    plt.plot(edged_summed,x)
    plt.plot(controle_line_x, x, 'r-')
    plt.plot(controle_line_x[idx],x[idx], 'ro')
    plt.show()
    
    
    plt.imshow(POI_th_not,'gray')
    plt.plot(y,POI_th_not_summed)
    plt.plot(y,controle_line_y, 'r-')
    plt.plot(y[idy],controle_line_y[idy], 'ro')
    plt.show()
    
    
    for key,POI in POIs.items():
        plt.imshow(POI,'gray')
        plt.show() 
    
    
    
    return POIs,POIs_th
    
    
def get_caps_color(img):
    pass
    
    