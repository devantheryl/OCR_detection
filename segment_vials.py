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


    
def get_POI_intensity(img_gray):
    
    #THRESHOLD
    th = cv.adaptiveThreshold(img_gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,61,18)

    edged = cv.Canny(th, 50, 100,L2gradient = True, apertureSize = 3)
    
    th_not = np.abs(th - 255)
    img_gray_not = np.abs(img_gray-255)
    edged_summed = np.sum(th_not,axis = 1)/max(np.sum(th_not,axis = 1)) * np.shape(edged)[0]
    
    
    
    
    
    
    x = np.array(range(np.shape(edged)[0]))
    
    controle_line_x = np.full(np.shape(edged)[0], 1)
    
    idx  = np.argwhere(np.diff(np.sign(controle_line_x - edged_summed))).flatten()

    idx_OI = []
    POI_x1=0
    POI_x2=0
    for i in range(1,len(idx)):
       if abs(idx[i] - idx[i-1])> 80:
           if np.sum(edged_summed[idx[i-1]:idx[i]]) > 5000:
               POI_x1 = idx[i-1]
               POI_x2 = idx[i]
    
    

    POI_th_not = th_not[POI_x1:POI_x2+1,:]
    POI_th_not_summed = np.sum(POI_th_not,axis = 0)/max(np.sum(POI_th_not,axis = 0)) * np.shape(POI_th_not)[0]
    
    controle_line_y = np.full(np.shape(POI_th_not)[1], 1)
    idy  = np.argwhere(np.diff(np.sign(controle_line_y - POI_th_not_summed))).flatten()
    
    POIs = {}
    POIs_th = {}
    for i in range(1,len(idy)):
       if abs(idy[i] - idy[i-1])> 70:
           
           if np.sum(POI_th_not_summed[idy[i-1]:idy[i]]) > 2000:
               POI_y1 = idy[i-1]
               POI_y2 = idy[i]
               POI = img_gray[POI_x1:POI_x2+1, POI_y1:POI_y2+1]
               
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
    
    """
    plt.imshow(img_gray,'gray')
    plt.show()
    
    plt.imshow(th,'gray')
    plt.show() 
    
    """
    
    """
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
    """
    """
    for key,POI in POIs.items():
        plt.imshow(POI,'gray')
        plt.show() 
    """
    
    return POIs,POIs_th
    
    
    
    
    