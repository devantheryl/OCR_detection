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
from scipy.signal import find_peaks

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


def remove_light_part(img, plot = False):
    
    #NINARY THRESHOLD TO GET THE WHITE PART
    _, img_th = cv.threshold(img,254,255,cv.THRESH_BINARY)
    img_th_summed = np.sum(img_th,axis = 1)/max(np.sum(img_th,axis = 1)) * np.shape(img_th)[0]
    
    
    """
    test avec scipy find_peaks
    """
    peaks, _ = find_peaks(img_th_summed, distance=300)
    if len(peaks):
        max_x1 = peaks[0] + 50
    else: 
        max_x1 = 100
    if len(peaks) > 1:
        max_x2 = peaks[1] - 20
    else:
        max_x2 = 400
    #find the two greather peaks
    
    
    #methode with a controle line
    
    #place a controle line 
    controle_line = np.full(np.shape(img_th_summed)[0],150)
    
    #get the intersection with the controle line
    idx = np.argwhere(np.diff(np.sign(controle_line - img_th_summed))).flatten()
    x = np.array(range(np.shape(img_th)[0]))
    """
    max_x1 = 0
    max_x2 = 0
    for i in range(1,len(idx)):
       if abs(idx[i] - idx[i-1])> abs(max_x1 - max_x2):
           
           
           print(np.sum(img_th_summed[idx[i-1]:idx[i]]))
           if np.sum(img_th_summed[idx[i-1]:idx[i]]) < 500000:
               max_x1 = idx[i-1]
               max_x2 = idx[i]
           
    if max_x2 ==0:
        max_x2 = 550
        
    """
    
    img_cropped = img[max_x1:max_x2,:]
    
    if plot:
        
        plt.imshow(img,'gray')
        plt.show()
        
        
        plt.imshow(img_th,'gray')
        plt.plot(img_th_summed,x)
        plt.plot(controle_line, x, 'r-')
        plt.plot(controle_line[idx],x[idx], 'ro')
        plt.plot(img_th_summed[peaks],peaks, "x")
        plt.show()
        
        
        plt.imshow(img_cropped,'gray')
        plt.show()
    
    
    return img_cropped

def get_ZOI_X(img, plot = False):
    
    #Blurres and adaptif threshold
    blurred = cv.GaussianBlur(img,(9,21),0)
    th = cv.adaptiveThreshold(blurred,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,151,18)

    #inverse the image
    th_not = np.abs(th - 255)
    
    #add a row of black at the top and bottom
    row_of_zero = np.zeros((1,np.shape(th_not)[1]))
    th_not = np.concatenate((row_of_zero,th_not,row_of_zero), axis = 0)
    
    #compute the insensity
    th_not_summed = np.sum(th_not,axis = 1)/max(np.sum(th_not,axis = 1)) * np.shape(th_not)[1]
    th_not_summed[0] = 0
    
    
    """
    merge certain part, usefull to capture the trucated number
    """
    counter = 0
    maybe_merge = False
    first_point = 0

    for i in range(1,len(th_not_summed)):
        diff = th_not_summed[i] - th_not_summed[i-1]
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
            th_not_summed[first_point: i+1] += 10
            counter = 0
            maybe_merge = False
            
    #place a controle line
    controle_line = np.full(np.shape(th_not_summed)[0], 1)
    x = np.array(range(np.shape(th_not_summed)[0]))
    
    #get the intersection with the controle line
    idx  = np.argwhere(np.diff(np.sign(controle_line - th_not_summed))).flatten()


    #get the x coordinate of intereset
    idx_OI = []
    max_x1=0
    max_x2=0
    for i in range(1,len(idx)):
       if abs(idx[i] - idx[i-1])> 100:
           if np.sum(th_not_summed[idx[i-1]:idx[i]]) > 5000 and abs(idx[i] - idx[i-1]) > abs(max_x1-max_x2) :
               max_x1 = idx[i-1]
               max_x2 = idx[i]
    
    ZOI_th = th_not[max_x1:max_x2+1,:]
    ZOI = img[max_x1-1:max_x2,:] # -1 because we add a line of 0 to the original img
    
    
    if plot:
        
        plt.imshow(th_not,'gray')
        plt.plot(th_not_summed,x)
        plt.plot(controle_line, x, 'r-')
        plt.plot(controle_line[idx],x[idx], 'ro')
        plt.show()
        
    
    return ZOI, ZOI_th

def get_POI_y(ZOI, ZOI_th, plot = False):
    
    #get the intensity in y
    ZOI_th_summed = np.sum(ZOI_th,axis = 0)/max(np.sum(ZOI_th,axis = 0)) * np.shape(ZOI_th)[0]
    
    #place a controle line and get the intersection
    controle_line = np.full(np.shape(ZOI_th)[1], 1)
    idy  = np.argwhere(np.diff(np.sign(controle_line - ZOI_th_summed))).flatten()
    y = np.array(range(np.shape(ZOI_th)[1]))
    
    #get all the POI
    POIs = {}
    POIs_th = {}
    for i in range(1,len(idy)):
       if abs(idy[i] - idy[i-1])> 70:
           
           if np.sum(ZOI_th_summed[idy[i-1]:idy[i]]) > 3000:
               POI_y1 = idy[i-1]
               POI_y2 = idy[i]
               POI_base = ZOI[:,POI_y1:POI_y2+1]
               
               #check if the POI has a good size, if not :
               if POI_base.shape[0] > 180:
                   
                   #we take the POI Thresholed and want to find a new POI
                   POI_prob = ZOI_th[:, POI_y1:POI_y2+1]
                   
                   #add a row of 0 at the top and bottom
                   row_of_zero = np.zeros((1,np.shape(POI_prob)[1]))
                   POI_prob = np.concatenate((row_of_zero,POI_prob,row_of_zero), axis = 0)
                   
                   #get the intensity in x of the POI
                   POI_prob_summed = np.sum(POI_prob,axis = 1)/max(np.sum(POI_prob,axis = 1)) * np.shape(POI_prob)[1]
                   
                   #get the intersection with the controle line
                   POI_controle_line = np.full(np.shape(POI_prob)[0], 1)
                   POI_idx  = np.argwhere(np.diff(np.sign(POI_controle_line - POI_prob_summed))).flatten()
                   
                   #get the best POI
                   max_prob_x1 = 0
                   max_prob_x2 = 0
                   for j in range(1,len(POI_idx)):
                       if abs(POI_idx[j]-POI_idx[j-1]) > abs(max_prob_x2-max_prob_x1):
                           max_prob_x1 = POI_idx[j-1]
                           max_prob_x2 = POI_idx[j]
                   
                   POI = POI_base[max_prob_x1-1:max_prob_x2,:] #again -1 because we add a row of 0
                   
                   if plot:
                       POI_x_ = np.array(range(np.shape(POI_prob)[0]))
                       plt.imshow(POI_prob,'gray')
                       plt.plot(POI_prob_summed,POI_x_)
                       plt.plot(POI_controle_line, POI_x_, 'r-')
                       plt.plot(POI_controle_line[POI_idx],POI_x_[POI_idx], 'ro')
                       plt.show()
               
               else:
                   POI = POI_base
               
               
               #resize the POI
               scale_percent = 40 # percent of original size
               width = int(POI.shape[1] * scale_percent / 100)
               height = int(POI.shape[0] * scale_percent / 100)
               dim = (width, height)
               POI = cv.resize(POI,dim, interpolation = cv.INTER_AREA)
               
               #make the POI exploitable by the deep learning algo
               POI_th = cv.adaptiveThreshold(POI,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,61,18)
               POI_th_blurred = cv.GaussianBlur(POI_th,(3,7),0)
               _,POI_th_blurred_th = cv.threshold(POI_th_blurred,240,255,cv.THRESH_BINARY)
               
               #store the POI and POI for deep learning
               POIs[idy[i-1]] = POI
               POIs_th[idy[i-1]] = POI_th_blurred_th
               
    if plot:
        plt.imshow(ZOI_th,'gray')
        plt.plot(y,ZOI_th_summed)
        plt.plot(y,controle_line, 'r-')
        plt.plot(y[idy],controle_line[idy], 'ro')
        plt.show()
        
        
        for key,POI in POIs.items():
            plt.imshow(POI,'gray')
            plt.show() 
               
    return POIs, POIs_th
    
def get_POI_intensity(img_gray, crop_entry = True):
    
    """
    CROP ENTRY
    """
    if crop_entry:
        img_cropped = img_gray[300:1000,340:1300]
    else:
        img_cropped = img_gray
    
    
    img_whithout_light = remove_light_part(img_cropped, True)
    ZOI, ZOI_th = get_ZOI_X(img_whithout_light, plot = False)
    POIs, POIs_th = get_POI_y(ZOI, ZOI_th, plot = False)
    

    
    return POIs,POIs_th
    
    
def get_caps_color(img):
    pass
    
    