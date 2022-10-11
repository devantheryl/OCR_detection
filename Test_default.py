# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 10:35:16 2022

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
checkpoint_path = "model/training_real_number_only_4_128_10/cp-0016.ckpt"
# Load the previously saved weights
model = keras.models.load_model(checkpoint_path)



"""
GET ALL THE FILE IN A FOLDER
"""

folder = "C:/Users/LDE/Prog/OCR_detection/Tests_Analyse/non-conforme/"

f = {'10_False.png' : np.array([6,6,6,6,6,6,6,6]),
 '11_True.png' : np.array([7,7,7,7,7,7,7,7]),
 '12_False.png' : np.array([8,8,8,8,8,8,8,8]),
 '13_True.png' : np.array([9,9,9,9,9,9,9,9]),
 '14_False.png' : np.array([8,8,8,8,8,8,8,8]),
 '15_True - Copie.png' : np.array([8,8,8,8,8,8,8,8]),
 '15_True.png' : np.array([8,8,8,8,8,8,8,8]),
 '16_False.png' : np.array([8,8,8,8,8,8,8,8]),
 '17_True.png' : np.array([8,8,8,8,8,8,8,8]),
 '18_False - Copie (2).png' : np.array([8,8,8,8,8,8,8,8]),
 '18_False - Copie (3).png' : np.array([8,8,8,8,8,8,8,8]),
 '18_False.png' : np.array([8,8,8,8,8,8,8,8]),
 '2_False_.png' : np.array([1,1,1,1,1,1,1,1]),
 '300_False.png' : np.array([2,2,0,1,5,0,1,5]),
 '302_False.png' : np.array([2,2,0,1,5,1,1,5]),
 '303_True.png' : np.array([2,2,2,1,5,0,1,5]),
 '304_False.png' : np.array([2,2,0,1,5,3,1,5]),
 '306_False.png' : np.array([2,2,0,1,5,4,1,5]),
 '312_False.png' : np.array([2,2,0,1,5,5,1,5]),
 '313_True.png' : np.array([2,2,6,1,5,0,1,5]),
 '325_True.png' : np.array([2,2,7,1,5,0,1,5]),
 '326_False.png' : np.array([2,2,0,1,5,8,1,5]),
 '338_False.png' : np.array([2,2,0,1,5,9,1,5]),
 '3_True.png' : np.array([0,0,0,0,0,0,0,0]),
 '5_True.png' : np.array([2,2,2,2,2,2,2,2]),
 '6_False.png' : np.array([2,2,2,2,2,2,2,2]),
 '7_True.png' : np.array([3,3,3,3,3,3,3,3]),
 '8_False.png' : np.array([4,4,4,4,4,4,4,4]),
 '9_True.png' : np.array([5,5,5,5,5,5,5,5]),
 'TEST2_False.png' : np.array([2,2,0,1,5,6,7,6]),
 'TEST_True.png' : np.array([2,2,0,1,5,6,7,6])}

"""
GO TROUGH ALL THE FILES
"""
not_passed = []
passed = []
plot = False

for filename,batch_number in f.items():

    start = time.time()
    
    first = True if filename.split("_")[1].split(".")[0] == "True" else False
    #first = True if filename.split("_")[1].split("-")[0] == "True" else False
    
    #GET ALL RELEVANT INFORMATION FROM IMAGE
    img = cv.rotate(cv.imread(folder + filename), cv.ROTATE_180)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    status = ocr.analyse_img(img_gray, first, model, batch_number, plot)
    
    if status == "ok":
        passed.append(filename)
    else:
        not_passed.append(filename)
        
        if plot:
            fig, axs = plt.subplots(2)
                
            axs[0].imshow(img_gray,'gray')
            axs[1].text(0,0, filename +"\n" + status)
    
            plt.show()
        
print("ok : ", len(passed)/len(f)*100)
print("rejet : ",len(not_passed)/len(f)*100)


"""
GET ALL THE FILE IN A FOLDER
"""
folder = "C:/Users/LDE/Prog/OCR_detection/Tests_Analyse/production_27.09.22_22-015676/"

f = []
for (dirpath, dirnames, filenames) in walk(folder):
    f = [file for file in filenames if ".png" in file]
    break
batch_number = np.array([2,2,0,1,5,6,7,6])

for filename in f:
    start = time.time()
    
    first = True if filename.split("_")[1].split(".")[0] == "True" else False
    #first = True if filename.split("_")[1].split("-")[0] == "True" else False
    
    #GET ALL RELEVANT INFORMATION FROM IMAGE
    img = cv.rotate(cv.imread(folder + filename), cv.ROTATE_180)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    status = ocr.analyse_img(img_gray, first, model, batch_number, plot)
    
    if status == "ok":
        passed.append(filename)
    else:
        not_passed.append(filename)
        
        if plot:
            fig, axs = plt.subplots(2)
                
            axs[0].imshow(img_gray,'gray')
            axs[1].text(0,0, filename +"\n" + status)
    
            plt.show()
    
print("ok : ", len(passed)/len(f)*100)
print("rejet : ",len(not_passed)/len(f)*100)



"""
GET ALL THE FILE IN A FOLDER
"""
folder = "C:/Users/LDE/Prog/OCR_detection/Tests_Analyse/production_29.09.22_22-015715/"

f = []
for (dirpath, dirnames, filenames) in walk(folder):
    f = [file for file in filenames if ".png" in file]
    break
batch_number = np.array([2,2,0,1,5,7,1,5])

for filename in f:
    start = time.time()
    
    first = True if filename.split("_")[1].split(".")[0] == "True" else False
    #first = True if filename.split("_")[1].split("-")[0] == "True" else False
    
    #GET ALL RELEVANT INFORMATION FROM IMAGE
    img = cv.rotate(cv.imread(folder + filename), cv.ROTATE_180)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    status = ocr.analyse_img(img_gray, first, model, batch_number, plot)
    
    if status == "ok":
        passed.append(filename)
    else:
        not_passed.append(filename)
        
        if plot:
            fig, axs = plt.subplots(2)
                
            axs[0].imshow(img_gray,'gray')
            axs[1].text(0,0, filename +"\n" + status)
    
            plt.show()
    
print("ok : ", len(passed)/len(f)*100)
print("rejet : ",len(not_passed)/len(f)*100)


"""
GET ALL THE FILE IN A FOLDER
"""
folder = "C:/Users/LDE/Prog/OCR_detection/Tests_Analyse/production_06.10.22_22-015556/"

f = []
for (dirpath, dirnames, filenames) in walk(folder):
    f = [file for file in filenames if ".png" in file]
    break
batch_number = np.array([2,2,0,1,5,5,5,6])

for filename in f:
    start = time.time()
    
    #first = True if filename.split("_")[1].split(".")[0] == "True" else False
    first = True if filename.split("_")[1].split("-")[0] == "True" else False
    
    #GET ALL RELEVANT INFORMATION FROM IMAGE
    img = cv.rotate(cv.imread(folder + filename), cv.ROTATE_180)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    status = ocr.analyse_img(img_gray, first, model, batch_number, plot)
    
    if status == "ok":
        passed.append(filename)
    else:
        not_passed.append(filename)
        
        if plot:
            fig, axs = plt.subplots(2)
                
            axs[0].imshow(img_gray,'gray')
            axs[1].text(0,0, filename +"\n" + status)
    
            plt.show()
    
print("ok : ", len(passed)/len(f)*100)
print("rejet : ",len(not_passed)/len(f)*100)
    

"""
GET ALL THE FILE IN A FOLDER
"""
folder = "C:/Users/LDE/Prog/OCR_detection/Tests_Analyse/bad_imgs_06.10.22_22-015556/"

f = []
for (dirpath, dirnames, filenames) in walk(folder):
    f = [file for file in filenames if ".png" in file]
    break
batch_number = np.array([2,2,0,1,5,5,5,6])

for filename in f:
    start = time.time()
    
    #first = True if filename.split("_")[1].split(".")[0] == "True" else False
    first = True if filename.split("_")[1].split("-")[0] == "True" else False
    
    #GET ALL RELEVANT INFORMATION FROM IMAGE
    img = cv.rotate(cv.imread(folder + filename), cv.ROTATE_180)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    status = ocr.analyse_img(img_gray, first, model, batch_number, plot)
    
    if status == "ok":
        passed.append(filename)
    else:
        not_passed.append(filename)
        
        if plot:
            fig, axs = plt.subplots(2)
                
            axs[0].imshow(img_gray,'gray')
            axs[1].text(0,0, filename +"\n" + status)
    
            plt.show()
    
print("ok : ", len(passed)/len(f)*100)
print("rejet : ",len(not_passed)/len(f)*100)
    
    
    
    
    
    
    