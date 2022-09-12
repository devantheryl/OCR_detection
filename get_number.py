# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 16:36:55 2022

@author: LDE
"""

import os
from matplotlib import pyplot as plt
from os import walk


os.chdir("C:/Users/LDE/Prog/OCR_detection")

import OCR_detection as ocr
import cv2 as cv
import random 
import numpy as np





from sklearn.metrics import accuracy_score
from PIL import Image, ImageOps, ImageEnhance

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds
import seaborn as sns
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator



def get_number_from_image_POI(model,POIs):
    # keep in mind that open CV loads images as BGR not RGB
    nbr_POIs = len(POIs)

    batch = np.zeros((nbr_POIs,32,32,3))
    i = 0

        
    for key, value in (POIs.items()):
        
        
        value = cv.resize(value, (32,32), interpolation = cv.INTER_AREA)/255
        value = np.array(value).reshape(-1, 32, 32).astype('float32')
        value = np.stack((value,)*3, axis = -1)
        
        batch[i] = value
        
        i+=1
    
    predictions = model.predict(batch)

            
    return np.argmax(predictions,axis = 1), predictions

    


                