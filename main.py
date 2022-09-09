# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 13:34:21 2022

@author: LDE
"""

import os
from matplotlib import pyplot as plt

os.chdir("C:/Users/LDE/Prog/OCR_detection")


import OCR_detection as ocr
import detect_quality as quality
import get_number
import cv2 as cv
from os import walk
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


"""
LOAD THE DEEP LEARNING MODEL
"""
checkpoint_path = "model/training_real_number_only_1/cp.ckpt"
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
folder = "Tests_Analyse/dataset/train/0"

f = []
for (dirpath, dirnames, filenames) in walk(folder):
    f.extend(filenames)
    break


"""
GO TROUGH ALL THE FILES
"""
for filename in f:
    
    filename = filename.split(".")[0].split("img")[1][2:]
    #GET ALL RELEVANT INFORMATION FROM IMAGE
    numbers, rectangles,imgs_cropped, imgs, imgs_th, POIs_total_th, POIs_total_img_resized, POIs_total_img = ocr.find_numbers_positions(folder, filename)
    
    #predict the classes
    classes, probas = get_number.get_number_from_image_POI(model,POIs_total_th)
    
    #compute the batch_number, TO CHANGE TO BE MORE GENERIC
    nbr_digit1 = len(POIs_total_img_resized[0])
    number1 = classes[:nbr_digit1]
    number2 = classes[-(8-nbr_digit1):]
    batch_number = np.concatenate([number1,number2])
    
    proba_score = np.mean([probas[i,classe] for i, classe in enumerate(classes)])
    
    maskeds, masks = quality.get_masked_POI(POIs_total_img)
    
    
    for i, masked in enumerate(maskeds):
        plt.imshow(masked,'gray')
        plt.show() 
        plt.imshow(masks[i],'gray')
        plt.show() 

    
        
    score_impression = 0
    
    print("BATCH NUMBER : ", batch_number)
    print("PROBA SCORE : ", proba_score)
    print("IMPRESSION SCORE : ", score_impression)
    
    for img in imgs_cropped:
        plt.imshow(img,'gray')
        plt.show() 
    
    
    
    
    
    
    
    
    
    
    