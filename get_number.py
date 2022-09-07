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



def get_number_from_image_POI(folder, filename, model):
    # keep in mind that open CV loads images as BGR not RGB
    
   
        
    numbers, rectangles,tests, imgs, imgs_th, POIs_total = ocr.find_numbers_positions(folder, filename)
    
    for i, poi in enumerate(POIs_total):
        
        for key, value in poi.items():
            
            
   
            
            plt.imshow(value,'gray')
            plt.show()
            
            value = cv.resize(value, (32,32), interpolation = cv.INTER_AREA)/255
            value = np.array(value).reshape(-1, 32, 32).astype('float32')
            value = np.stack((value,)*3, axis = -1)
                 
            print(model.predict_classes(value))

checkpoint_path = "model/training_real_number_only_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


latest = tf.train.latest_checkpoint(checkpoint_dir)

# Create a new model instance
model = Sequential()
model.add(ResNet50(include_top = False, pooling = 'avg', weights = 'imagenet'))
model.add(Dense(512,activation = 'relu'))
model.add(Dense(10,activation = 'softmax'))

#set resnet layer not trainable
model.layers[0].trainable = False #layers 0 is the pretrained resnet model

model.summary()

model.compile(optimizer = "Adam", loss = 'categorical_crossentropy', metrics = ["accuracy"])

# Load the previously saved weights
model.load_weights(latest)


get_number_from_image_POI(folder = "Tests_Analyse/Numeros_new_police/Valeurs_0a9/", filename="0-1",model = model)
                