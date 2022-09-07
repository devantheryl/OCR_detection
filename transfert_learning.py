# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 08:40:25 2022

@author: LDE
"""

import os
os.chdir("C:/Users/LDE/Prog/OCR_detection")



import cv2 as cv

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

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


def change_size(img):
    img = array_to_img(img, scale=False) #returns PIL Image
    img = img.resize((75, 75)) #resize image
    img = img.convert(mode='RGB') #makes 3 channels
    arr = img_to_array(img) #convert back to array
    return arr.astype(np.float64)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = np.array(x_train).reshape(-1, 28, 28).astype('float32')
x_test = np.array(x_test).reshape(-1, 28, 28).astype('float32')

x_train = np.stack((x_train,)*3, axis = -1)
x_test = np.stack((x_test,)*3, axis = -1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)




train_generator = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 40,
    shear_range = 0.2,
    zoom_range = 0.2,
    fill_mode = 'nearest'
    )

val_generator = ImageDataGenerator(
    rescale = 1./255
    )

train_iterator = train_generator.flow(x_train, y_train, batch_size = 512, shuffle = True)
val_iterator = val_generator.flow(x_test,y_test,batch_size = 512, shuffle = False)


model = Sequential()
model.add(ResNet50(include_top = False, pooling = 'avg', weights = 'imagenet'))
model.add(Dense(512,activation = 'relu'))
model.add(Dense(10,activation = 'softmax'))

#set resnet layer not trainable
model.layers[0].trainable = False #layers 0 is the pretrained resnet model

model.summary()

model.compile(optimizer = "Adam", loss = 'categorical_crossentropy', metrics = ["accuracy"])

model.fit(train_iterator, epochs = 10, validation_data=val_iterator)


