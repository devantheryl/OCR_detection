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



def train_model():

    target_size = (32,32)
    batch_size = 32
    
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
    test_generator = ImageDataGenerator(
        rescale = 1./255
        )
    
    train_iterator = train_generator.flow_from_directory(
        "dataset/train",
        target_size= target_size,
        batch_size = batch_size,
        class_mode = "categorical",
        shuffle = True
        )
    
    val_iterator = val_generator.flow_from_directory(
        "dataset/val",
        target_size= target_size,
        batch_size = batch_size,
        class_mode = "categorical",
        shuffle = False
        )
    """
    test_iterator = test_generator.flow_from_directory(
        "dataset/test",
        target_size= target_size,
        batch_size = batch_size,
        class_mode = "categorical",
        shuffle = False
    )
    """
    
    checkpoint_path = "model/training_real_number_only_2/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq = 50)
    
    model = Sequential()
    model.add(ResNet50(include_top = False, pooling = 'avg', weights = 'imagenet'))
    model.add(Dense(512,activation = 'relu'))
    model.add(Dense(10,activation = 'softmax'))
    
    #set resnet layer not trainable
    model.layers[0].trainable = False #layers 0 is the pretrained resnet model
    
    model.summary()
    
    model.compile(optimizer = "Adam", loss = 'categorical_crossentropy', metrics = ["accuracy"])
    
    model.fit(train_iterator, epochs = 400, validation_data=val_iterator, callbacks=[cp_callback])
    
    print(model.evaluate(val_iterator))


    
    


train_model()

checkpoint_path = "model/training_real_number_only_2/cp.ckpt"
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


target_size = (32,32)
batch_size = 64

val_generator = ImageDataGenerator(
        rescale = 1./255
        )

val_iterator = val_generator.flow_from_directory(
        "dataset/val",
        target_size= target_size,
        batch_size = batch_size,
        class_mode = "categorical",
        shuffle = False
        )


# Re-evaluate the model
loss, acc = model.evaluate(val_iterator, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


