# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 16:16:39 2022

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
import pandas as pd
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds
import seaborn as sns
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator

data = pd.read_csv("dataset/production_08.09.22/datas.csv", sep = ";").drop(["Unnamed: 0"], axis=1)

data.plot.scatter(x="number", y="proba")

# Convert DataFrame to matrix
mat = data.values
mat[:,0] /= 9
print(mat)
# Using sklearn
km = KMeans(n_clusters=2)
km.fit(mat)
# Get cluster assignment labels
labels = km.labels_
# Format results as a DataFrame
results = pd.DataFrame([labels]).T

results.head()

for i, row in results.iterrows():
    if row.values[0] == 1:
        img = cv.imread("dataset/production_08.09.22/" + str(i)+ ".png",0)
        plt.imshow(img,'gray')
        plt.show()  