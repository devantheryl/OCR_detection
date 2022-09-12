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
folder = "Tests_Analyse/production_08.09.22/"

f = []
for (dirpath, dirnames, filenames) in walk(folder):
    f.extend(filenames)
    break


"""
GO TROUGH ALL THE FILES
"""
problem_filename = []

datas = pd.DataFrame(data = None, columns=("number", "proba", "stats0", "stats1", "stats2", "stats3"))
img_index = 0
out_directory = "dataset/production_08.09.22/"

for filename in f:
    
    img_number = filename.split("img")[1][0]
    #GET ALL RELEVANT INFORMATION FROM IMAGE
    
    numbers, rectangles,imgs_cropped, imgs, imgs_th, POIs_total_th, POIs_total_img_resized, POIs_total_img = ocr.find_numbers_positions(folder, filename, img_number)
    
    #predict the classes
    classes, probas = get_number.get_number_from_image_POI(model,POIs_total_th)
    
    
    batch_number = classes
    
    proba_score = [probas[i,classe] for i, classe in enumerate(classes)]
    
    maskeds, masks = quality.get_masked_POI(POIs_total_img)
    
    
    stats_0 = []
    stats_1 = []
    stats_2 = []
    stats_3 = []
    
    for i,(_, poi) in enumerate(POIs_total_img.items()):
            
        ret3,th = cv.threshold(poi,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        th = cv.resize(th, (32,32), interpolation = cv.INTER_AREA)/255
        #plt.imshow(th,'gray')
        #plt.show()  
        
        th_no_white = th[th<1]
        th_no_white_no_black = th_no_white[th_no_white>0]
        
        stats_0.append(np.mean((th_no_white)/np.mean(th_no_white_no_black)))
        stats_1.append(len(th_no_white)/(32*32))
        stats_2.append(len(th_no_white_no_black)/(32*32))
        stats_3.append(np.mean(th_no_white_no_black))
        
        data = pd.DataFrame({"number" :batch_number[i],"proba": proba_score[i],"stats0": stats_0[-1],"stats1":stats_1[-1],"stats2" : stats_2[-1],"stats3":stats_3[-1]},index = [img_index])
        datas = pd.concat([datas,data], sort = False)
        
        cv.imwrite(out_directory + str(img_index) + ".png", poi)
        img_index+=1
        
    score_impression = 1-np.array(stats_0)
    
    text = "BATCH NUMBER : "  + str(batch_number) + "\n"
    text += "PROBA SCORE : " +  str(proba_score) + "\n"
    text += "IMPRESSION SCORE : " + str(score_impression) + "\n   , mean : "+ str(np.mean(score_impression)) + "\n"
    text += "NO WHITE : " + str(stats_1) + "\n"
    text += "NO WHITE AND BLACK: " + str(stats_2) + "\n"
    
    
    
    if (score_impression < 0.48).any() or np.mean(score_impression) < 0.55:
          
        text += "test not passed"
        fig, axs = plt.subplots(2)
        
        axs[0].imshow(imgs_cropped,'gray')
        axs[1].text(0,0, text)
        plt.show() 
        

        
    else:
        
        text += "test passed"
        
        fig, axs = plt.subplots(2)
        
        axs[0].imshow(imgs_cropped,'gray')
        axs[1].text(0,0, text)
        plt.show() 


    
    
datas.to_csv(out_directory + "datas.csv", sep=";")       
    

print(problem_filename)
    
    
    
    
    
    