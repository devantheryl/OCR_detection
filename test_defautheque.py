# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 17:10:16 2022

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
import random
from random import sample

df = pd.DataFrame(data = None,columns = ["number", "type_of_occlusion", "percentage_of_occlusion", 
                             "proba0", "proba1", "proba2", "proba3", "proba4", "proba5", "proba6", "proba7", "proba8", "proba9", "predicted_number",
                             "OCR_score", "file_path"])

defautheque_folder = "C:/Users/LDE/Prog/OCR_detection/Defautheque/"
ref_folder = "C:/Users/LDE/Prog/OCR_detection/number_ref_new/"

params = {"th_quality" : 83,
          "quality_filter_size" : 29,
          "quality_constant" : 2,
          "quality_constrast_norm" : 0
    }
plot = False


"""
LOAD THE DEEP LEARNING MODEL
"""
checkpoint_path = "model/training_real_number_only_4_128_10/cp-0095.ckpt"
# Load the previously saved weights
model = keras.models.load_model(checkpoint_path)


for type_of_occlusion in ["blob", "noise", "row"]:
    print(type_of_occlusion)
    for percentage_of_occlusion in [5,10,15,20,25,30,35,40,45,50]:
        print(percentage_of_occlusion)
        for number in [0,1,2,3,4,5,6,7,8,9]:
            
            
            folder = defautheque_folder + type_of_occlusion + "/" + str(percentage_of_occlusion) + "/" + str(number) + "/"
            
            filenames = []
            for (dirpath, dirnames, filenames) in walk(folder):
                filenames = [file for file in filenames if ".png" in file]
                break
            
            ref = cv.imread(ref_folder + "ref_" + str(number) + ".png" ,0)
            weigths = cv.imread(ref_folder + "weights_" + str(number) + ".png" ,0)
            for f in filenames:
                
                img = cv.imread(folder  + f)
                img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

                
                d1,equ_masked_th = ocr.get_impression_score(img_gray,ref,params["quality_filter_size"],params["quality_constant"],params["quality_constrast_norm"],weigths, plot)    
                
                
                #make the POI exploitable by the deep learning algo
                POI_th = cv.adaptiveThreshold(img_gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,61,10)
                POI_th_blurred = cv.GaussianBlur(POI_th,(3,7),0)
                _,POI_th_blurred_th = cv.threshold(POI_th_blurred,240,255,cv.THRESH_BINARY)
                
                classes, probas = get_number.get_number_from_image_POI(model,{"poi" : POI_th_blurred_th})
                probas = probas.flatten()
                
                if plot:
                    plt.imshow(img_gray,'gray')
                    plt.show()
                    
                    
                    plt.imshow(POI_th_blurred_th,'gray')
                    plt.show()
                
                
                new_row = pd.Series({"number" : number, 
                                     "type_of_occlusion":type_of_occlusion, 
                                     "percentage_of_occlusion":percentage_of_occlusion, 
                                     "proba0" : probas[0],
                                     "proba1" : probas[1],
                                     "proba2" : probas[2],
                                     "proba3" : probas[3],
                                     "proba4" : probas[4],
                                     "proba5" : probas[5],
                                     "proba6" : probas[6],
                                     "proba7" : probas[7],
                                     "proba8" : probas[8],
                                     "proba9" : probas[9],
                                     "predicted_number" : classes[0],
                                     "OCR_score": d1,
                                     "file_path" : folder  + f})
                df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
                
                
df.to_csv("C:/Users/LDE/Prog/OCR_detection/Defautheque/database.csv", sep = ";")