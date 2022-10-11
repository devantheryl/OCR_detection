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

folder = "C:/Users/LDE/Prog/OCR_detection/Tests_Analyse/production_29.09.22_22-015715/"

f = []
for (dirpath, dirnames, filenames) in walk(folder):
    f = [file for file in filenames if ".png" in file]
    break


"""
GO TROUGH ALL THE FILES
"""
write_out = False
img_number = 10000
out_directory = "dataset/production_22.09.22/"
output_dir = "dataset/"

batch_number = np.array([2,2,0,1,5,7,1,5])


problem_file = ['1040_False.png', '1120_False.png', '1462_False.png', '1608_False.png', '282_False.png', '336_False.png', '34_False.png', '358_False.png', '400_False.png', '422_False.png', '432_False.png', '474_False.png', '506_False.png', '564_False.png', '566_False.png', '608_False.png', '624_False.png', '634_False.png', '682_False.png', '928_False.png', '98_False.png']


not_passed = []

for filename in problem_file:
    print(filename)
    start = time.time()
    
    first = True if filename.split("_")[1].split(".")[0] == "True" else False
    #first = True if filename.split("_")[1].split("-")[0] == "True" else False
    
    #GET ALL RELEVANT INFORMATION FROM IMAGE
    img = cv.rotate(cv.imread(folder + filename), cv.ROTATE_180)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    
    #rectangles,imgs_cropped, img_resized, imgs_th, POIs_total_th, POIs_total_img_resized, POIs_total_img = ocr.find_numbers_positions(img)
    
    POIs_total_img_resized,POIs_total_th  = sv.get_POI_intensity(img_gray, True)
    
    
    
    if len(POIs_total_img_resized):
        if write_out:
            poi_to_save = []
            for i,(key,poi) in enumerate(POIs_total_th.items()):
                
                poi_to_save.append(poi)
                
            if first:
                poi_to_save = poi_to_save[:3]
                for number,p in enumerate(poi_to_save):
                    filename = str(batch_number[number]) + "/"  + str(img_number) + ".png"
                        
                    rand = np.random.uniform(0, 1)
                    if rand < 0.8:
                        directory = output_dir + "train/" 
                        img_number +=1
                    else:
                        directory = output_dir + "val/"
                        img_number += 1
                            
                        
                    cv.imwrite(directory  + filename, p)
            else:
                poi_to_save = poi_to_save[-5:]
                for number,p in enumerate(poi_to_save):
        
                    filename = str(batch_number[3+number]) + "/"  + str(img_number) + ".png"
                        
                    rand = np.random.uniform(0, 1)
                    if rand < 0.8:
                        directory = output_dir + "train/" 
                        img_number +=1
                    else:
                        directory = output_dir + "val/"
                        img_number += 1 
                    cv.imwrite(directory  + filename, p)
        
        #predict the classes
        classes, probas = get_number.get_number_from_image_POI(model,POIs_total_th)
        proba_score = [probas[i,classe] for i, classe in enumerate(classes)]
        print(classes)
        
        
        """
        for _,r in rectangles.items():
            img_resized = cv.rectangle(img_resized, (r[0],r[1]), (r[0]+r[2],r[1]+r[3]),(255,0,0),2)
        """
        
        
        if len(classes) >=3:
            
            if first:
                classes_prob = str(classes[:3]).strip("[]")
                batch_number_ref = str(batch_number[:3]).strip("[]")
                iter_dict = list(POIs_total_img_resized)[0:3]
            else:
                classes_prob = str(classes[-5:]).strip("[]")
                batch_number_ref = str(batch_number[-5:]).strip("[]")
                iter_dict = list(POIs_total_img_resized)[-5:]
                
            
            if classes_prob in batch_number_ref:
                batch_number_ok = True
            else:
                batch_number_ok = False
            
                    
            if batch_number_ok == False:
                
                fig, axs = plt.subplots(2)
                
                axs[0].imshow(img_gray,'gray')
                axs[1].text(0,0, filename +"\n"+ str(classes))
        
                plt.show()
                
                not_passed.append(filename)

            for i,key in enumerate(iter_dict):
                
                poi = POIs_total_img_resized[key]
                if first:
                    shape_mean_th_not = cv.imread("number_ref/ref_" + str(batch_number[:3][i]) + ".png",0)
                else:
                    shape_mean_th_not = cv.imread("number_ref/ref_" + str(batch_number[-5:][i]) + ".png",0)   
                
                prob = cv.resize(poi, (40,70), interpolation = cv.INTER_AREA)
                d1,d2,equ_masked_th = ocr.get_impression_score(prob,shape_mean_th_not, False)    
                
                quality_ok = True
                if d1 < 10:
                    fig, axs = plt.subplots(3)
                    axs[0].imshow(poi,'gray')
                    axs[1].text(0,0, str(d1) + "\n" + str(d2) + "\n" + filename)
                    axs[2].imshow(equ_masked_th,'gray')
            
                    plt.show()
                    quality_ok = False
                    
                if not quality_ok:
                    not_passed.append(filename)
            
        else:
            
            fig, axs = plt.subplots(2)
            
            axs[0].imshow(img_gray,'gray')
            axs[1].text(0,0, filename+ "  quality sucks")
    
            plt.show()   
            
            not_passed.append(filename)
            
    else:
        fig, axs = plt.subplots(2)
            
        axs[0].imshow(img_gray,'gray')
        axs[1].text(0,0, filename + "no batch number")

        plt.show() 
        
        not_passed.append(filename)
        
        
print(not_passed)

    
    
    
    
    
    
    
    
    