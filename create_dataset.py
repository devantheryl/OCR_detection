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

folder = "C:/Users/LDE/Prog/OCR_detection/Tests_Analyse/bad_imgs_06.10.22_22-015556/"

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

batch_number = np.array([2,2,0,1,5,5,5,5,6])


problem_file = ['10136_False-batch_number_False.png', '11010_False-bad_quality_digit.png', '11059_False-batch_number_False.png', '11201_False-bad_quality_digit.png', '12131_False-bad_quality_digit.png', '12311_False-no_batch_number.png', '12313_False-no_batch_number.png', '12455_False-no_batch_number.png', '12535_False-bad_quality_digit.png', '12599_False-no_batch_number.png', '12935_False-no_batch_number.png', '13151_False-bad_quality_digit.png', '13745_False-no_batch_number.png', '13900_False-batch_number_False.png', '14580_False-bad_quality_digit.png', '14831_False-batch_number_False.png', '16258_False-batch_number_False.png', '16751_False-batch_number_False.png', '18261_False-no_batch_number.png', '18553_False-batch_number_False.png', '18577_False-batch_number_False.png', '19019_False-no_batch_number.png', '19109_False-no_batch_number.png', '19169_False-no_batch_number.png', '192_False-bad_quality_digit.png', '19361_False-no_batch_number.png', '19505_False-no_batch_number.png', '19573_False-batch_number_False.png', '19683_False-batch_number_False.png', '19745_False-no_batch_number.png', '19971_False-no_batch_number.png', '20081_False-no_batch_number.png', '20129_False-no_batch_number.png', '20211_False-bad_quality_digit.png', '20301_False-no_batch_number.png', '20469_False-no_batch_number.png', '20559_False-no_batch_number.png', '20705_False-no_batch_number.png', '20741_False-no_batch_number.png', '21137_False-no_batch_number.png', '2124_False-no_batch_number.png', '21521_False-no_batch_number.png', '21665_False-no_batch_number.png', '21905_False-no_batch_number.png', '22097_False-no_batch_number.png', '22961_False-no_batch_number.png', '23297_False-no_batch_number.png', '23571_False-bad_quality_digit.png', '23967_False-no_batch_number.png', '24021_False-no_batch_number.png', '24143_False-no_batch_number.png', '24203_False-no_batch_number.png', '24278_False-batch_number_False.png', '24280_False-batch_number_False.png', '24304_False-no_batch_number.png', '24400_False-batch_number_False.png', '2442_False-no_batch_number.png', '24544_False-no_batch_number.png', '24740_False-no_batch_number.png', '24824_False-no_batch_number.png', '24976_False-no_batch_number.png', '25024_False-no_batch_number.png', '25231_False-batch_number_False.png', '25456_False-no_batch_number.png', '25536_False-bad_quality_digit.png', '25598_False-no_batch_number.png', '25674_False-bad_quality_digit.png', '25888_False-no_batch_number.png', '25970_False-bad_quality_digit.png', '26566_False-no_batch_number.png', '26888_False-no_batch_number.png', '2689_False-batch_number_False.png', '27048_False-no_batch_number.png', '27468_False-no_batch_number.png', '28556_False-bad_quality_digit.png', '28602_False-batch_number_False.png', '28768_False-bad_quality_digit.png', '29438_False-bad_quality_digit.png', '29674_False-bad_quality_digit.png', '30402_False-bad_quality_digit.png', '30496_False-bad_quality_digit.png', '30502_False-batch_number_False.png', '30544_False-bad_quality_digit.png', '31068_False-no_batch_number.png', '31252_False-bad_quality_digit.png', '31782_False-batch_number_False.png', '31832_False-bad_quality_digit.png', '31882_False-no_batch_number.png', '31966_False-bad_quality_digit.png', '32032_False-bad_quality_digit.png', '32054_False-bad_quality_digit.png', '32060_False-bad_quality_digit.png', '32154_False-bad_quality_digit.png', '32160_False-bad_quality_digit.png', '32198_False-bad_quality_digit.png', '32204_False-bad_quality_digit.png', '32210_False-bad_quality_digit.png', '32216_False-bad_quality_digit.png', '32218_False-no_batch_number.png', '32232_False-bad_quality_digit.png', '32254_False-bad_quality_digit.png', '32282_False-bad_quality_digit.png', '32484_False-bad_quality_digit.png', '32596_False-batch_number_False.png', '32598_False-batch_number_False.png', '366_False-no_batch_number.png', '5396_False-no_batch_number.png', '5872_False-batch_number_False.png', '5912_False-bad_quality_digit.png', '6126_False-batch_number_False.png', '6431_False-batch_number_False.png', '6979_False-no_batch_number.png', '7359_False-batch_number_False.png', '828_False-batch_number_False.png', '9059_False-no_batch_number.png', '9216_False-batch_number_False.png', '9346_False-no_batch_number.png', '9666_False-bad_quality_digit.png']

not_passed = []

for filename in f:
    print(filename)
    start = time.time()
    
    first = True if filename.split("_")[1].split(".")[0] == "True" else False
    first = True if filename.split("_")[1].split("-")[0] == "True" else False
    
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

    
    
    
    
    
    
    
    
    