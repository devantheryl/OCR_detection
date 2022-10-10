# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 10:35:16 2022

@author: LDE
"""

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

folder = "C:/Users/LDE/Prog/OCR_detection/Tests_Analyse/non-conforme/"

f = {'10_False.png' : np.array([6,6,6,6,6,6,6,6]),
 '11_True.png' : np.array([7,7,7,7,7,7,7,7]),
 '12_False.png' : np.array([8,8,8,8,8,8,8,8]),
 '13_True.png' : np.array([9,9,9,9,9,9,9,9]),
 '14_False.png' : np.array([8,8,8,8,8,8,8,8]),
 '15_True - Copie.png' : np.array([8,8,8,8,8,8,8,8]),
 '15_True.png' : np.array([8,8,8,8,8,8,8,8]),
 '16_False.png' : np.array([8,8,8,8,8,8,8,8]),
 '17_True.png' : np.array([8,8,8,8,8,8,8,8]),
 '18_False - Copie (2).png' : np.array([8,8,8,8,8,8,8,8]),
 '18_False - Copie (3).png' : np.array([8,8,8,8,8,8,8,8]),
 '18_False.png' : np.array([8,8,8,8,8,8,8,8]),
 '2_False_.png' : np.array([1,1,1,1,1,1,1,1]),
 '300_False.png' : np.array([2,2,0,1,5,0,1,5]),
 '302_False.png' : np.array([2,2,0,1,5,1,1,5]),
 '303_True.png' : np.array([2,2,2,1,5,0,1,5]),
 '304_False.png' : np.array([2,2,0,1,5,3,1,5]),
 '306_False.png' : np.array([2,2,0,1,5,4,1,5]),
 '312_False.png' : np.array([2,2,0,1,5,5,1,5]),
 '313_True.png' : np.array([2,2,6,1,5,0,1,5]),
 '325_True.png' : np.array([2,2,7,1,5,0,1,5]),
 '326_False.png' : np.array([2,2,0,1,5,8,1,5]),
 '338_False.png' : np.array([2,2,0,1,5,9,1,5]),
 '3_True.png' : np.array([0,0,0,0,0,0,0,0]),
 '5_True.png' : np.array([2,2,2,2,2,2,2,2]),
 '6_False.png' : np.array([2,2,2,2,2,2,2,2]),
 '7_True.png' : np.array([3,3,3,3,3,3,3,3]),
 '8_False.png' : np.array([4,4,4,4,4,4,4,4]),
 '9_True.png' : np.array([5,5,5,5,5,5,5,5]),
 'TEST2_False.png' : np.array([2,2,0,1,5,6,7,6]),
 'TEST_True.png' : np.array([2,2,0,1,5,6,7,6])}

"""
GO TROUGH ALL THE FILES
"""


batch_number = np.array([2,2,0,1,5,7,1,5])


problem_file = ["6849_False.png"]

for filename,batch_number in f.items():
    print(filename)
    start = time.time()
    
    first = True if filename.split("_")[1].split(".")[0] == "True" else False
    
    #GET ALL RELEVANT INFORMATION FROM IMAGE
    img = cv.rotate(cv.imread(folder + filename), cv.ROTATE_180)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    
    #rectangles,imgs_cropped, img_resized, imgs_th, POIs_total_th, POIs_total_img_resized, POIs_total_img = ocr.find_numbers_positions(img)
    
    POIs_total_img_resized,POIs_total_th  = sv.get_POI_intensity(img_gray, True)
    
    
    
    if len(POIs_total_img_resized):

        
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
            else:
                print("ok")
            
            
            for i,key in enumerate(iter_dict):
                
                poi = POIs_total_img_resized[key]
                if first:
                    shape_mean_th_not = cv.imread("number_ref/ref_" + str(batch_number[:3][i]) + ".png",0)
                else:
                    shape_mean_th_not = cv.imread("number_ref/ref_" + str(batch_number[-5:][i]) + ".png",0)   
                
                prob = cv.resize(poi, (40,70), interpolation = cv.INTER_AREA)
                d1,d2,equ_masked_th = ocr.get_impression_score(prob,shape_mean_th_not, False)    
                
                if d1 < 79:
                    fig, axs = plt.subplots(3)
                    axs[0].imshow(poi,'gray')
                    axs[1].text(0,0, str(d1) + "\n" + str(d2) + "\n" + filename)
                    axs[2].imshow(equ_masked_th,'gray')
            
                    plt.show()
               
                
            
 

            
        else:
            
            fig, axs = plt.subplots(2)
            
            axs[0].imshow(img_gray,'gray')
            axs[1].text(0,0, filename+ "  quality sucks")
    
            plt.show()   
            
    else:
        fig, axs = plt.subplots(2)
            
        axs[0].imshow(img_gray,'gray')
        axs[1].text(0,0, filename + "no batch number")

        plt.show()   
        

    
    
    
    
    
    
    
    
    