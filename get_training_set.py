# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 13:40:26 2022

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


def create_get_dataset(train_test = 0.8):
    # keep in mind that open CV loads images as BGR not RGB
    number = "9"
    folder = "C:/Users/LDE/Prog/OCR_detection/Tests_Analyse/numbers/img_number" + number + "/"
    
    output_dir = "dataset/"
    
    test_nbr = 0
    train_nbr = 0
    img_number = 0
    
    """
    create all datasetDir
    """
    for i in range(10):
        path = output_dir + "train/" + str(i)
        try:
            os.makedirs(path)
        except:
            print ("Creation of the directory %s failed" % path)

        path = output_dir + "val/" + str(i)
        try:
            os.makedirs(path)
        except:
            print ("Creation of the directory %s failed" % path)
    
    
    f = []
    for (dirpath, dirnames, filenames) in walk(folder):
        f.extend(filenames)
        break
    
    for filename in f:
    
        img = cv.rotate(cv.imread(folder + filename), cv.ROTATE_180)
        rectangles,imgs_cropped, img_resized, imgs_th, POIs_total_th, POIs_total_img_resized, POIs_total_img = ocr.find_numbers_positions(img)
        
        
        if filename.split("_")[1] == "True":
            first = "1"
        else:
            first = "2"
            
        
        for i,(key, poi) in enumerate(POIs_total_th.items()):
            
            poi_blurred = cv.GaussianBlur(poi,(3,7),0)
            _,poi_blurred_th = cv.threshold(poi_blurred,240,255,cv.THRESH_BINARY)
            
            x,y,w,h = rectangles[key]
            
            
            filename = str(img_number)+"_"+number + ".png"
                
            rand = np.random.uniform(0, 1)
            if rand < train_test:
                directory = output_dir + "train/" 
                train_nbr +=1
            else:
                directory = output_dir + "val/"
                test_nbr += 1
                    
                
            cv.imwrite(directory + number + "/" + filename, poi_blurred_th)
            print(img_number)
    
            img_number += 1
        
                
    print(train_nbr,test_nbr,img_number)
                
                
create_get_dataset(0.8)
        
    
    
    