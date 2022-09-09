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
    folder = "Tests_Analyse/Numeros_new_police/Valeurs_0a9/"
    batch_numbers = []
    
    output_dir = "dataset/"
    
    test_nbr = 0
    train_nbr = 0
    
    """
    create all datasetDir
    """
    for i in range(10):
        path = output_dir + "train/" + str(i)
        try:
            os.makedirs(path)
        except:
            print ("Creation of the directory %s failed" % path)
        path = output_dir + "test/" + str(i)
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
        split = filename.split("_")[1].split(".")[0]
        batch_numbers.append(split)
    
    batch_numbers = [*set(batch_numbers)]
    
    dataset = {}
    for i in range(10):
        dataset[str(i)] = []
    
    img_number = 0
    for batch_number in batch_numbers:
        number = batch_number.split("-")[0]
        
        numbers, rectangles,tests, imgs, imgs_th, POIs_total_th, POIs_total_img,_ = ocr.find_numbers_positions(folder, batch_number)
        
        
        
        imgs_test = imgs.copy()
        
        
        for rectangle in rectangles[0]:
            x,y,w,h = rectangle    
            cv.rectangle(imgs[0],(x,y),(x+w,y+h),(255,255,0),2)
            
        for rectangle in rectangles[1]:
            x,y,w,h = rectangle    
            cv.rectangle(imgs[1],(x,y),(x+w,y+h),(255,255,0),2)
            
        
        """
        plt.imshow(imgs[0],'gray')
        plt.show()
    
        plt.imshow(imgs_th[0],'gray')
        plt.show()
        
        plt.imshow(imgs[1],'gray')
        plt.show()  
    
        plt.imshow(imgs_th[1],'gray')
        plt.show()
        
        
        for rectangle in tests[0]:
            x,y,w,h = rectangle    
            cv.rectangle(imgs_test[0],(x,y),(x+w,y+h),(255,255,0),2)
            
        for rectangle in tests[1]:
            x,y,w,h = rectangle    
            cv.rectangle(imgs_test[1],(x,y),(x+w,y+h),(255,255,0),2)
         
        plt.imshow(imgs_test[0],'gray')
        plt.show()
        
        plt.imshow(imgs_test[1],'gray')
        plt.show()
        """
        
        
        for i, poi in enumerate(POIs_total_th):
            
            for key, value in poi.items():
                filename = number + "/"  + str(img_number) +".png"
                
                rand = np.random.uniform(0, 1)
                if rand < train_test:
                    directory = output_dir + "train/"
                    train_nbr +=1
                else:
                    directory = output_dir + "val/"
                    test_nbr += 1
                    
                
                cv.imwrite(directory + filename, value)
                

                img_number += 1
        
                
    print(train_nbr,test_nbr,img_number)
                
                
create_get_dataset(0.8)
        
    
    
    