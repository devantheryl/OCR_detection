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


# keep in mind that open CV loads images as BGR not RGB
folder = "Tests_Analyse/Numeros_new_police/Valeurs_0a9/"
batch_numbers = []

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

for batch_number in batch_numbers:
    number = batch_number.split("-")[0]
    
    numbers, rectangles,tests, imgs, imgs_th, POIs_total = ocr.find_numbers_positions(folder, batch_number)
    
    imgs_test = imgs.copy()
    
    
    for rectangle in rectangles[0]:
        x,y,w,h = rectangle    
        cv.rectangle(imgs[0],(x,y),(x+w,y+h),(255,255,0),2)
        
    for rectangle in rectangles[1]:
        x,y,w,h = rectangle    
        cv.rectangle(imgs[1],(x,y),(x+w,y+h),(255,255,0),2)
        
    
    
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
    
    """    
    plt.imshow(imgs_test[0],'gray')
    plt.show()
    
    plt.imshow(imgs_test[1],'gray')
    plt.show()
    """
    
    dataset[number].append(POIs_total)