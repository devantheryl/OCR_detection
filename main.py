# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 13:34:21 2022

@author: LDE
"""

import os
from matplotlib import pyplot as plt

os.chdir("C:/Users/LDE/Prog/OCR_detection")


import OCR_detection as ocr
import cv2 as cv

# keep in mind that open CV loads images as BGR not RGB
folder = "Tests_Analyse/Conforme_police actuelle/"
batch_number = "28-20220310_130128"


numbers, rectangles,tests, imgs, imgs_th, POIs_total = ocr.find_numbers_positions(folder, batch_number)


imgs_test = imgs.copy()

print(numbers)
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


plt.imshow(imgs_test[0],'gray')
plt.show()

plt.imshow(imgs_test[1],'gray')
plt.show()

