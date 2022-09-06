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
folder = "Tests_Analyse/Production_17.03.22/"
batch_number = "1-20220317_084100"


numbers, rectangles, imgs, imgs_th, POIs_total = ocr.find_numbers_positions(folder, batch_number)

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


