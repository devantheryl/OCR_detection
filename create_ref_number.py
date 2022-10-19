# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 14:24:13 2022

@author: LDE
"""

import os
from matplotlib import pyplot as plt

os.chdir("C:/Users/LDE/Prog/OCR_detection")


import cv2 as cv
from os import walk
import numpy as np
from sklearn.metrics import accuracy_score
from PIL import Image, ImageOps, ImageEnhance
import pandas as pd
import numpy as np


"""
GET ALL THE FILE IN A FOLDER
"""
number = "9"
folder = "C:/Users/LDE/Prog/OCR_detection/dataset_resized_noTh/" + number + "/"

f = []
for (dirpath, dirnames, filenames) in walk(folder):
    f = [file for file in filenames if ".png" in file]
    break

img_tot_mean = np.zeros((70,40))
plot = False
for filename in f:
    
    #GET ALL RELEVANT INFORMATION FROM IMAGE
    img = cv.imread(folder + filename,0)
    img_th = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,29,14)
    
    if plot: 
        plt.imshow(img,'gray')
        plt.show()
        plt.imshow(img_th,'gray')
        plt.show()

    
    img_tot_mean  += img
    
img_tot_mean /= len(f)

plt.imshow(img_tot_mean,'gray')
plt.show()


img_tot_mean = img_tot_mean.astype('u1')
img = cv.adaptiveThreshold(img_tot_mean,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,39,0)
plt.imshow(img,'gray')
plt.show()


img = np.abs(img - 255)*255
directory = "number_ref_new/"
cv.imwrite(directory + "ref_" + number + ".png", img)