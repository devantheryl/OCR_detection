# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 16:00:17 2022

@author: LDE
"""

import cv2 as cv
import sys
import os
os.chdir("C:/Users/LDE/Prog/OCR_detection")
from matplotlib import pyplot as plt
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import collections
import numpy as np
import graph_utils as Graph
from os import walk



folder = "Tests_Analyse/temp/"
filename="img2_8-20220830_132140.png"
output = "Tests_Analyse/production_08.09.22/"

f = []
for (dirpath, dirnames, filenames) in walk(folder):
    f.extend(filenames)
    break

i = 0
for filename in f:
    img = cv.imread(folder + filename)
    if (img == 0).all() == False:
        plt.imshow(img,'gray')
        plt.show()
        i+=1
        cv.imwrite(output + filename, img)

print(i)