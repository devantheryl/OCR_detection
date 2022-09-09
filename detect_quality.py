# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 09:03:35 2022

@author: LDE
"""

import cv2 as cv
import sys
import os
os.chdir("C:/Users/LDE/Prog/OCR_detection")
from os import walk
from matplotlib import pyplot as plt
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import collections
import numpy as np
import graph_utils as Graph



import OCR_detection as ocr





def get_masked_POI(POIs_resized):
    
    maskeds = []
    masks = []
    for POIs in POIs_resized:
        for key, POI in POIs.items():
            
            
            blur = cv.GaussianBlur(POI,(9,9),0)
            ret3,th = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        
            
            masked = POI.copy()
            masked[th == 255] = 255
            
            maskeds.append(masked)
            masks.append(th)
            
            
           
            
    return maskeds, masks

