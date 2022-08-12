# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 13:57:23 2022

@author: LDE
"""

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
import cv2 as cv
import sys
import os
from matplotlib import pyplot as plt
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import collections


os.chdir("C:/Users/LDE/Prog/OCR_detection")

# keep in mind that open CV loads images as BGR not RGB
folder = "Tests_Analyse/Conforme_police actuelle/"
batch_number = "28-20220310_130128"
filename1 = folder + "img1_" + batch_number + ".png"
filename2 = folder + "img2_" + batch_number + ".png"

img = cv.imread(filename2)





h_cropped = int(img.shape[0] * 0.5)

img  = img[h_cropped:, ]

scale_percent = 40 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
img = cv.resize(img, dim, interpolation = cv.INTER_AREA)



gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


th = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,301,20)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (1, 5))
th = cv.morphologyEx(th, cv.MORPH_OPEN, kernel)

th = cv.bitwise_not(th)



cnts = cv.findContours(th.copy(), cv.RETR_EXTERNAL,
	cv.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(cnts)
digitCnts = []

POIs = {}
# loop over the digit area candidates
for c in cnts:
	# compute the bounding box of the contour
    (x, y, w, h) = cv.boundingRect(c)
    
	# if the contour is sufficiently large, it must be a digit
    if w >= 40 and (h >= 60):
        digitCnts.append((c[0,0,0],c[0,0,1]))
        cv.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        
        POI = cv.bitwise_not(gray[y:y+h, x:x+w])
        
        POIs[x] = POI

POIs = collections.OrderedDict(sorted(POIs.items()))


numbers = []
for k,v in POIs.items():
    text = pytesseract.image_to_string(v,config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
    if text == "":
        numbers.append("00")
    else:
        numbers.append(text[0])
    cv.imshow('Contours', v)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
print(numbers)
    




            
            

