# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 07:49:41 2022

@author: LDE
"""

import OCR_detection as ocr
import cv2 as cv
import random 
import numpy as np
import os
from matplotlib import pyplot as plt
from os import walk
import pandas as pd



def get_number_df():
    
    folder = "C:/Users/LDE/Prog/OCR_detection/Tests_Analyse/numbers/img_number" 
    numbers = ["0","1","2","3","4","5","6","7","8","9"]
    columns = ["img", "number", "first","position", "posx","posy","width", "height"]


    df = pd.DataFrame(columns = columns)
    
    
    for number in numbers:
        filename = []
        for (dirpath, dirnames, filenames) in walk(folder + number):
            filename = [f for f in filenames if ".png" in f]
            break
            
        for f in filename:
            img = cv.rotate
            img = cv.rotate(cv.imread(folder + number + "/" + f), cv.ROTATE_180)
            rectangles,imgs_cropped, img_resized, imgs_th, POIs_total_th, POIs_total_img_resized, POIs_total_img = ocr.find_numbers_positions(img)
            for i,(key,poi) in enumerate(POIs_total_img_resized.items()):
                
                x,y,w,h = rectangles[key]
                first = f.split("_")[1].split(".")[0]
                first = True if first == "True" else False
                d = {"img":[poi],"number" : number, "first" : first,"position" : i, "posx" : x,"posy":y,"width":w,"height": h}
                new_row = pd.Series(data = d, index = columns)
                df = pd.concat([df,new_row.to_frame().T],ignore_index = True)
    return df

      
            
def get_mean_shape(query,img_shape):
    

    shape_mean = np.zeros((img_shape[1],img_shape[0]))
    
    for i,(index, row) in enumerate(query.iterrows()):
        
        number = row["number"]
        poi = row["img"][0]
        poi_resized = cv.resize(poi, img_shape, interpolation = cv.INTER_AREA)
        
        poi_th = cv.adaptiveThreshold(poi_resized,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,51,10)
        poi_th_blurred = cv.GaussianBlur(poi_th,(1,5),0)
        _,poi_th_blurred_th = cv.threshold(poi_th_blurred,240,255,cv.THRESH_BINARY)
        
        shape_mean += poi_resized
        

    shape_mean /= len(query)
    shape_mean = shape_mean.astype('uint8')
    plt.imshow(shape_mean,'gray')
    plt.show()
    
    
    _, shape_mean_th = cv.threshold(shape_mean,90,255,cv.THRESH_BINARY)
    
    shape_mean_th_not = np.abs(shape_mean_th - 255)*255

    
    plt.imshow(shape_mean_th_not,'gray')
    plt.show()
    
    directory = "number_ref/"
    #cv.imwrite(directory + "ref_" + str(number) + ".png", shape_mean_th_not)
    
    return shape_mean_th_not


"""
img_shape = (40,70)
df = get_number_df()
number = "0"
query = df[((df.number == number))]  #& (df["first"] == True))]     
shape_mean_th_not = cv.imread("number_ref/ref_" + number + ".png",0)
    
for i,(index, row) in enumerate(query.iterrows()):
    prob = cv.resize(row["img"][0], img_shape, interpolation = cv.INTER_AREA)
    ocr.get_impression_score(prob,shape_mean_th_not, True)         
"""


