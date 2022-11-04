"""
Created on Tue Sep  6 13:34:21 2022

@author: LDE
"""

import os
from matplotlib import pyplot as plt

os.chdir("C:/Users/LDE/Prog/OCR_detection")


import cv2 as cv
from os import walk
import numpy as np
from numpy.linalg import inv
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import time 


ref_folder = "C:/Users/LDE\Prog/OCR_detection/number_ref_new/"
ref_filenames = []
prob_folder_dict = {}

for i in range(10):
    ref_filenames.append("ref_" + str(i) + ".png")
    prob_folder_dict[str(i)] = "C:/Users/LDE/Prog/OCR_detection/dataset_resized_noTh/" + str(i) + "/"


prob_filenames_dict = {}
for key, value in prob_folder_dict.items():
    f = []
    for (dirpath, dirnames, filenames) in walk(value):
        f = [file for file in filenames if ".png" in file]
        break
    prob_filenames_dict[key] = f


    
horizontal_data = {}
vertical_data = {}
sum_distance_data = {}

for filename in ref_filenames:
    
    number = filename[-5]
    horizontal_data[number] = []
    vertical_data[number] = []
    sum_distance_data[number] = []
    
    #compute the signal for the ref
    ref = cv.imread(ref_folder + filename)
    ref_gray = cv.cvtColor(ref, cv.COLOR_BGR2GRAY)
    
    x = np.array(range(np.shape(ref_gray)[1]))
    y = np.array(range(np.shape(ref_gray)[0]))
    
    ref_horizontal_summed = np.sum(ref_gray,axis = 0)/max(np.sum(ref_gray,axis = 0)) * np.shape(ref_gray)[0]
    ref_vertical_summed = np.sum(ref_gray,axis = 1)/max(np.sum(ref_gray,axis = 1)) * np.shape(ref_gray)[1]
    
    prob_folder = prob_folder_dict[number]
    for prob_filename in prob_filenames_dict[number]:
        #compute the signal for the ref
        prob = cv.imread(prob_folder + prob_filename)
        prob_gray = cv.cvtColor(prob, cv.COLOR_BGR2GRAY)
        
        prob_th = cv.adaptiveThreshold(prob_gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,29,2)
        prob_th_not = np.abs(prob_th - 255)
        
        prob_horizontal_summed = np.sum(prob_th_not,axis = 0)/max(np.sum(prob_th_not,axis = 0)) * np.shape(prob_th_not)[0]
        prob_vertical_summed = np.sum(prob_th_not,axis = 1)/max(np.sum(prob_th_not,axis = 1)) * np.shape(prob_th_not)[1]
        
        
        
        horizontal_distance = np.sum(np.abs(ref_horizontal_summed-prob_horizontal_summed))
        vertical_distance = np.sum(np.abs(ref_vertical_summed - prob_vertical_summed))
        
        horizontal_data[number].append(horizontal_distance)
        vertical_data[number].append(vertical_distance)
        
        sum_distance_data[number].append(horizontal_distance + vertical_distance)
        
        if horizontal_distance > 350 or vertical_distance > 500:

            fig, axs = plt.subplots(3)
                    
            axs[0].imshow(ref_gray,'gray')
            axs[1].imshow(prob_th_not,'gray')
            axs[2].text(0,0, "horizontal distance : " + str(horizontal_distance) + "\n\n" + "vertical distance : " + str(vertical_distance))
            
            plt.show()
            print(prob_filename)
            pass
        

    """
    plt.imshow(ref_gray,'gray')
    plt.show()
    
    plt.plot(x, ref_horizontal_summed)
    plt.show()
    
    plt.plot(y, ref_vertical_summed)
    plt.show()
    """
    
    
    
    
    

