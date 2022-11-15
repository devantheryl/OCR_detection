# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 10:01:15 2022

@author: LDE
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 13:34:21 2022

@author: LDE
"""

import os
from matplotlib import pyplot as plt

os.chdir("C:/Users/LDE/Prog/OCR_detection")


import OCR_detection as ocr
import segment_vials as sv
import get_number
import cv2 as cv
from os import walk
import numpy as np
from sklearn.metrics import accuracy_score
from PIL import Image, ImageOps, ImageEnhance
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds
import seaborn as sns
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
import time 


"""
LOAD THE DEEP LEARNING MODEL
"""


checkpoint_path = "model/training_real_number_only_4_128_10/cp-0095.ckpt"


# Load the previously saved weights
model = keras.models.load_model(checkpoint_path)


"""
GET ALL THE FILE IN A FOLDER
"""
folder = "C:/Users/LDE/Prog/OCR_detection/Tests_Analyse/production_22-015915/"

f = []
for (dirpath, dirnames, filenames) in walk(folder):
    f = [file for file in filenames if ".png" in file]
    break


"""
GO TROUGH ALL THE FILES
"""
write_out = False
img_number = 0
out_directory = "dataset/production_22.09.22/"
output_dir = "dataset_resized_noTh/"

batch_number = np.array([2,2,0,1,5,9,1,5])


problem_file = ['11368_False-quality_problem.png', '11378_False-quality_problem.png', '15249_True-quality_problem.png', '15257_True-quality_problem.png', '15258_False-quality_problem.png', '15266_False-quality_problem.png', '15272_False-quality_problem.png', '22642_False-quality_problem.png', '22646_False-quality_problem.png', '22648_False-quality_problem.png', '22652_False-batch_number False.png', '22656_False-quality_problem.png', '22658_False-quality_problem.png', '22660_False-quality_problem.png', '22662_False-quality_problem.png', '22664_False-quality_problem.png', '22670_False-quality_problem.png', '22672_False-quality_problem.png', '22698_False-quality_problem.png', '22702_False-quality_problem.png', '3077_True-quality_problem.png', '38298_False-quality_problem.png', '38304_False-quality_problem.png', '38306_False-batch_number False.png', '38310_False-quality_problem.png', '38314_False-quality_problem.png', '4012_False-quality_problem.png', '4146_False-quality_problem.png', '5348_False-quality_problem.png', '6420_False-quality_problem.png', '6422_False-quality_problem.png', '6424_False-quality_problem.png']

not_passed = []
passed = []
plot = False
prod_type = 2


params = {"th_quality" : 83,
          "quality_filter_size" : 29,
          "quality_constant" : 2,
          "quality_constrast_norm" : 0
    }


for filename in f:
    print(filename)
    start = time.time()
    
    if "-" in filename:
        first = True if filename.split("_")[1].split("-")[0] == "True" else False
    else:
        first = True if filename.split("_")[1].split(".")[0] == "True" else False
    
    
    #GET ALL RELEVANT INFORMATION FROM IMAGE
    img = cv.rotate(cv.imread(folder + filename), cv.ROTATE_180)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    flipoff_zoi = sv.get_flipoff_zoi(img, True)
    print("color mean : ", np.mean(flipoff_zoi,axis=(0, 1)))
    
    #prod_type = 0 if img full resolution
    #prod_type = 1 if img truncated 
    status, summary = ocr.analyse_img(img_gray, first, model, batch_number, plot, prod_type,params, write_out, img_number,filename, output_dir, False)
    
    
    if status == "ok":
        passed.append((filename, summary))
    else:
        not_passed.append((filename, summary))
        print(status)
        
        if False:
            fig, axs = plt.subplots(2)
                
            axs[0].imshow(img_gray,'gray')
            axs[1].text(0,0, filename +"\n" + status)
    
            plt.show()



print(passed)
print(not_passed)

print("ok : ", len(passed)/len(f)*100)
print("rejet : ",len(not_passed)/len(f)*100)

    
    
    
    
    
    
    
    
    