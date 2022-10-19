# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 10:35:16 2022

@author: LDE
"""

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


FP = []
FN = []

"""
LOAD THE DEEP LEARNING MODEL
"""
checkpoint_path = "model/training_real_number_only_4_128_10/cp-0095.ckpt"
# Load the previously saved weights
model = keras.models.load_model(checkpoint_path)


""""""""""""""""""
"""BAD ANALYSES"""
""""""""""""""""""
"""
GET ALL THE FILE IN A FOLDER
"""
folder = "C:/Users/LDE/Prog/OCR_detection/Tests_Analyse/non-conforme/"

f = {'10_False.png' : np.array([6,6,6,6,6,6,6,6]),
 '11_True.png' : np.array([7,7,7,7,7,7,7,7]),
 '12_False.png' : np.array([8,8,8,8,8,8,8,8]),
 '13_True.png' : np.array([9,9,9,9,9,9,9,9]),
 '14_False.png' : np.array([8,8,8,8,8,8,8,8]),
 '15_True - Copie.png' : np.array([8,8,8,8,8,8,8,8]),
 '15_True.png' : np.array([8,8,8,8,8,8,8,8]),
 '16_False.png' : np.array([8,8,8,8,8,8,8,8]),
 '17_True.png' : np.array([8,8,8,8,8,8,8,8]),
 '18_False - Copie (2).png' : np.array([8,8,8,8,8,8,8,8]),
 '18_False - Copie (3).png' : np.array([8,8,8,8,8,8,8,8]),
 '18_False.png' : np.array([8,8,8,8,8,8,8,8]),
 '2_False_.png' : np.array([1,1,1,1,1,1,1,1]),
 '300_False.png' : np.array([2,2,0,1,5,0,1,5]),
 '302_False.png' : np.array([2,2,0,1,5,1,1,5]),
 '303_True.png' : np.array([2,2,2,1,5,0,1,5]),
 '304_False.png' : np.array([2,2,0,1,5,3,1,5]),
 '306_False.png' : np.array([2,2,0,1,5,4,1,5]),
 '312_False.png' : np.array([2,2,0,1,5,5,1,5]),
 '313_True.png' : np.array([2,2,6,1,5,0,1,5]),
 '325_True.png' : np.array([2,2,7,1,5,0,1,5]),
 '326_False.png' : np.array([2,2,0,1,5,8,1,5]),
 '338_False.png' : np.array([2,2,0,1,5,9,1,5]),
 '3_True.png' : np.array([0,0,0,0,0,0,0,0]),
 '5_True.png' : np.array([2,2,2,2,2,2,2,2]),
 '6_False.png' : np.array([2,2,2,2,2,2,2,2]),
 '7_True.png' : np.array([3,3,3,3,3,3,3,3]),
 '8_False.png' : np.array([4,4,4,4,4,4,4,4]),
 '9_True.png' : np.array([5,5,5,5,5,5,5,5]),
 'TEST2_False.png' : np.array([2,2,0,1,5,6,7,6]),
 'TEST_True.png' : np.array([2,2,0,1,5,6,7,6])}

"""
GO TROUGH ALL THE FILES
"""
ok_all = []
reject_all = []
ALL_FILE = []

not_passed = []
passed = []
times_all = []
times_max = []
plot = False

for filename,batch_number in f.items():

    start = time.time()
    
    if "-" in filename:
        first = True if filename.split("_")[1].split("-")[0] == "True" else False
    else:
        first = True if filename.split("_")[1].split(".")[0] == "True" else False
    
    #GET ALL RELEVANT INFORMATION FROM IMAGE
    img = cv.rotate(cv.imread(folder + filename), cv.ROTATE_180)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    #prod_type = 0 if img full resolution
    #prod_type = 1 if img truncated 
    status = ocr.analyse_img(img_gray, first, model, batch_number, plot, 0)
    
    if status == "ok":
        FP.append(img)
        passed.append(filename)
    else:
        not_passed.append(filename)
        
        if plot:
            fig, axs = plt.subplots(2)
                
            axs[0].imshow(img_gray,'gray')
            axs[1].text(0,0, filename +"\n" + status)
    
            plt.show()
            
print(folder)
print("ok : ", len(passed)/len(f)*100)
print("rejet : ",len(not_passed)/len(f)*100)


ok_all.append(len(passed)/len(f)*100)
reject_all.append(len(not_passed)/len(f)*100)


#22-015556
folder = "C:/Users/LDE/Prog/OCR_detection/Tests_Analyse/bad_production_06.10.22_22-015556/"
f = []
for (dirpath, dirnames, filenames) in walk(folder):
    f = [file for file in filenames if ".png" in file]
    break
batch_number = np.array([2,2,0,1,5,5,5,6])
ALL_FILE.append((folder,f,batch_number,0))

#22-015778
folder = "C:/Users/LDE/Prog/OCR_detection/Tests_Analyse/bad_production_14.10.22-22-015778/"
f = []
for (dirpath, dirnames, filenames) in walk(folder):
    f = [file for file in filenames if ".png" in file]
    break
batch_number = np.array([2,2,0,1,5,7,7,8])
ALL_FILE.append((folder,f,batch_number,0))

#22-015818
folder = "C:/Users/LDE/Prog/OCR_detection/Tests_Analyse/bad_production_17.10.22-22-015818/"
f = []
for (dirpath, dirnames, filenames) in walk(folder):
    f = [file for file in filenames if ".png" in file]
    break
batch_number = np.array([2,2,0,1,5,8,1,8])
ALL_FILE.append((folder,f,batch_number,0))


#22-015675
folder = "C:/Users/LDE/Prog/OCR_detection/Tests_Analyse/bad_production_22.09.22_22-015675/"
f = []
for (dirpath, dirnames, filenames) in walk(folder):
    f = [file for file in filenames if ".png" in file]
    break
batch_number = np.array([2,2,0,1,5,6,7,5])
ALL_FILE.append((folder,f,batch_number,1))

#22-015716
folder = "C:/Users/LDE/Prog/OCR_detection/Tests_Analyse/bad_production_26.09.22_22-015716/"
f = []
for (dirpath, dirnames, filenames) in walk(folder):
    f = [file for file in filenames if ".png" in file]
    break
batch_number = np.array([2,2,0,1,5,7,1,6])
ALL_FILE.append((folder,f,batch_number,1))

#22-015676
folder = "C:/Users/LDE/Prog/OCR_detection/Tests_Analyse/bad_production_27.09.22_22-015676/"
f = []
for (dirpath, dirnames, filenames) in walk(folder):
    f = [file for file in filenames if ".png" in file]
    break
batch_number = np.array([2,2,0,1,5,6,7,6])
ALL_FILE.append((folder,f,batch_number,0))

#22-015715
folder = "C:/Users/LDE/Prog/OCR_detection/Tests_Analyse/bad_production_29.09.22_22-015715/"
f = []
for (dirpath, dirnames, filenames) in walk(folder):
    f = [file for file in filenames if ".png" in file]
    break
batch_number = np.array([2,2,0,1,5,7,1,5])
ALL_FILE.append((folder,f,batch_number,0))



passed_all = []
not_passed_all = []

for descr in ALL_FILE:
    folder = descr[0]
    f = descr[1]
    batch_number = descr[2]
    prod_type = descr[3]
    print(folder)
    
    not_passed = []
    passed = []
    plot = False
    times = []
    for filename in f:
        start = time.time()
        
        if "-" in filename:
            first = True if filename.split("_")[1].split("-")[0] == "True" else False
        else:
            first = True if filename.split("_")[1].split(".")[0] == "True" else False
        
        #GET ALL RELEVANT INFORMATION FROM IMAGE
        img = cv.rotate(cv.imread(folder + filename), cv.ROTATE_180)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        #prod_type = 0 if img full resolution
        #prod_type = 1 if img truncated 
        status = ocr.analyse_img(img_gray, first, model, batch_number, plot, prod_type)
        
        if status == "ok":
            FP.append(img)
            passed.append(filename)
            passed_all.append(filename)
        else:
            not_passed.append(filename)
            not_passed_all.append(filename)
            
            if plot:
                fig, axs = plt.subplots(2)
                    
                axs[0].imshow(img_gray,'gray')
                axs[1].text(0,0, filename +"\n" + status)
        
                plt.show()
        times.append(time.time()-start)
        
    
    print("ok : ", len(passed)/len(f)*100)
    print("rejet : ",len(not_passed)/len(f)*100)
    ok_all.append(len(passed)/len(f)*100)
    reject_all.append(len(not_passed)/len(f)*100)
    times_all.append((np.mean(times), np.std(times)))
    times_max.append(max(times))





print("ok all     : ",ok_all)
print("reject all : ",reject_all)
print("mean time  : ", times_all)
print("max time   : ",times_max)

print("number images analysed : " , len(passed_all) + len(not_passed_all))

print("total passed           :" ,len(passed_all)/(len(passed_all) + len(not_passed_all))*100, "%")
print("total not passed       :" ,len(not_passed_all)/(len(passed_all) + len(not_passed_all))*100, "%")

for img in FP:
    plt.imshow(img,'gray')
    plt.show()
    
#wait = input("tape enter")



"""
GOOD ANALYSES
"""
passed_all = []
not_passed_all = []
ok_all = []
reject_all = []
ALL_FILE = []

not_passed = []
passed = []
times_all = []
times_max = []


"""
GET ALL THE FILE IN ALL FOLDER
"""
#22-015556
folder = "C:/Users/LDE/Prog/OCR_detection/Tests_Analyse/production_06.10.22_22-015556/"
f = []
for (dirpath, dirnames, filenames) in walk(folder):
    f = [file for file in filenames if ".png" in file]
    break
batch_number = np.array([2,2,0,1,5,5,5,6])
ALL_FILE.append((folder,f,batch_number,0))

#22-015778
folder = "C:/Users/LDE/Prog/OCR_detection/Tests_Analyse/production_14.10.22-22-015778/"
f = []
for (dirpath, dirnames, filenames) in walk(folder):
    f = [file for file in filenames if ".png" in file]
    break
batch_number = np.array([2,2,0,1,5,7,7,8])
ALL_FILE.append((folder,f,batch_number,0))

#22-015818
folder = "C:/Users/LDE/Prog/OCR_detection/Tests_Analyse/production_17.10.22-22-015818/"
f = []
for (dirpath, dirnames, filenames) in walk(folder):
    f = [file for file in filenames if ".png" in file]
    break
batch_number = np.array([2,2,0,1,5,8,1,8])
ALL_FILE.append((folder,f,batch_number,0))


#22-015675
folder = "C:/Users/LDE/Prog/OCR_detection/Tests_Analyse/production_22.09.22_22-015675/"
f = []
for (dirpath, dirnames, filenames) in walk(folder):
    f = [file for file in filenames if ".png" in file]
    break
batch_number = np.array([2,2,0,1,5,6,7,5])
ALL_FILE.append((folder,f,batch_number,1))

#22-015716
folder = "C:/Users/LDE/Prog/OCR_detection/Tests_Analyse/production_26.09.22_22-015716/"
f = []
for (dirpath, dirnames, filenames) in walk(folder):
    f = [file for file in filenames if ".png" in file]
    break
batch_number = np.array([2,2,0,1,5,7,1,6])
ALL_FILE.append((folder,f,batch_number,1))

#22-015676
folder = "C:/Users/LDE/Prog/OCR_detection/Tests_Analyse/production_27.09.22_22-015676/"
f = []
for (dirpath, dirnames, filenames) in walk(folder):
    f = [file for file in filenames if ".png" in file]
    break
batch_number = np.array([2,2,0,1,5,6,7,6])
ALL_FILE.append((folder,f,batch_number,0))

#22-015715
folder = "C:/Users/LDE/Prog/OCR_detection/Tests_Analyse/production_29.09.22_22-015715/"
f = []
for (dirpath, dirnames, filenames) in walk(folder):
    f = [file for file in filenames if ".png" in file]
    break
batch_number = np.array([2,2,0,1,5,7,1,5])
ALL_FILE.append((folder,f,batch_number,0))

#34-899843
folder = "C:/Users/LDE/Prog/OCR_detection/Tests_Analyse/production_17.10.22-34-899843/"
f = []
for (dirpath, dirnames, filenames) in walk(folder):
    f = [file for file in filenames if ".png" in file]
    break
batch_number = np.array([3,4,8,9,9,8,4,3])
ALL_FILE.append((folder,f,batch_number,2))




for descr in ALL_FILE:
    folder = descr[0]
    f = descr[1]
    batch_number = descr[2]
    prod_type = descr[3]
    print(folder)
    
    not_passed = []
    passed = []
    plot = False
    times = []
    for filename in f:
        start = time.time()
        
        if "-" in filename:
            first = True if filename.split("_")[1].split("-")[0] == "True" else False
        else:
            first = True if filename.split("_")[1].split(".")[0] == "True" else False
        
        #GET ALL RELEVANT INFORMATION FROM IMAGE
        img = cv.rotate(cv.imread(folder + filename), cv.ROTATE_180)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        #prod_type = 0 if img full resolution
        #prod_type = 1 if img truncated 
        status = ocr.analyse_img(img_gray, first, model, batch_number, plot, prod_type)
        
        if status == "ok":
            passed.append(filename)
            passed_all.append(filename)
        else:
            FN.append(img)
            not_passed.append(filename)
            not_passed_all.append(filename)
            
            if plot:
                fig, axs = plt.subplots(2)
                    
                axs[0].imshow(img_gray,'gray')
                axs[1].text(0,0, filename +"\n" + status)
        
                plt.show()
        times.append(time.time()-start)
        
    
    print("ok : ", len(passed)/len(f)*100)
    print("rejet : ",len(not_passed)/len(f)*100)
    ok_all.append(len(passed)/len(f)*100)
    reject_all.append(len(not_passed)/len(f)*100)
    times_all.append((np.mean(times), np.std(times)))
    times_max.append(max(times))





print("ok all     : ",ok_all)
print("reject all : ",reject_all)
print("mean time  : ", times_all)
print("max time   : ",times_max)

print("number images analysed : " , len(passed_all) + len(not_passed_all))

print("total passed           :" ,len(passed_all)/(len(passed_all) + len(not_passed_all))*100, "%")
print("total not passed       :" ,len(not_passed_all)/(len(passed_all) + len(not_passed_all))*100, "%") 





for img in FN:
    plt.imshow(img,'gray')
    plt.show()
    
    
    
    
    
    