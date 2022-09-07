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
import numpy as np
import graph_utils as Graph

os.chdir("C:/Users/LDE/Prog/OCR_detection")

def intersects(box1, box2):
    
    xa1 = box1[0]
    ya1 = box1[1]
    xa2 = box1[0] + box1[2]
    ya2 = box1[1] + box1[3]
    
    xb1 = box2[0]
    yb1 = box2[1]
    xb2 = box2[0] + box2[2]
    yb2 = box2[1] + box2[3]
    
    return not (xa2 < xb1 or xa1 >xb2 or ya1 > yb2 or ya2 < yb1)




def find_numbers_positions(folder, batch_number):
    total_numbers = []
    imgs = []
    imgs_th = []
    merged_rectangles = {0:[],1:[]}
    POIs_total = []
    test_rectangles = []
    
    
    for i in range(2):
        img_number = i+1
        filename = folder + "img" + str(img_number) + "_" + batch_number + ".png"
        
        img = cv.imread(filename,0)
         
        h_cropped_low = int(img.shape[0] * 0.55)
        h_cropped_high = int(img.shape[0] * 0.9)   
            
        if img_number == 1:
            w_cropped_low = int(img.shape[1] * 0.05)
            w_cropped_high = int(img.shape[1] * 0.9)
        else:
            w_cropped_low = int(img.shape[1] * 0.1)
            w_cropped_high = int(img.shape[1] * 0.9)    
            
        img  = img[h_cropped_low:h_cropped_high, w_cropped_low:w_cropped_high]
        
        scale_percent = 40 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv.resize(img,dim, interpolation = cv.INTER_AREA)
        
        imgs.append(img)
        
        #THRESHOLD
        filter_size = int(152 * scale_percent /100)
        filter_size = filter_size+1 if filter_size %2 == 0 else filter_size
        
        th = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,filter_size,10)
        th = cv.GaussianBlur(th,(7,15),0)
        
        # Find Canny edges
        edged = cv.Canny(th, 50, 100,L2gradient = True)
        
        
        cnts,h = cv.findContours(edged, cv.RETR_EXTERNAL,
        	cv.CHAIN_APPROX_SIMPLE)
         
        #cnts = imutils.grab_contours(cnts)
        digitCnts = []
        
        POIs = {}
        
        rectangles = []
        # loop over the digit area candidates
        for c in cnts:
        	# compute the bounding box of the contour
            (x, y, w, h) = cv.boundingRect(c)
            
            rectangles.append(cv.boundingRect(c))
            #if w > 5 and h > 5:
                #rectangles.append(cv.boundingRect(c))   
                #pass

        """
        CODE TO DETECT THE OVERLAPPING RECTANGLE
        """
        V = len(rectangles)
        graph = Graph.Graph(V)
    

        test = {}
        for k,rect1 in enumerate(rectangles):
            test[k] = []
            for l,rect2 in enumerate(rectangles):
                if intersects(rect1,rect2):
                    graph.addEdge(k, l)
                    test[k].append(l)

        # For every ith element in the arr
        # find all reachable nodes from query[i]
        arr = [n for n in range(V)]
         
        reach_node = graph.findReachableNodes(arr, V)
        
        overlapping = {}
        already_visited=[]
        for key, value in reach_node.items():
            if key not in already_visited:
                overlapping[key] = value
                already_visited.append(key)
                for v in value:
                    already_visited.append(v)
            
            
        merged_rectangle = []
        for key, value in overlapping.items():
            if len(value)>1:
                x = []
                y = []
                x2 = []
                y2 = []
                for index in value:
                    rect = rectangles[index]
                    x.append(rect[0])
                    y.append(rect[1])
                    x2.append(rect[0]+rect[2])
                    y2.append(rect[1]+rect[3])
                merged_rectangle.append((min(x),min(y),max(x2)-min(x),max(y2)-min(y)))
            else:
                merged_rectangle.append(rectangles[value[0]])
        
        merged_rectangle = [*set(merged_rectangle)] 
        
        width_limit = int(75 * scale_percent/100)
        height_limit = int(150 * scale_percent/100)
        
        for x,y,w,h in merged_rectangle:
            # if the contour is sufficiently large, it must be a digit
            if w >= width_limit and h >= height_limit:
                
                POI = th[y:y+h, x:x+w]
            
                POIs[x] = POI
                merged_rectangles[i].append((x,y,w,h))
        
        POIs = collections.OrderedDict(sorted(POIs.items()))
        
        total_number = []
        
            

        imgs_th.append(th)
        
        total_numbers.append(total_number)
        POIs_total.append(POIs)
        
        #to remove
        test_rectangles.append(rectangles)
          
    return total_numbers, merged_rectangles, test_rectangles, imgs, imgs_th, POIs_total


  

    




            
            

