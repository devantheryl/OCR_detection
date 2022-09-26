# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 13:57:23 2022

@author: LDE
"""


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


def cropp_resize(img):
    h_cropped_low = int(img.shape[0] * 0.45)
    h_cropped_high = int(img.shape[0] * 0.85)   
    w_cropped_low = int(img.shape[1] * 0.2)
    w_cropped_high = int(img.shape[1] * 0.9)
    
    img_cropped = img[h_cropped_low:h_cropped_high, w_cropped_low:w_cropped_high]
    
    
    
    scale_percent = 40 # percent of original size
    width = int(img_cropped.shape[1] * scale_percent / 100)
    height = int(img_cropped.shape[0] * scale_percent / 100)
    dim = (width, height)
    img_resized = cv.resize(img_cropped,dim, interpolation = cv.INTER_AREA)
    
    return img_cropped, img_resized

def find_numbers_positions(img):
    
    
    
    
    
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_cropped,img_resized = cropp_resize(img)
    
    
    #THRESHOLD
    th = cv.adaptiveThreshold(img_resized,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,61,18)
    th_blurred = cv.GaussianBlur(th,(3,7),0)
    _,th_blurred_th = cv.threshold(th_blurred,240,255,cv.THRESH_BINARY)
    edged = cv.Canny(th_blurred_th, 50, 100,L2gradient = True, apertureSize = 3)
    
    
    
    cnts,h = cv.findContours(edged, cv.RETR_EXTERNAL,
    	cv.CHAIN_APPROX_SIMPLE)
     
    #cnts = imutils.grab_contours(cnts)
    
    POIs_th = {}
    POIs_img_resized = {}
    POIs_img = {}
    POI_blurred_th = {}
    merged_rectangles = {}
    
    rectangles = []
    th_plot = th.copy()
    # loop over the digit area candidates
    for c in cnts:
    	# compute the bounding box of the contour
        (x, y, w, h) = cv.boundingRect(c)
        if w < 60:
            if y <25 and h <50:
                rectangles.append((x, y, w, h+5))
                th_plot = cv.rectangle(th_plot, (x,y), (x+w, y+h+5),(0,0,0),2)
            elif y >50 and y <80 and h <50:
                rectangles.append((x, y-5, w, h+5))
                th_plot = cv.rectangle(th_plot, (x,y-5), (x+w, y+h),(0,0,0),2)
            else:
                    
                rectangles.append(cv.boundingRect(c))
                th_plot = cv.rectangle(th_plot, (x,y), (x+w, y+h),(0,0,0),2)

    """
    plt.imshow(th,'gray')
    plt.show() 
    
    
    plt.imshow(th_blurred,'gray')
    plt.show() 
    
    
    plt.imshow(th_blurred_th,'gray')
    plt.show() 
    
    # Find Canny edges
   
    plt.imshow(edged,'gray')
    plt.show() 
    
    plt.imshow(th_plot,'gray')
    plt.show()  
    """
    
    """
    CODE TO DETECT THE OVERLAPPING RECTANGLE
    """
    V = len(rectangles)
    graph = Graph.Graph(V)



    for k,rect1 in enumerate(rectangles):

        for l,rect2 in enumerate(rectangles):
            if intersects(rect1,rect2):
                graph.addEdge(k, l)
                

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
    
    width_limit_low = int(60 * 40/100)
    height_limit_low = int(140 * 40/100)
    
    for x,y,w,h in merged_rectangle:
        # if the contour is sufficiently large, it must be a digit
        if w >= width_limit_low and h >= height_limit_low:
            
            POI_th = th_blurred_th[y:y+h, x:x+w]
            POI_img_resized = img_resized[y:y+h, x:x+w]
            POI_img = img_cropped[int((100/40 )* y):int((100/40 )* (y+h)), int((100/40 )* x):int((100/40 )* (x+w))]
        
            POIs_th[x] = POI_th
            POIs_img_resized[x] = POI_img_resized
            POIs_img[x] = POI_img
            
            
            merged_rectangles[x] = (x,y,w,h)
    
    POIs_th = collections.OrderedDict(sorted(POIs_th.items()))
    POIs_img_resized = collections.OrderedDict(sorted(POIs_img_resized.items()))
    POIs_img = collections.OrderedDict(sorted(POIs_img.items()))
    merged_rectangles = collections.OrderedDict(sorted(merged_rectangles.items()))
    

          
    return merged_rectangles,img_cropped, img_resized, th, POIs_th, POIs_img_resized, POIs_img


  


    




            
            

