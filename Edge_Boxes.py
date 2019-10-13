#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 09:23:18 2019

@author: chaitralikshirsagar
"""

import cv2
import numpy as np
import xml.etree.ElementTree as ET

#read input image
ip_img = '/Users/chaitralikshirsagar/Downloads/HW2_Data/JPEGImages/000480.jpg'
ip_img=cv2.imread(ip_img)

#Convert Input image to RGB before running Structured Edge Detection
rgb_img=cv2.cvtColor(ip_img,cv2.COLOR_BGR2RGB)

#convert uint8 image to float as Structured Edge Detection needs float values
rgb_img =np.float32(rgb_img)/255.0

#Path to Structured edge detection model
model='/Users/chaitralikshirsagar/Downloads/model.yml.gz'

#create an object for Structured edge detection
edge_detection=cv2.ximgproc.createStructuredEdgeDetection(model)

#Detect edges in input image
edges=edge_detection.detectEdges(rgb_img)

#Create a edge orientation map from detected edges
orimap=edge_detection.computeOrientation(edges)

#Suppress certain edges according to threshold values
edges_nms=edge_detection.edgesNms(edges,orimap)

#Create object for EdgeBoxes algorithm
edge_boxes=cv2.ximgproc.createEdgeBoxes()

#Setting parameters alpha and beta for step size in window search and nms threshold
edge_boxes.setAlpha(0.5)
edge_boxes.setBeta(0.75)

#Generate bounding boxes from nms output and edge orientation map using EdgeBoxes algorithm
boxes=edge_boxes.setMaxBoxes(100)
boxes=edge_boxes.getBoundingBoxes(edges_nms,orimap)

#Parse the ground truth xml files to get ground truth bouding box locations
dimensions=[]
root=ET.parse('/Users/chaitralikshirsagar/Downloads/HW2_Data/Annotations/000480.xml').getroot()
for dims in root.findall('object/bndbox'):
    xmin=int(dims.find('xmin').text)
    xmax=int(dims.find('xmax').text)
    ymin=int(dims.find('ymin').text)
    ymax=int(dims.find('ymax').text)
    dimensions.append([xmin,ymin,xmax,ymax])
    
num_total=len(dimensions)

#a list that will contain the predicted boxes that have a iou>0.5 with GT box
count=np.zeros((num_total,1))
tp=0

#main loop that iterates for every predicted box, and for every ground truth box and calculates union area, interesection area, iou 
while(True):
    imOut = ip_img.copy()
    im_algo = ip_img.copy()
    im_gt = ip_img.copy()
     
    for rect in boxes:
        #returns coordinates of ground truth box
        x, y, w, h = rect
        for j,box in enumerate(dimensions):
            #returns coordinates of ground truth box
            xmin,ymin,xmax,ymax=box
            
            #calculate minimum and maximum values of (x,y) for proposed and GT boxes
            xa=max(x,xmin)
            ya=max(y,ymin)
            xb=min(x+w,xmax)
            yb=min(y+h,ymax)
                    
            #calculate intersection area
            intersection_area=max(0,xb-xa)*max(0,yb-ya)
                    
            #calculate ground truth bbox area
            gt_area=((xmax-xmin))*((ymax-ymin))
                    
            #calculate ground truth image area
            img_area=((x+w)-x)*((y+h)-y)
                    
            #calculate IoU
            iou=intersection_area/float(gt_area+img_area-intersection_area)
                    
            #defines a rectangle around the corresponding coordinates
            cv2.rectangle(im_algo, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(im_gt, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1, cv2.LINE_AA)
                    
            #Check if IoU is greater than 0.5
            if(iou>0.5):
                count[j]+=1
                cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(imOut, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1, cv2.LINE_AA)
            else:
                continue


    #calculate how many predicted boxes have iou>0.5 for each ground truth box
    for i in range(num_total):
        if(count[i]>=1):
            tp+=1
    
    #calculate recall as ratio of number of predicted boxes with iou>0.5 and total number of ground truth boxes
    recall=tp/num_total
    print("Recall value: ", recall)

    #Display the overlapped(imOut) / ground -truth(im_gt)/ algorithm result(im_algo)
    cv2.imshow("Output", imOut)
    cv2.imshow("Algorithm Output", im_algo)
    cv2.imshow("Ground truth boxes", im_gt)
    
    #quit program by pressing 'q'
    k = cv2.waitKey(0) & 0xFF
    if k==113:
        break

cv2.destroyAllWindows()
	

	
