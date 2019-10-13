#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 07:44:39 2019

@author: chaitralikshirsagar
"""

import cv2
import xml.etree.ElementTree as ET
import numpy as np
 
if __name__ == '__main__':

    #read input image
    ip_img='/Users/chaitralikshirsagar/Downloads/HW2_Data/JPEGImages/000480.jpg'
    im=cv2.imread(ip_img)
    
    #Convert Input image to RGB before running Selective Search
    im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

    # create Selective Search and Graph Segmentation Object using default parameters
    ss=cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    sg=cv2.ximgproc.segmentation.createGraphSegmentation()
 
    # add input image on which we will run selective search algorithm
    ss.addImage(im)
    
    #Add Graph segmentation object to Selective Search object
    ss.addGraphSegmentation(sg)
 
    #Create an object to carry Selective Search algoritm using color strategy
    sc = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()
    
    #Objects created to carry Selective Search using different strategies like Texture,Size,Fill
    st = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyTexture()
    sz = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategySize()
    sf = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyFill()
    
    #Create an object to carry Selective Search algoritm using all strategies 
    sm = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyMultiple(sc,st,sz,sf)
    
    #Uncomment/Comment when only color strategy required
    ss.addStrategy(sc)
    
    #Uncomment/Comment when all strategies are to be used
    #ss.addStrategy(sm)

    # run selective search segmentation on input image
    rects = ss.process()
    
    #Prints total region proposals the algorithm gives.
    print('Total Number of Region Proposals: {}'.format(len(rects)))
     
    # number of region proposals to show
    numShowRects = 100
    
    tp=0
    
    #Parse the ground truth xml files to get ground trut bouding box locations
    dimensions=[]
    root=ET.parse('/Users/chaitralikshirsagar/Downloads/HW2_Data/Annotations/000480.xml').getroot()
    for dims in root.findall('object/bndbox'):
        xmin=int(dims.find('xmin').text)
        xmax=int(dims.find('xmax').text)
        ymin=int(dims.find('ymin').text)
        ymax=int(dims.find('ymax').text)
        dimensions.append([xmin,ymin,xmax,ymax])

    #defines total ground truth boxes present in the input image        
    num_total=len(dimensions)
    
    #a list that will contain the predicted boxes that have a iou>0.5 with GT box
    count=np.zeros((num_total,1))
    
    #main loop that iterates for every predicted box, and for every ground truth box and calculates union area, interesection area, iou 
    while(True):
        imOut = im.copy()
        im_algo=im.copy()
        im_gt=im.copy()
        
        for i, rect in enumerate(rects):
            if (i < numShowRects):
                #returns coordinates of predicted box
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
                    cv2.rectangle(im_gt, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1, cv2.LINE_AA)
 
                    #Check if IoU is greater than 0.5
                    if(iou>0.5):
                        print("Box num",j)
                        count[j]+=1
                        cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
                        cv2.rectangle(imOut, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1, cv2.LINE_AA)

            else:
                break
        
        #calculate how many predicted boxes have iou>0.5 for each ground truth box
        for i in range(num_total):
            if(count[i]>=1):
                tp+=1
        
        #calculate recall as ratio of number of predicted boxes with iou>0.5 and total number of ground truth boxes
        recall=tp/num_total
        print("Recall value: ", recall)
        
        #Convert image back to BGR for display
        imOut=cv2.cvtColor(imOut,cv2.COLOR_RGB2BGR)
        im_algo=cv2.cvtColor(im_algo, cv2.COLOR_RGB2BGR)
        im_gt=cv2.cvtColor(im_gt,cv2.COLOR_RGB2BGR)

        #Display the overlapped(imOut) / ground -truth(im_gt)/ algorithm result(im_algo)
        cv2.imshow("Output", imOut)
        cv2.imshow("Algorithm Output", im_algo)
        cv2.imshow("Ground truth boxes", im_gt)
        
        #quit program by pressing 'q'
        k = cv2.waitKey(0) & 0xFF
        if k==113:
            break

    cv2.destroyAllWindows()
	
