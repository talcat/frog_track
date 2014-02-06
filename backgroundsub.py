#!/usr/bin/env python
#-*- encoding: utf-8 -*-

# modified from http://weirdinventionoftheday.blogspot.com/2012/10/opencv-background-subtraction-in-python.html

import cv2
import numpy as np
import sys
import os
from pylab import imread
import re
from FrogFrames import *
        
def back_sub(video_fn):
    try:
        cap = cv2.VideoCapture(video_fn)
    except:
        print "Filename does not exist - Try again?"
        return

    bg = cv2.BackgroundSubtractorMOG(50, 5, .9, 0.01)

    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        fgmask = bg.apply(frame)

        cv2.imshow('frame', fgmask)
        k = cv2.waitKey(30)
        if k == 27: #esc
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__jhg__":
    bgs = cv2.BackgroundSubtractorMOG(50, 5, .9, 0.01)

    video = FrogFrames('/home/talcat/Desktop/Bio Interface/Frogs/frog_frame/Shot2/', False)
    cv2.namedWindow("input")

    cond = True
    while(cond):
    
        img = video.get_next_im()
        if img == None:
            cond=False
            break
            
        img = cv2.equalizeHist(img.astype('uint8'))
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 15)
        #fgmask = bgs.apply(img)
        
        cv2.imshow("input", img)
        key = cv2.waitKey(3)
        #cv2.imwrite("./pngs/image-"+str(a).zfill(5)+".png", fgmask)
        
        if key ==32:
            cond=False
        
        #print(video.index)
    cv2.destroyAllWindows() 
