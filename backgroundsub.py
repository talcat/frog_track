#!/usr/bin/env python
#-*- encoding: utf-8 -*-

# modified from http://weirdinventionoftheday.blogspot.com/2012/10/opencv-background-subtraction-in-python.html

import cv2
import numpy as np
import sys
import os
from pylab import imread
import re

class FrogFrames:
    """An easy way to access all the frog images in a directory"""
    def __init__(self, path, loop=True, gray=True):
        self.path = path
        self.list = self.get_list()
        self.index = 0
        self.loop = loop
        self.gray= gray
        
    def get_list(self):
        """Gives a list of all the images in chronological order"""    
        #os.chdir(self.path)
        dirList = os.listdir(self.path)
        #print dirList
        dirList.sort()
        
        dirList = [ m.group(0) for m in (re.search(r"im-\d\d\d\d.png", img) for img in dirList) if m]   
        #print dirList
        return dirList
        
    def get_next_im(self):
        """Gives the next frame in the frog video frame directory.
        If not looping, will return None when the video is done"""
        #print self.list[self.index]
        if self.index >= len(self.list):
            #print 'Here we are'
            if self.loop: 
                self.index = np.mod(self.index, len(self.list))
            else:
                return None       
        
        im = imread('%s%s' % (self.path, self.list[self.index]))
        if self.gray:
            im = 255*cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            im = im.astype('uint8')
            
        self.index = self.index+1    
        
        return im

    def get_frame(self, num):
        """Returns frame number <num> mod number of frames
        NOTE: first frame is frame 0"""
        num = np.mod(num, len(self.list))

        im = imread('%s%s' % (self.path, self.list[num]))   
        
        if self.gray:
            im = 255*cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            im = im.astype('uint8')
            
        return im
        
    
if __name__ == "__main__":
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
