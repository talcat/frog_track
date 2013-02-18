# FrogFrames.py

# Contains the FrogFrames Class

import numpy as np
import sys
import os
from pylab import imread
import re
import cv2

class FrogFrames:
    """An easy way to access all the frog images in a directory"""
    def __init__(self, path, loop=True, gray=True, eq=False):
        self.path = path
        self.list = self.get_list()
        self.index = 0
        self.loop = loop
        self.gray= gray
        self.eq = eq
        
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
                return None, np.NaN       
        
        im = imread('%s%s' % (self.path, self.list[self.index]))
        if self.gray or self.eq:
            im = 255*cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            im = im.astype('uint8')
        if self.eq:
            im = cv2.equalizeHist(im.astype('uint8'))    
        self.index = self.index+1    
        
        return im, self.index-1

    def get_frame(self, num):
        """Returns frame number <num> mod number of frames
        NOTE: first frame is frame 0"""
        num = np.mod(num, len(self.list))

        im = imread('%s%s' % (self.path, self.list[num]))   
        
        if self.gray or self.eq:
            im = 255*cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            im = im.astype('uint8')
        if self.eq:
            im = cv2.equalizeHist(im.astype('uint8'))    
        return im
