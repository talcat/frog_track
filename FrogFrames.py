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
    def __init__(self, path, loop=False, wantgray=False, eq=False):
        self.path = path
        if self.path[-1] is not os.sep:
            self.path += os.sep
        self.list = self.get_list()
        self.index = 0
        self.loop = loop
        self.wantgray= wantgray 
        #self.eq = eq
        self.num_frames = len(self.list)
        self.isgray = self._check_gray() #are the images we are loading inherantly grayscale?
        self.wantgray = self.wantgray or self.isgray #if it is gray, we have no choice
        self.avg = None
        self._max = 0
        self._min = 0
        self._get_stats()

    def __repr__(self):
        return "FrogFrames Object from %s" %(self.path)
    def __str__(self):
        return self.__repr__()   
    
    def _get_stats(self):
        """want to calc min and max values of all frames so we can convert them to
        arbitrary bit depths/formats/etc"""
        max = 0
        min = 0
        for k in range(self.num_frames):
            im = self.get_frame(k, wantgray=True)
            if im.max() > max:
                max = im.max()
            if im.min() < min:
                min = im.min()
        self._min = min
        self._max = max
    
    def _check_gray(self):
        try:
            im = imread('%s%s' % (self.path, self.list[0]))
        except:
            im = cv2.imread('%s%s' % (self.path, self.list[0]))
        if len(im.shape) == 2:
            return True
        elif im.shape[2] == 1:
            return True
        else:
            return False


    def get_list(self):
        """Gives a list of all the images in chronological order"""    
        #os.chdir(self.path)
        dirList = os.listdir(self.path)
        #print dirList
        dirList.sort()
        
        dirList = [ m.group(0) for m in (re.search(r"(im-\d{4}|img_\d{4}|\d{7}|im-\d{5}).(png|tif)", img) for img in dirList) if m]   
        #print dirList
        return dirList
        
    def get_next_im(self, wantgray=None):
        """Gives the next frame in the frog video frame directory.
        If not looping, will return None when the video is done"""
        #print self.list[self.index]
        if self.index >= len(self.list):
            #print 'Here we are'
            if self.loop: 
                self.index = np.mod(self.index, len(self.list))
            else:
                return None, np.NaN    
        im = self.get_frame(self.index, wantgray)           
        #try:
        #    im = imread('%s%s' % (self.path, self.list[self.index]))
        #except: #imread does not like these tifs?
        #    tmpim = cv2.imread('%s%s' % (self.path, self.list[self.index]))
        #    im = tmpim.copy() #CV2 reads in BGR - its RGB now
        #    im[:,:,0] =tmpim[:,:,2]
        #    im[:,:,2] = tmpim[:,:,0]

        #if self.gray or self.eq or wantgray:
        #    im = 255*cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        #    im = im.astype('uint8')
        #if self.eq:
        #    im = cv2.equalizeHist(im.astype('uint8'))    
        self.index = self.index+1    
        
        return im, self.index-1

    def get_frame(self, num, wantgray=None):
        """Returns frame number <num> mod number of frames
        NOTE: first frame is frame 0"""
        num = np.mod(num, len(self.list))
        if wantgray is None:
            wantgray = self.wantgray

        GRAY = False

        if not self.isgray: # if its alreasy gray, we dont need to do anything
            if wantgray:
                GRAY = True
        #print GRAY
        
        try:
            im = imread('%s%s' % (self.path, self.list[num]), flatten = GRAY)
            im = im.astype('uint16')
        except:
            tmpim = cv2.imread('%s%s' % (self.path, self.list[num]), not GRAY )
            if not self.isgray and not wantgray:
                im = tmpim.copy() #CV2 reads in BGR - its RGB now
                im[:,:,0] =tmpim[:,:,2]
                im[:,:,2] = tmpim[:,:,0]
            else: im = tmpim
        #if not self.isgray: # if its alreasy gray, we dont need to do anything
        #    if (self.wantgray or self.eq) and wantgray:
        #        im = 255*cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        #        im = im.astype('uint16')
        #if self.eq:
        #    im = cv2.equalizeHist(im.astype('uint8'))    
        #print im.dtype
        return im

    def calc_avg(self):
        sum_f = np.zeros(self.get_frame(0).shape)
        for k in range(self.num_frames):
            try:
                sum_f += self.get_frame(k)
            except:
                print "something went wrong on frame %d" %k
        sum_f /= self.num_frames

        self.avg = sum_f

    def get_mean_sub_frame(self, num):
        if self.avg is None:
            print "Need to calculate average first"
            self.calc_avg()
        im = self.get_frame(num)
        #cast to uint8 because fuck it

        im = abs(im.astype('float') - self.avg)
        #im = cv2.inRange(im, avg - 5, avg + 5)
        return im.astype('uint16')\

    #def get_median_im(self):
    #    #get the median image of the frog video
    #    winsize = 



def cvt16to8(im, min, max):
    mx16 = 65535
    mx8 = 255
    im = im.astype('float')
    im = 255*(im - min)/max
    return im.astype('uint8')