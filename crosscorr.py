# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 12:31:02 2013

@author: talcat

Cross Correlation Pattern Matching
"""

import cv2
from sel_prim_roi import * #contains select_prim_ROI function
from select_com import *
from FrogFrames import *   #contains FrogFrames class
from select_com import *   #contains select_COM function  
import numpy as np 
from pylab import imread, imshow, show, figure, subplot, title, close, connect, Rectangle, savefig, draw
from matplotlib import pyplot as plt
import time
from scipy import signal

RGB2GRAY = 7 #lets not import cv as well shall we
PATH = '/home/talcat/Desktop/Bio Interface/Frogs/frog_frame/Shot2/'
WIN = 64


def vid_pc(vid_fn):
    try:
        cap = cv2.VideoCapture(vid_fn)
    except:
        print "File not found"
        return
        cond = True

    prev_frame = []
    point_list = dict()
    ds_list = []
    idx = 0
    st_frame = 0
    while True:
        ret, frame = cap.read()
        idx += 1
        if ret is False:
            break        
               
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          
        #Get the ROI
        if idx == 1:   #If first frame, initialize it
            cond = True
            #Go to first relevant frame:
            while cond:
                cv2.imshow('frame', frame)
                k = cv2.waitKey(-1)
                if k == ord('a'):
                    st_frame = idx
                    break
                else:
                    ret, frame = cap.read()
                    idx +=1
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


            roi = select_prim_ROI(frame)
            (minr, maxr), (minc, maxc) = roi
            point_to_track, roi = select_COM(frame, (minr, maxr), (minc, maxc))
            #point_list.append(point_to_track)
            point_list[idx] = point_to_track
            prev_frame = frame
            
        else: #we are on frame 2 and beyond
            ptr, ptc = point_list[idx-1]
            old_roi = np.array([[ptr - WIN/2, ptr+WIN/2],[ptc - WIN/2,ptc + WIN/2]])
            col, row, colr, rowr = test_phasecorrelate(old_roi, prev_frame, frame)
            
            if abs(col + colr) > 1 or abs(row + rowr) > 1:
                print "cross correlation broke"
                break
            else:
                row = (row - rowr)/2
                col = (col - colr)/2
                ds_list.append([row, col])
                #point_list.append([ptr + row, ptc + col])
                point_list[idx] = [ptr + row, ptc + col]
                prev_frame = frame
        
    # Now lets draw the thing
    print "Done with phase correlation"
    cap.release()
    cap = cv2.VideoCapture(vid_fn)
    loop = 1
    while True:
        ret, frame = cap.read()
        #pts when at st_frame
        if ret is False:
            break
        if loop in point_list.keys():
            (y, x) = point_list[loop]
            cv2.circle(frame, (int(x), int(y)), 8, (0, 255, 0), 2)
        cv2.imshow('ack', frame)
        k = cv2.waitKey(200)
        if k == 27:
            break    
        loop += 1    




def crop(roi, image):
    (minr, maxr), (minc, maxc) = roi
    return image[minr:maxr, minc:maxc]


def test_phasecorrelate(roi, frame1, frame2):
    (minr, maxr), (minc, maxc) = roi
    subim1 = crop(roi, frame1)
    subim2 = crop(roi, frame2)
    
    #need to convert to gray and float32 or float64
    #subim1 = np.float32(cv2.cvtColor(subim1, RGB2GRAY))
    #subim2 = np.float32(cv2.cvtColor(subim2, RGB2GRAY))
    
    #both = np.hstack([subim1, subim2])
    #cv2.imshow('ack', both)
    #k = cv2.waitKey(100)
    
    subim1 = np.float32(subim1)
    subim2 = np.float32(subim2)
    
    col, row = cv2.phaseCorrelate(subim1, subim2)
    #print (col, row)
    colr, rowr = cv2.phaseCorrelate(subim2, subim1)
    #print (colr, rowr)
    return col, row, colr, rowr


def findwidth(percent, win_size):
    """Finds the width of a gaussian such that the edges are percent of the max
    = 1 """
    p_sq = -8*np.log(percent)/win_size**2
    return np.sqrt(p_sq)

def guass_win(percent, subim):
    winsize = subim.shape[0]
    per = np.sqrt(percent)
    sig = 1/findwidth(per, win_size);
    g1d_g = signal.get_window( ('gaussian', sig), win_size)
    g2d_g = np.outer(g1d_g, g1d_g)
    
    return g2d_g * subim
   

def custom_rpc(roi, frame1, frame2):
    """still fixing this"""
    (minr, maxr), (minc, maxc) = roi
    subim1 = crop(roi, frame1)
    subim2 = crop(roi, frame2)
    
    if subim1.shape[2] == 3: #RGB
        for k in range(3):
            im1 = subim1[:,:,k]
            im2 = subim2[:,:,k]
            
            #gauss window
            im1 = gauss_win(.1, im)
            im2 = gauss_win(.1, im)
            
            f_im1 = cv2.dft(im1, flags = cv2.DFT_COMPLEX_OUTPUT)
            
            
def template_match(roi, frame1, frame2):
    (minr, maxr), (minc, maxc) = roi
    subim1 = crop(roi, frame1)
    img = frame2.copy()
    res = cv2.matchTemplate(img, subim1,  cv2.TM_SQDIFF_NORMED)
    
    minv, maxc, minl, maxl = cv2.minMaxLoc(res)
    

    
    w, h = subim1.shape[::-1]
    
    top_left = (minl[0]), (minl[1])
    
    bottom_right = (minl[0] + w), (minl[1] + h)            
    
    print top_left
    print bottom_right
    
    cv2.rectangle(img, (int(top_left[0]), int(top_left[1])), (int(bottom_right[0]), int(bottom_right[1])), 255, 2)

    cv2.imshow('Tracked', img)
    
    k = cv2.waitKey(-1)
    
    if k==32:
        return np.array([ [top_left[1], top_left[1] + h], [top_left[0], top_left[0] + w]])

    #plt.subplot(121),plt.imshow(res,cmap = 'gray')
    #plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    #plt.subplot(122),plt.imshow(img,cmap = 'gray')
    #plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    #plt.suptitle("RAWWWWWWWWWWR")

    #plt.show()
    
    
def template_vid_run():
    video = cv2.VideoCapture("/home/talcat/Google Drive/Lab Stuff/AEThER & Socha/Frogs/Indian Skitter Frog/Movies_pictures/Gulo Film Clips/test.mp4")
    
    roi_list = []
    
    cond=True
    idx = 0
    while cond:
        ret, frame = video.read()        
        idx += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)       
        #If frame == None, no more files:
        if ret == False:
            cond = False
            break
        
        #Get the ROI
        if idx == 1:   #If first frame, initialize it
            roi = select_prim_ROI(frame)
            (minr, maxr), (minc, maxc) = roi
            roi_list.append(roi)
            
            prev_frame = frame
            
        else: #we are on frame 2 and beyond
            #get prev roi to template match
            prev_roi = roi_list[idx-2]
            
            #template match
            new_roi = template_match(prev_roi, prev_frame, frame)
            prev_frame = frame
            roi_list.append(new_roi)    

def template_run():
    video = FrogFrames(PATH, loop=False, gray=True, eq=False  )
    
    roi_list = []
    
    cond=True
    
    while cond:
        frame, idx = video.get_next_im()        
               
        #If frame == None, no more files:
        if frame == None:
            cond = False
            break
        
        #Get the ROI
        if idx == 0:   #If first frame, initialize it
            roi = select_prim_ROI(frame)
            (minr, maxr), (minc, maxc) = roi
            roi_list.append(roi)
            
            prev_frame = frame
            
        else: #we are on frame 2 and beyond
            #get prev roi to template match
            prev_roi = roi_list[idx-1]
            
            #template match
            new_roi = template_match(prev_roi, prev_frame, frame)
            prev_frame = frame
            roi_list.append(new_roi)
            
    
if __name__ == "__k__":
    video = FrogFrames(PATH, loop=False, gray=True, eq=False  )
    
    
    cond = True
    prev_frame = []
    point_list = []
    ds_list = []
    while cond:
        frame, idx = video.get_next_im()        
               
        #If frame == None, no more files:
        if frame == None:
            cond = False
            break
        
        #Get the ROI
        if idx == 0:   #If first frame, initialize it
            roi = select_prim_ROI(frame)
            (minr, maxr), (minc, maxc) = roi
            point_to_track, roi = select_COM(frame, (minr, maxr), (minc, maxc))
            point_list.append(point_to_track)
            
            prev_frame = frame
            
        else: #we are on frame 2 and beyond
            ptr, ptc = point_list[idx-1]
            old_roi = np.array([[ptr - WIN/2, ptr+WIN/2],[ptc - WIN/2,ptc + WIN/2]])
            col, row, colr, rowr = test_phasecorrelate(old_roi, prev_frame, frame)
            
            if abs(col + colr) > 1 or abs(row + rowr) > 1:
                print "cross correlation broke"
                break
            else:
                row = (row - rowr)/2
                col = (col - colr)/2
                ds_list.append([row, col])
                point_list.append([ptr + row, ptc + col])
                prev_frame = frame
    # Now lets draw the thing
    print "Done with phase correlation"
      

    
        

        
            
            
 
            
