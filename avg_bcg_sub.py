import cv2
from sel_prim_roi import * #contains select_prim_ROI function
from FrogFrames import *   #contains FrogFrames class
from select_com import *   #contains select_COM function  
import numpy as np 
from pylab import imread, imshow, show, figure, subplot, title, close, connect, Rectangle, savefig, draw
import time
from grabcut import get_grabcut_seg

RGB2GRAY = 7
BGR2GRAY = 6
BGR2Lab = 44
RGB2Lab = 45
PATH = '/home/talcat/Desktop/Bio Interface/Frogs/frog_frame/Shot2/'

lk_params = dict( winSize  = (10, 10),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def create_mask(mean_bgnd, image):
    """Given the mean background and an image, will return a foreground mask"""
    
    #Assume color at the moment
    res = abs(mean_bgnd - image)
    
    return res


def calculate_mean(video):
    
    for k in range(video.num_frames):
        frame = video.get_frame(k)
        if k == 0:
            mean = frame
        else:
            mean += frame
    return mean/video.num_frames
    
def lab_distance(video):
    mean = calculate_mean(video)
    mean = (255*mean).astype('uint8')
    mean_lab = cv2.cvtColor(mean, BGR2Lab)    
    cv2.imshow('mean', mean_lab)
    for k in range(video.num_frames):
        frame = video.get_frame(k)
        frame = (255*frame).astype('uint8')
        frame = cv2.cvtColor(frame, BGR2Lab)
        cv2.imshow('frame', frame)
        sq_dif = (mean - frame)**2
        #print sq_dif.shape
        #dis0 = np.sum(sq_dif, axis=0)
        #dis1 = np.sum(sq_dif, axis=1)
        dis2 = np.sum(sq_dif, axis=2)
        #print (dis0.shape, dis1.shape, dis2.shape)
        dis2 = (255*(dis2 - dis2.min())/dis2.max()).astype('uint8')
        imshow(dis2)
        cv2.imshow('dif', dis2)
        k = cv2.waitKey(-1)
        if k == 65363: #->
            continue
        if k == 27: #escape
            cv2.destroyAllWindows()
            return dis2
            break
            
def rgb_distance(video):
    mean = calculate_mean(video)
    mean = (255*mean).astype('uint8')
        
    cv2.imshow('mean', mean)
    for k in range(video.num_frames):
        frame = video.get_frame(k)
        frame = (255*frame).astype('uint8')
        
        cv2.imshow('frame', frame)
        sq_dif = (mean - frame)**2
        #print sq_dif.shape
        #dis0 = np.sum(sq_dif, axis=0)
        #dis1 = np.sum(sq_dif, axis=1)
        dis2 = np.sum(sq_dif, axis=2)
        #print (dis0.shape, dis1.shape, dis2.shape)
        imshow(dis2, cmap='gray')
        dis2 = (255*(dis2 - dis2.min())/dis2.max()).astype('uint8')
        cv2.imshow('dif', dis2)
        k = cv2.waitKey(-1)
        if k == 65363: #->
            continue
        if k == 27: #escape
            cv2.destroyAllWindows()
            break
            
              

def test():
    while False:
        for k in range(2):
            left = create_mask(mean, video.get_frame(k))
            if k == 0:
               roi = select_prim_ROI(left)
               (minr, maxr), (minc, maxc) = roi.astype('int')
               prev_pts = [(x, y) for x in range(minc, maxc, 5) for y in range(minr, maxr, 5)]
            else:
                p_im = create_mask(mean,video.get_frame(k-1))
                c_im = create_mask(mean,video.get_frame(k))
                p_im = cv2.cvtColor((255*p_im).astype('uint8'), RGB2GRAY)
                c_im = cv2.cvtColor((255*c_im).astype('uint8'), RGB2GRAY)
                p0 = np.float32([tr[-1] for tr in prev_pts]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(p_im, c_im, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(c_im, p_im, p1, None, **lk_params)
                d = abs(prev_pts-p0r).reshape(-1, 2).max(-1)
                good = d < 1
            
        
    show()
    
    return

def MOG_bck(vid_file):

    cap = cv2.VideoCapture(vid_file)

    fgbg = cv2.BackgroundSubtractorMOG()

    while(1):
        ret, frame = cap.read()
        
        if ret == False:
            break
        
        fgmask = fgbg.apply(frame)

        cv2.imshow('frame',fgmask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
def histogram(video):    
    roi = video.get_frame(0)
    hsv = cv2.cvtColor(roi,cv2.COLOR_RGB2HSV)
    roi = (255 * roi).astype('uint8')
    target = roi
    hsvt = cv2.cvtColor(target,cv2.COLOR_RGB2HSV)

    # calculating object histogram
    mask = get_grabcut_seg(roi)
    roihist = cv2.calcHist([hsv],[0, 1], mask, [180, 256], [0, 180, 0, 256] )

    # normalize histogram and apply backprojection
    cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
    
    # Now convolute with circular disc
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    cv2.filter2D(dst,-1,disc,dst)
    cv2.imshow('a', dst)
    cv2.waitKey(-1)
    # threshold and binary AND
    ret,thresh = cv2.threshold(dst,50,255,0)
    thresh = cv2.merge((thresh,thresh,thresh))
    res = cv2.bitwise_and(target,thresh)

    res = np.vstack((target,thresh,res))
    #cv2.imwrite('res.jpg',res)    
    imshow(res)

if __name__ == "__main__":
    video = FrogFrames(PATH, loop=False, gray=False, eq=False  )
    
    

        

    
    
