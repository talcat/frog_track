# updateroi.py

from matplotlib import use
use('Qt4Agg')
from matplotlib.widgets import  RectangleSelector
from pylab import *
import cv2

def updateroi(mask0, im0, im1, pyr_scale=0.5, levels=1, winsize=1000, iterations=4,
                poly_n =7, poly_sigma=1.5, flags=cv2.OPTFLOW_USE_INITIAL_FLOW):
    """Given two sequential images, and the roi mask for the first, will return
    a roi mask for the second image by using OpenCV's OpticalFlowFarneback 
    algorithm to estimate the amount the mask should be moved"""
    
    im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2GRAY)
    im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    mask0 = cv2.cvtColor(mask0, cv2.COLOR_RGB2GRAY)
    
    #make them 
    
    winsize=(20, 20)
    maxlevel=3
    
    r0, r1, c0, c1 = returncorners(mask0)
    
    XX, YY = meshgrid(range(r0, r1), range(c0, c1))
    oldpts = np.vstack([XX.ravel(), YY.ravel()]).transpose()
    
    ret, pyr = cv2.buildOpticalFlowPyramid(im0.view('uint8'), winsize, maxlevel)
    next, stat, err = cv2.calcOpticalFlowPyrLK(im0.view('uint8'), im1.view('uint8'), oldpts ) 
    
    
    #flow = cv2.calcOpticalFlowFarneback(im0, im1, pyr_scale, levels, winsize, 
    #        iterations, poly_n, poly_sigma, flags)
            
    return next, stat, err
    

def returncorners(mask):
    """Given a mask, finds the and returns the four corners"""
    if len(mask.shape) == 3:
        dim = mask.shape[-1]
        if dim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    
    row, col = nonzero(mask==1)
    
    return [min(row), max(row), min(col), max(col)]


    
if __name__ == "__main__":
    mask0 = imread('mask-0.png')
    im0 = imread('im-0.png')
    im1 = imread('im-1.png')
    
    
    next, stat, err = updateroi(mask0, im0, im1)
    
    
    
    #flow = updateroi(mask0, im0, im1)
    
    #r0, r1, c0, c1 = returncorners(mask0)
    #XX, YY = np.mgrid[r0:r1, c0:c1]
    #flowr = flow[r0:r1, c0:c1, 0]
    #flowc = flow[r0:r1,c0:c1, 1]
    #flowr = flow[:, :, 0]
    #flowc = flow[:, :, 1]
    #quiver(XX, YY, flowc, flowr, units='xy',scale=1.0,   scale_units='dots')
    #show()
    
    #figure()
    #quiver(flowc, flowr, units='xy',scale=1.0,   scale_units='dots')
    #Averging check
    #avflowr = sum(flowr.ravel())/nonzero(flowr.ravel() > 1).size
    #avflowc = sum(flowc.ravel())/nonzero(flowc.ravel() > 1).size
    
    #tofill = ones(size(flowr))
    
    #avflowr_t = avflowr*tofill
    #avflowc_t = avflowc*tofill
    
    #quiver(XX, YY, avflowc_t, avflowr_t)
   # show()
