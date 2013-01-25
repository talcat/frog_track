#matchpts.py  

from matplotlib import use
use('Qt4Agg')
from matplotlib.widgets import  RectangleSelector
from pylab import *
import cv2

def matchpts(mask0, im0, im1):
    """ASSUMING SMALL MOVEMENTS, the roi between 2 adjacent frames should be the
    same.  Thus, between two pairs of images, the same mask is used.
    This calculates SURF points for two sequential images, and matches them.  
    An average velocity is then calculated for the movement in the ROI.
    The velocity is returned so the ROI can be updated"""
    
    im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2GRAY)
    im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    mask0 = cv2.cvtColor(mask0, cv2.COLOR_RGB2GRAY)
    
    thesurf = cv2.SURF()
    
    keypts0 = thesurf.detect(im0.view('uint8'), mask0.view('uint8'))
    
    return keypts0
    
    
    
if __name__ == "__main__":
    mask0 = imread('mask-0.png')
    im0 = imread('im-0.png')
    im1 = imread('im-1.png')
    
    keypts0 = matchpts(mask0, im0, im1)
