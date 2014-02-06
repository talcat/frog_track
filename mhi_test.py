#mhi_test.py
# seeing if OpenCV's motion history does anything useful....

from matplotlib import use
use('Qt4Agg')
from matplotlib.widgets import  RectangleSelector
from pylab import *
import cv2

def init_motionhis(im0):
    """initializes a motion history"""
    h, w = frame.shape[:2]
    motion_hist = np.zeros((h, w), np.float32)
    return motion_hist
    
def update_mh(im0, im1, motion_hist):





    
if __name__ == "__main__":
    mask0 = imread('mask-0.png')
    im0 = imread('im-0.png')
    im1 = imread('im-1.png')
    
    
