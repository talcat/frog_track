# updateroi.py

from matplotlib import use
use('Qt4Agg')
from matplotlib.widgets import  RectangleSelector
from pylab import *
import cv2
from recsel import create_mask


def updateroi(mask0, change):
    """Given the vector flow from 2 images (change = [x, y]) and the mask to 
    change, returns a mask that has been moved by that amount"""
    
    #first map the change vector to int to move the mask
    change = np.array(change).astype('int')
        
    #get roi corners
    r0, r1, c0, c1 = returncorners(mask0)
    
    #move the corners appropriately
    r0 = r0 + change[1]
    r1 = r1 + change[1]
    c0 = c0 + change[0]
    c1 = c1 + change[0]    
    
    #get the mask
    mask = create_mask(c0, r0, c1, r1, mask0)
            
    return mask
    

def returncorners(mask):
    """Given a mask, finds the and returns the four corners"""
    mask = mask.astype('uint8')
    maxval = np.max(mask)

    if len(mask.shape) == 3:
        dim = mask.shape[-1]
        if dim == 3:
            #mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            mask = mask[:, :, 0]

    row, col = nonzero(mask==maxval)
    
    return [min(row), max(row), min(col), max(col)]


    
if __name__ == "__main__":
    mask0 = imread('mask-0.png')
    im0 = imread('im-0.png')
    im1 = imread('im-1.png')
    
    
    next, stat, err = updateroi(mask0, im0, im1)
    
    
    

