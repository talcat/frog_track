#batchcropy.py

from FrogFrames import FrogFrames
import numpy as np
from scipy.misc import imsave
import trackbyhand


PATH = '/home/talcat/Desktop/Bio Interface/Frogs/frog_frame/Shot2/'

def cutsmall():
    PATH = '/home/talcat/Desktop/Bio Interface/Frogs/frog_frame/Shot2/'

    video = FrogFrames(PATH, loop=False, gray=False)
                    
    im, idx = video.get_next_im()          

    while im != None:
           print idx
           #num = '%04d' % idx
           hei, wid, _ = im.shape
           im = im[hei/2:hei, wid/2:wid, :]
           imsave('%sim-%04d.png' %(PATH, idx), im)
           
           im, idx = video.get_next_im()
              
              

def cut_ROI(ROI_list, pathto, pathfrom=PATH, ):
    
    video = FrogFrames(pathfrom, loop=False, gray=False)
                    
    im, idx = video.get_next_im()          

    ROINUM = len(ROI_list)
    
    roi = 0
    while im != None and roi < ROINUM:
           print roi 
           #num = '%04d' % idx
           hei, wid, _ = im.shape
           im = im[ROI_list[roi].minr:ROI_list[roi].maxr, ROI_list[roi].minc:ROI_list[roi].maxc, :]
           imsave('%sim-%04d.png' %(pathto, idx), im)
           
           im, idx = video.get_next_im()
           roi += 1              
