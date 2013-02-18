#batchcropy.py

from FrogFrames import FrogFrames
import numpy as np
from scipy.misc import imsave

PATH = '/home/talcat/Desktop/Bio Interface/Frogs/frog_frame/Shot2/'

video = FrogFrames(PATH,
                    loop=False, gray=False)
                    
im, idx = video.get_next_im()          

while im != None:
       print idx
       #num = '%04d' % idx
       hei, wid, _ = im.shape
       im = im[hei/2:hei, wid/2:wid, :]
       imsave('%sim-%04d.png' %(PATH, idx), im)
       
       im, idx = video.get_next_im()
              
