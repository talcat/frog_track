# label.py

#labels images with timestamp, saves a video using matplotlib (hopefully)

from FrogFrames import FrogFrames
import numpy as np
from scipy.misc import imsave
import matplotlib.animation as manimation
import matplotlib.pyplot as plt

PATH = '/home/talcat/Desktop/Bio Interface/Frogs/frog_frame/Shot2/'
DELTA_T = .002
DPI = 800

video = FrogFrames(PATH, loop=False, gray=False)

#set up to write video:
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='timestamps', artist='matplotlib', comment='who knows')
writer = FFMpegWriter(fps=15, metadata = metadata)

                    
im, idx = video.get_next_im()      

hei, wid = im.shape[0:2]    

figsize = (wid/np.float(DPI), hei/np.float(DPI))
fig1 = plt.figure(figsize = figsize, dpi= DPI, frameon=False)
ax_size=[0,0,1,1]
fig1.add_axes(ax_size)
plt.axis('off')
ax = fig1.get_axes()[0]

axesim = plt.imshow(im)
t = plt.text(0.1, .95, '0', color='white', fontsize=5, horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes) 

with writer.saving(fig1, '%stimestamp_subvid.mp4' %PATH, DPI):

    while im != None:
           print idx
           
           axesim.set_data(im)
        
           time = '%05.3f sec' % (idx*DELTA_T)
           t.set_text(time)
           print time
        
            
           writer.grab_frame()
           
                      
           im, idx = video.get_next_im()
