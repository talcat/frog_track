#sel_prim_roi.py

from matplotlib import use
import numpy as np
use('Qt4Agg')
from matplotlib.widgets import  RectangleSelector
from matplotlib.ticker import LinearLocator, FixedLocator
from pylab import imread, imshow, show, figure, subplot, title, close, connect, axes, gca
from scipy.misc import imsave



def select_prim_ROI(input_image):
    """Given an input image, will allow you to draw a rectangle arount a ROI
        returns the resulting bounds of the initial ROI
        hold down mouse: allows you to draw a rectangle
          - the resulting SUBIMAGE will show below it
          - if you like that selection, press 'A' or 'a'
          - otherwise, redraw another rectangle, or press 'R' or 'r' to redo
    """
    
    print 'Use a mouse to draw a rectangle around the region of interest!'
    print ''.join(['If you want to redraw the rectangle, press "R", "r", or just',   
          ' redraw it directly!'])
    print 'Press "A" or "a" to return the mask you selected!'
           
    
    im = input_image
    fig = figure(1)
    
    start_ROI = np.ones(im.shape)
    st_row, st_col, _ = start_ROI.shape
    ax = subplot(211) 
    
    def to_plot(orig_im, (minr, maxr), (minc, maxc)):
          
        #Subplots on top of each other
        ax = subplot(211) 
        ax.imshow(orig_im, interpolation='none')
        title('Input Image')
        bx = subplot(212)
        title('Selected ROI')
        subim = orig_im[minr:maxr, minc:maxc, :]    
        bx.imshow(subim, interpolation='none')
    
        #Set ticks to limits of image
        ax.xaxis.set_major_locator(LinearLocator(2))
        ax.yaxis.set_major_locator(LinearLocator(2))    
        bx.xaxis.set_major_locator(LinearLocator(2))
        bx.yaxis.set_major_locator(LinearLocator(2))
    
        #name them?
        row, col, _ = im.shape
        ax.xaxis.set_ticklabels([0, col])
        ax.yaxis.set_ticklabels([0, row])
        bx.xaxis.set_ticklabels([int(minc), int(maxc)])
        bx.yaxis.set_ticklabels([int(minr), int(maxr)])
    
        fig.canvas.draw() 
        fig.set_size_inches(20, 10, forward=True)
        
        return    
    
    to_plot(im, (0, st_row), (0, st_col))
    
    def return_ROI(startx, starty, endx, endy, im):
        """Return mask of same size input image that only shows selected box"""
        minc = np.min([startx, endx])
        minr = np.min([starty, endy])
        maxc = np.max([startx, endx])
        maxr = np.max([starty, endy])

        return im[minr:maxr, minc:maxc, :], np.round([(minr, maxr), (minc, maxc)])
    
    
    def onselect(eclick, erelease):
      'eclick and erelease are matplotlib events at press and release'
      startx = eclick.xdata
      starty = eclick.ydata
      endx = erelease.xdata
      endy = erelease.ydata
  
      print ' startposition : (%f, %f)' % (eclick.xdata, eclick.ydata)
      print ' endposition   : (%f, %f)' % (erelease.xdata, erelease.ydata)
      print ' used button   : ', eclick.button

      toggle_selector.ROI, toggle_selector.range = return_ROI(startx, starty, endx, endy, im)
      rows, cols = toggle_selector.range
      #bx.imshow(toggle_selector.ROI, interpolation='none')
      to_plot(im, rows, cols)
      toggle_selector.RS.update()

   
    
    def toggle_selector(event):
        print ' Key pressed.'
        if event.key in ['A', 'a'] and toggle_selector.RS.active:
            print ' RectangleSelector deactivated - Mask accepted'
            toggle_selector.RS.set_active(False)
            close()
            return toggle_selector.ROI

            
        if event.key in ['R', 'r'] and toggle_selector.RS.active:
            print 'Redoing rectangle.'
            toggle_selector.RS.set_active(True)
            toggle_selector.ROI = start_ROI
            
            #bx.imshow(toggle_selector.ROI, interpolation='none')
            to_plot(im, (0, st_row), (0, st_col))
            toggle_selector.RS.update()

      
    toggle_selector.ROI = start_ROI
    toggle_selector.RS = RectangleSelector(ax, onselect, drawtype='box', 
                rectprops = dict(facecolor='none', edgecolor = 'white',
                                alpha=1, fill=False))
    connect('key_press_event', toggle_selector)
    show(block=True)
        
     
    return toggle_selector.range
    
    
if __name__ == "__main__":
    im = imread('im-0.png')
    test = select_prim_ROI(im)
    (minr, maxr), (minc, maxc) = test
    imshow(im[minr:maxr, minc:maxc, :])
    show()
    print test
    

