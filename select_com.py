# select_com.py

# Selects a COM of a given frame

from matplotlib import use
import numpy as np
use('Qt4Agg')
from matplotlib.widgets import  RectangleSelector
from matplotlib.ticker import LinearLocator, FixedLocator
from pylab import imread, imshow, show, figure, subplot, title, close, connect, Circle, subplot2grid
from scipy.misc import imsave



def select_COM(input_image, (minr, maxr), (minc, maxc)):
    """Given an input image, and its focussing ROI = [minr, maxr], [minc, maxc]
        will allow you to draw a rectangle around the point of interest (COM)
        returns the CENTER OF THE CHOSEN REC rounded to nearest coordinate IN THE
        FULL IMAGE SPACE (not subimage space)
        -----------------------------------
        |subim to draw on |point on subim |
        -----------------------------------
        |   point on full image           |    
        -----------------------------------
        USE: hold down mouse: allows you to draw a rectangle ON THE ROI IMAGE
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
  
    
    #grid setup
    #ax = subplot2grid( (2, 2), (0, 0) ) #subimage for selecting
    ax = subplot(221)
    #bx = subplot2grid( (2, 2), (0, 1)) #subimage showing point
    bx = subplot(222)
    cx = subplot2grid ( (2,2), (1, 0), colspan=2) #full image showing point
    
    def to_plot(orig_im, (minr, maxr), (minc, maxc), subim_pt=None):
        #subim_pt is the COM from the subimage = (row, col) in SUBIM COORD
        subim = orig_im[minr:maxr, minc:maxc, :]
        nrow, ncol, _ = orig_im.shape          
        #Subplots on top of each other
        #ax = subplot2grid( (2, 2), (0, 0) ) #subimage for selecting
        ax = subplot(221)
        
        ax.imshow(subim, interpolation='none', aspect='equal')
        title('Select Here')
        
        #bx = subplot2grid( (2, 2), (0, 1)) #subimage showing point
        bx = subplot(222)
        bx.clear()
        bx.imshow(subim, interpolation='none', aspect='equal')
        title('Shown COM')
               
        cx = subplot2grid ( (2,2), (1, 0), colspan=2) #full image showing point
        cx.clear()
        cx.imshow(orig_im, interpolation='none', aspect='equal') 
        title('Full Image')
        
        #Draw COM if we have one
        if subim_pt != None:
            ptr, ptc = subim_pt
            #draw a circle around the COM
            ## Circle goes (x, y) = (col, row)
            #In the subimage
            c = Circle( (ptc, ptr), 3, facecolor='none', edgecolor='red', linewidth=1)
            bx.add_patch(c)
            #in the full image
            ptr_full = ptr + minr
            ptc_full = ptc + minc
            c = Circle( (ptc_full, ptr_full), 4, facecolor='none', edgecolor='red', linewidth=1)
            cx.add_patch(c)
        
        #Set ticks to limits of image
        ax.xaxis.set_major_locator(LinearLocator(2))
        ax.yaxis.set_major_locator(LinearLocator(2))
        ax.autoscale(False)    
        bx.xaxis.set_major_locator(LinearLocator(2))
        bx.yaxis.set_major_locator(LinearLocator(2))
        bx.autoscale(False)
        cx.xaxis.set_major_locator(LinearLocator(2))
        cx.yaxis.set_major_locator(LinearLocator(2))
    
        #name them?
        row, col, _ = im.shape
        ax.xaxis.set_ticklabels([int(minc), int(maxc)])
        ax.yaxis.set_ticklabels([int(minr), int(maxr)])
        bx.xaxis.set_ticklabels([int(minc), int(maxc)])
        bx.yaxis.set_ticklabels([int(minr), int(maxr)])
        cx.xaxis.set_ticklabels([0, ncol])
        cx.yaxis.set_ticklabels([0, nrow])
    
        fig.canvas.draw() 
        fig.set_size_inches(20, 10, forward=True)
    
        return    
    
    to_plot(im, (minr, maxr), (minc, maxc), None)
    
    def return_COM(startx, starty, endx, endy, im):
        """Return COM point that is the center of the selected rectangle"""
        minc = np.min([startx, endx])
        minr = np.min([starty, endy])
        maxc = np.max([startx, endx])
        maxr = np.max([starty, endy])

        ptr = (maxr-minr)/2 + minr
        ptc = (maxc-minc)/2 + minc

        return np.round([ptr, ptc])
    
    
    def onselect(eclick, erelease):
      'eclick and erelease are matplotlib events at press and release'
      startx = eclick.xdata
      starty = eclick.ydata
      endx = erelease.xdata
      endy = erelease.ydata
  
      print ' startposition : (%f, %f)' % (eclick.xdata, eclick.ydata)
      print ' endposition   : (%f, %f)' % (erelease.xdata, erelease.ydata)
      print ' used button   : ', eclick.button

      toggle_selector.COM_sub = return_COM(startx, starty, endx, endy, im)
      ptr, ptc = toggle_selector.COM_sub
      toggle_selector.COM_ful = [ptr + minr, ptc + minc]
      to_plot(im, (minr, maxr), (minc, maxc), toggle_selector.COM_sub)
      toggle_selector.RS.update()

   
    
    def toggle_selector(event):
        print ' Key pressed.'
        if event.key in ['A', 'a'] and toggle_selector.RS.active:
            print ' RectangleSelector deactivated - Mask accepted'
            toggle_selector.RS.set_active(False)
            close()
            return toggle_selector.COM_ful

            
        if event.key in ['R', 'r'] and toggle_selector.RS.active:
            print 'Redoing rectangle.'
            toggle_selector.RS.set_active(True)
            toggle_selector.COM_sub = None
            
            #bx.imshow(toggle_selector.ROI, interpolation='none')
            to_plot(im, (minr, maxr), (minc, maxc), toggle_selector.COM_sub)
            toggle_selector.RS.update()

      
    toggle_selector.COM_sub = None
    toggle_selector.COM_ful = None
    toggle_selector.RS = RectangleSelector(ax, onselect, drawtype='box', 
                        rectprops = dict(facecolor='none', edgecolor = 'white',
                                         alpha=1, fill=False))
    
    connect('key_press_event', toggle_selector)
    show(block=True)
        
     
    return toggle_selector.COM_ful
    
    
if __name__ == "__main__":
    im = imread('im-0.png')
    rrange, crange = np.array([[523, 692],[62,437]])
    test = select_COM(im, rrange, crange)

    #imshow(im[minr:maxr, minc:maxc, :])
    show()
    print test
    
