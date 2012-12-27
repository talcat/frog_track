from matplotlib.widgets import  RectangleSelector
from pylab import *




def select_area(input_image):
    """Given an input image, will allow you to draw a rectangle arount a ROI
        returns the resulting mask
        hold down mouse: allows you to draw a rectangle
          - the resulting masked image will show below it
          - if you like that selection, press 'A' or 'a'
          - otherwise, redraw another rectangle, or press 'R' or 'r' to redo
    """
    im = input_image
    fig = figure()
    ax = subplot(211)
    ax.imshow(im)
    title('Input Image')
    bx = subplot(212)
    title('Masked Image')
    mask = ones(im.shape)
    bx.imshow(mask*im)

    
    def onselect(eclick, erelease):
      'eclick and erelease are matplotlib events at press and release'
      startx = eclick.xdata
      starty = eclick.ydata
      endx = erelease.xdata
      endy = erelease.ydata
  
      print ' startposition : (%f, %f)' % (eclick.xdata, eclick.ydata)
      print ' endposition   : (%f, %f)' % (erelease.xdata, erelease.ydata)
      print ' used button   : ', eclick.button

      mask = create_mask(startx, starty, endx, endy, im)
      #bx = subplot(212)
      #title('Masked Image')
      bx.imshow(mask*im)
      toggle_selector.RS.update()

    def toggle_selector(event):
        print ' Key pressed.'
        if event.key in ['A', 'a'] and toggle_selector.RS.active:
            print ' RectangleSelector deactivated - Mask accepted'
            toggle_selector.RS.set_active(False)
            return mask

            
        if event.key in ['R', 'r'] and toggle_selector.RS.active:
            print 'Redoing rectangle.'
            toggle_selector.RS.set_active(True)
            #bx = subplot(212)
            #title('Masked Image')
            bx.imshow(im)
            toggle_selector.RS.update()

    
    
    
    toggle_selector.RS = RectangleSelector(ax, onselect, drawtype='box')
    connect('key_press_event', toggle_selector)
    show()
        

    



def create_mask(startx, starty, endx, endy, im):
    """Return mask of same size input image that only shows selected box"""
    mask = zeros(im.shape)
    minc = min(startx, endx)
    minr = min(starty, endy)
    maxc = max(startx, endx)
    maxr = max(starty, endy)
    
    mask[minr:maxr, minc:maxc, :] = 1
    return mask
    
im = imread('test.png')
select_area(im)
    
