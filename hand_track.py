# select_com.py

# Selects a COM of a given frame

from matplotlib import use
import numpy as np
use('Qt4Agg')
from matplotlib.widgets import  RectangleSelector
from matplotlib.ticker import LinearLocator, FixedLocator
from pylab import imread, imshow, show, figure, subplot, title, close, connect, Circle, subplot2grid, Rectangle, Line2D
from scipy.misc import imsave
from FrogFrames import *


def hand_track(frog_frame, (minr, maxr), (minc, maxc), draw='box', list_ROI=None, list_PTS = None, list_PTS_un=None, list_line=None):
    """Given an intial frog_frame object and its focussing ROI = [minr, maxr], [minc, maxc]
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
    

    


    
    def to_plot(im, (minr, maxr), (minc, maxc), fulim_pt=None, ax_clear=False):
        fig = toggle_selector.fig
        #filim_pt is the COM from the subimage = (row, col) in FULL IM COORD
        subim = im[minr:maxr, minc:maxc]
        nrow, ncol = im.shape[0:2]          
        #Subplots on top of each other
        #ax = subplot2grid( (2, 2), (0, 0) ) #subimage for selecting
        ax = subplot(221)
        if ax_clear: #This is if we zoomed in or out:
            ax.autoscale()
        #ax.autoscale()    
        ax.imshow(subim, interpolation='none', aspect='equal')
        title('Select Here')
        
        #bx = subplot2grid( (2, 2), (0, 1)) #subimage showing point
        bx = subplot(222)
        #bx.autoscale()
        bx.clear()
        bx.imshow(subim, interpolation='none', aspect='equal')
        title('Selected Point')
               
        cx = subplot2grid ( (2,2), (1, 0), colspan=2) #full image showing point
        cx.clear()
        #cx.autoscale()
        cx.imshow(im, interpolation='none', aspect='equal') 
        title('Frame %d/%d'%(toggle_selector.idx + 1, frog_frame.num_frames), fontsize=14)
        
        
        #Draw box around the current ROI:
        theROI = Rectangle( (minc, minr ), (maxc-minc), (maxr - minr), edgecolor='white', fill=False, linewidth=1 )
        #cx.add_patch(theROI) 
        cx.add_patch(theROI)
        #cx.draw_artist(theROI)


        #Draw Traked point if we have one
        if fulim_pt != None:
            ptr_full, ptc_full = fulim_pt
            #get the subim points
            ptr = ptr_full - minr
            ptc = ptc_full - minc
            #draw a circle around the COM
            ## Circle goes (x, y) = (col, row)
            #In the subimage
            c = Circle( (ptc, ptr), .5, facecolor='white', edgecolor='white', fill=True, linewidth=5)
            bx.add_patch(c)
            #in the full image
            c = Circle( (ptc_full, ptr_full), 4, facecolor='none', edgecolor='white', linewidth=2)
            cx.add_patch(c)
            #cx.draw_artist(c)
        
        #Draw uncertainty rectrangle if have one
        if toggle_selector.COM_rect is not None:
            (rminr, rmaxr), (rminc, rmaxc) = toggle_selector.COM_rect
            if draw is 'line' and toggle_selector.line is not None:
                #draw a line as well?
                (startx, endx), (starty, endy) = toggle_selector.line
                cx.add_line(Line2D([startx, endx], [starty, endy], linewidth=2, color='white'))
                bx.add_line(Line2D([startx - minc, endx - minc], [starty - minr, endy - minr], linewidth=2, color='white'))

            rminr = rminr - minr
            rmaxr = rmaxr - minr
            rminc = rminc - minc    
            rmaxc = rmaxc - minc
        
            the_un = Rectangle( (rminc, rminr), (rmaxc - rminc), (rmaxr - rminr), edgecolor='red', fill = False, linewidth = 2)
            bx.add_patch(the_un)
            #bx.draw_artist(the_un)

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
        row, col = im.shape[0:2]
        ax.xaxis.set_ticklabels([int(minc), int(maxc)])
        ax.yaxis.set_ticklabels([int(minr), int(maxr)])
        bx.xaxis.set_ticklabels([int(minc), int(maxc)])
        bx.yaxis.set_ticklabels([int(minr), int(maxr)])
        cx.xaxis.set_ticklabels([0, ncol])
        cx.yaxis.set_ticklabels([0, nrow])
        
        
        bx.figure.canvas.draw()
        cx.figure.canvas.draw()
        #fig.canvas.draw() 
        #canvas.blit(bx.bbox)
        #canvas.blit(cx.bbox)
        fig.set_size_inches(20, 10, forward=True)
    
        return    
    
    
     
    
    
    def return_COM(startx, starty, endx, endy):
        """Return COM point that is the center of the selected rectangle"""
        minc = np.min([startx, endx])
        minr = np.min([starty, endy])
        maxc = np.max([startx, endx])
        maxr = np.max([starty, endy])

        ptr = (maxr-minr)/2 + minr
        ptc = (maxc-minc)/2 + minc

        #return np.round([ptr, ptc])
        return [ptr, ptc]
    
    def onselect(eclick, erelease):
      'eclick and erelease are matplotlib events at press and release'
      startx = eclick.xdata
      starty = eclick.ydata
      endx = erelease.xdata
      endy = erelease.ydata
  
      print ' startposition : (%f, %f)' % (eclick.xdata, eclick.ydata)
      print ' endposition   : (%f, %f)' % (erelease.xdata, erelease.ydata)
      print ' used button   : ', eclick.button

      toggle_selector.COM_sub = return_COM(startx, starty, endx, endy)
      ptr, ptc = toggle_selector.COM_sub
      (minr, maxr), (minc, maxc) = toggle_selector.ROI
      toggle_selector.COM_ful = [ptr + minr, ptc + minc]
      #Uncertainty in picked point is the selected rectangle
      un_minc = np.min([startx, endx])
      un_minr = np.min([starty, endy])
      un_maxc = np.max([startx, endx])
      un_maxr = np.max([starty, endy])
      toggle_selector.COM_rect = np.array([[un_minr + minr, un_maxr + minr],
                                           [un_minc + minc, un_maxc + minc]])
      #if it is a line!
      toggle_selector.line = np.array([[startx + minc, endx + minc], [starty + minr, endy + minr]])
      img = toggle_selector.img
      (minr, maxr), (minc, maxc) = toggle_selector.ROI
      to_plot(img, (minr, maxr), (minc, maxc), toggle_selector.COM_ful)
      toggle_selector.RS.update()

   
    def move_ROI(direc):
        """Moves the ROI in 10% of the wid/height direction the arrow keys 
        show"""
        (minr, maxr), (minc, maxc) = toggle_selector.ROI
        wid_mv = int((maxc - minc)/10)
        hei_mv = int((maxr - minr)/10)
        if direc == 'down':
            if maxr + hei_mv <= nrow:
                minr = minr + hei_mv
                maxr = maxr + hei_mv
        if direc == 'up':
            if minr - hei_mv >= 0:
                minr = minr - hei_mv
                maxr = maxr - hei_mv
        if direc == 'left':
            if minc - wid_mv >= 0:
                minc = minc - wid_mv
                maxc = maxc - wid_mv
        if direc == 'right':
            if maxc + wid_mv<= ncol:
                minc = minc + wid_mv
                maxc = maxc + wid_mv
        toggle_selector.ROI = np.array([ [minr, maxr], [minc, maxc]   ])
        return toggle_selector.ROI
    
    def zoom_ROI(direc):
        """Zooms the ROI in or out 10%"""
        (minr, maxr), (minc, maxc) = toggle_selector.ROI
        wid_5 = int((maxc - minc)/20)
        hei_5 = int((maxr - minr)/20)
        if direc == 'in':
            #Zooming in
            #We need to check that we don't zoom in too much >_<
            if minr + hei_5 < maxr -hei_5:
                minr = minr + hei_5
                maxr = maxr - hei_5
            if minc + wid_5 < maxr - wid_5:    
                minc = minc + wid_5
                maxc = maxc - wid_5
        if direc == 'out':
            #Zooming out
            #we need to make sure we are zooming out within the feasibility of the image >_<
            if minr - hei_5 <= 0: 
                minr = 0
            else:
                minr = minr - hei_5
            if maxr + hei_5 >= nrow:
                maxr = nrow
            else:
                maxr = maxr + hei_5
            if minc - wid_5 <= 0:
                minc = 0
            else:
                minc = minc - wid_5
            if maxc + wid_5 >= ncol:
                maxc = ncol
            else:
                maxc = maxc + wid_5         
        toggle_selector.ROI = np.array([[minr, maxr], [minc, maxc]])        
        return toggle_selector.ROI

    def update_ROI((p1r, p1c), (p2r, p2c), (minr, maxr), (minc, maxc)):
        """Using the prev and prev prev points, moves the region of interest to a new predicted region"""
        mover = p1r - p2r
        movec = p1c - p2c

        #Check bounds again => cannot move new ROI outside the image
        if (minr + mover) >=0 and (maxr + mover) <= nrow:
            minr = minr + mover
            maxr = maxr + mover
        if (minc + movec) >= 0 and (maxc + movec) <= ncol:
            minc = minc + movec
            maxc = maxc + movec
        return (minr, maxr), (minc, maxc)    
    
    def save_frame_info():
        idx = toggle_selector.idx
        
        #Pts only get saved if they exist.  
        if toggle_selector.COM_ful is not None:
            (ptr, ptc) = toggle_selector.COM_ful
            toggle_selector.list_PTS[idx] = [ptr, ptc]
        if toggle_selector.COM_rect is not None:
            toggle_selector.list_PTS_un[idx] = toggle_selector.COM_rect

        if toggle_selector.line is not None:
            toggle_selector.list_line[idx] = toggle_selector.line
        if toggle_selector.COM_rect is not None:
            toggle_selector.list_PTS_un[idx] = toggle_selector.COM_rect    

        (minr, maxr), (minc, maxc) = toggle_selector.ROI
        toggle_selector.list_ROI[idx] = np.array([ [minr, maxr], [minc, maxc]])


    def toggle_selector(event):
        """ALL USER INPUT IS HANDLED HERE BABY"""
        save_flag = False
        #print ' Key pressed.'
        if event.key in ['A', 'a'] and toggle_selector.RS.active:
            print 'Saving point and ROI and proceeding to next frame'
            save_frame_info()
            save_flag = True
             
            
        elif event.key in ['R', 'r'] and toggle_selector.RS.active:
            #Reset current frame's tracked point
            print 'Redoing COM.'
            toggle_selector.RS.set_active(True)
            toggle_selector.COM_sub = None
            image = toggle_selector.img
            (minr, maxr), (minc, maxc) = toggle_selector.ROI
            to_plot(image, (minr, maxr), (minc, maxc), None)
            toggle_selector.COM_ful = None
            toggle_selector.COM_sub = None
            toggle_selector.RS.update()
            
        
        elif event.key in ['up', 'down', 'left', 'right']:
            # Moving an ROI manually with keys up down left right
            (minr, maxr), (minc, maxc) = move_ROI(event.key)
            image = toggle_selector.img
            to_plot(image, (minr, maxr), (minc, maxc), toggle_selector.COM_ful)
            toggle_selector.RS.update()
        
        elif event.key is 'escape':
            #Close program
            toggle_selector.RS.set_active(False)
            close()
            return  toggle_selector.list_ROI, toggle_selector.list_PTS, toggle_selector.list_PTS_un

        elif event.key in ['ctrl+up', 'ctrl+down']:  
            #zooming the ROI in and out
            print event.key
            zoom_state = ['in', 'out']
            stat = zoom_state[['ctrl+up', 'ctrl+down'].index(event.key)]
            print 'zoom %s'%(stat)
            (minr, maxr), (minc, maxc) = zoom_ROI(stat)
            image = toggle_selector.img
            to_plot(image, (minr, maxr), (minc, maxc), toggle_selector.COM_ful, ax_clear=True)
            toggle_selector.RS.update()    
        
        if event.key in ['ctrl+left', 'ctrl+right'] or save_flag:
            print event.key
            if save_flag:
                stat = 'right'
            else:
                stat = ['left', 'right'][['ctrl+left', 'ctrl+right'].index(event.key)]
            idx = toggle_selector.idx 
            print "Moving frame %s"%(stat)
            if stat is 'left':
                if idx - 1 >= 0:
                    toggle_selector.idx = idx - 1
                    toggle_selector.img = frog_frame.get_frame(idx - 1)
                else:
                    print "First Frame REACHED!"
            elif stat is 'right':
                max_frame = frog_frame.num_frames
                if idx + 1 <= max_frame:
                    toggle_selector.idx = idx + 1
                    toggle_selector.img = frog_frame.get_frame(idx + 1)      
                else:
                    print "Last Frame REACHED!"  

            #OK now that we moved frames, lets see if an ROI or COM has been set already for this frame 
            img = toggle_selector.img
            idx = toggle_selector.idx
            if idx in toggle_selector.list_ROI.keys():
                toggle_selector.ROI = toggle_selector.list_ROI[idx] #An ROI for this frame has previously been saved
            else:
                #Lets check to see if we have 2 points available to move the ROI:
                if idx - 1 in toggle_selector.list_PTS.keys() and idx - 2 in toggle_selector.list_PTS.keys():
                    p1r, p1c = toggle_selector.list_PTS[idx -1]
                    p2r, p2c = toggle_selector.list_PTS[idx - 2]
                    (minr, maxr), (minc, maxc) = toggle_selector.ROI
                    #Now we update the ROI based on our previous knowledge
                    (minr, maxr), (minc, maxc) = update_ROI((p1r, p1c), (p2r, p2c), (minr, maxr), (minc, maxc))
                    toggle_selector.ROI = (minr, maxr), (minc, maxc)
                #Otherwise, we just use the current ROI as our guess for the new one

            (minr, maxr), (minc, maxc) = toggle_selector.ROI

            #Now, has their been a point set for this new frame?
            if idx in toggle_selector.list_PTS.keys():
                toggle_selector.COM_ful = toggle_selector.list_PTS[idx]
                ptr, ptc = toggle_selector.COM_ful
                toggle_selector.COM_sub = [ptr - minr, ptc - minc]
            else: #there hasn't!
                toggle_selector.COM_ful = None
                toggle_selector.COM_sub = None
            #similarly for the pts uncertainty
            if idx in toggle_selector.list_PTS_un.keys():
                toggle_selector.COM_rect = toggle_selector.list_PTS_un[idx]
            else:
                toggle_selector.COM_rect = None
            # and lines
            if idx in toggle_selector.list_line.keys():
                toggle_selector.line = toggle_selector.list_line[idx]
            else:
                toggle_selector.line = None
            to_plot(img, (minr, maxr), (minc, maxc), toggle_selector.COM_ful)   
            toggle_selector.RS.update()       

        else:
            print "%s key does not do anything!"%(event.key)
            
    #############################
    # OK Now that all those functions are out of the way, lets set this shit up
    ##############################
    #Get initial image to draw:
    init_im = frog_frame.get_frame(0)
    nrow, ncol = init_im.shape[0:2] 
    fig = figure(1)
    toggle_selector.fig = fig
    #find 10% movement in up/down or left/right for ROI moving purposes

    
    #wid/hei of subimage to make dummy subim
    wid = maxc-minc
    hei = maxr - minr
    
    #grid setup
    #ax = subplot2grid( (2, 2), (0, 0) ) #subimage for selecting
    ax = subplot(221) #Where you are selecting
    #bx = subplot2grid( (2, 2), (0, 1)) #subimage showing point
    bx = subplot(222) #shows zoomed final selection
    cx = subplot2grid ( (2,2), (1, 0), colspan=2) #full image showing point
    
    

    #class/state variables, essentially
    toggle_selector.idx = 0   
    toggle_selector.img = frog_frame.get_frame(toggle_selector.idx)
    toggle_selector.ROI = np.array([ [minr, maxr], [minc, maxc ]  ])
    toggle_selector.COM_rect = None  
    toggle_selector.COM_sub = None
    toggle_selector.COM_ful = None
    toggle_selector.RS = RectangleSelector(ax, onselect, drawtype=draw, #useblit=True,
                        rectprops = dict(facecolor='white', edgecolor = 'white',
                                         alpha=1, fill=False), 
                        lineprops = dict(color='white', linestyle='-',
                                         linewidth = 2, alpha=0.5))
    toggle_selector.line = None

    # Now to set up the lists of POINTS OF INTEREST (PTS) and ROIS
    # these are kept in dicts, with the keys being the frame numbers (holy shit man)
    # Stuff is only saved to these guys when save_frame_info is called (ie. when you press a or A)
    # This way, if the current point being tracked is not in the first n frames, or some random frame
    # in the middle , it simply does not exist. WOOOOOOOO
    if list_ROI is not None:
        toggle_selector.list_ROI = list_ROI
        if toggle_selector.idx in list_ROI.keys(): toggle_selector.ROI = list_ROI[toggle_selector.idx]
    else:
        toggle_selector.list_ROI = dict()
    if list_PTS is not None:
        toggle_selector.list_PTS = list_PTS
        if toggle_selector.idx in list_PTS.keys(): toggle_selector.COM = list_PTS[toggle_selector.idx]
    else:
        toggle_selector.list_PTS = dict()
    if list_PTS_un is not None:
        toggle_selector.list_PTS_un = list_PTS_un
        if toggle_selector.idx in list_PTS_un.keys(): toggle_selector.COM_rect = list_PTS_un[toggle_selector.idx]
    else:
        toggle_selector.list_PTS_un = dict()    
    if list_line is not None:
        toggle_selector.list_line = list_line
    else:
        toggle_selector.list_line = dict()


    #Plot the initial image finally - everything else is a key handle
    to_plot(init_im, (minr, maxr), (minc, maxc), None)


    connect('key_press_event', toggle_selector)
    show(block=True)
        
     
    return  toggle_selector.list_ROI, toggle_selector.list_PTS, toggle_selector.list_PTS_un, toggle_selector.list_line
    
    
if __name__ == "__main__":
    PATH = '/home/talcat/Desktop/Bio_Interface/Frogs/frog_frame/Shot2/'
    rrange, crange = np.array([[523, 692],[62,437]])
    video = FrogFrames(PATH, loop=False, wantgray=False)

    test = hand_track(video, rrange, crange, draw='line')

    #imshow(im[minr:maxr, minc:maxc, :])
    show()
    print test
    

