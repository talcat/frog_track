# trackbyhand.py
#
# Final script that allows a user to draw a box aroun
# the ROI or point (care about the center of the box) in each frame.


from sel_prim_roi import * #contains select_prim_ROI function
from FrogFrames import *   #contains FrogFrames class
from select_com import *   #contains select_COM function  
import numpy as np 

DELTA_T = .002
PATH = '/home/talcat/Desktop/Bio Interface/Frogs/frog_frame/Shot2/'

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class DefineError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expr -- input expression in which the error occurred
        msg  -- explanation of the error
    """

    def __init__(self, expr, msg):
        self.expr = expr
        self.msg = msg


class RecROI(np.ndarray):
    """Easy way to save the ROI in an object"""
    def __new__(cls, nparray):
        """ This allows for the ROIs to just be numpy arrays, but have some nice
        accessor functions
        """		
        
        obj = np.asarray(nparray).view(cls)
        shape = nparray.shape
        if shape != (2,2):
            raise DefineError(obj, ' is not a (2,2) array or list')
            return
        else:
            (obj.minr, obj.maxr), (obj.minc, obj.maxc) = obj
            return obj
            return

	def __array_finalize__(self, obj):
		if obj is None: return
		self.minr = getattr(obj, 'minr', None) 
        self.maxr = getattr(obj, 'maxr', None)
        self.minc = getattr(obj, 'minc', None)
        self.maxc = getattr(obj, 'maxc', None)
        
    def __getstate(self):
        return {'minr':self.minr, 'maxr': self.maxr, 'minc':self.minc, 'maxc':self.maxc}

class Point(np.ndarray):
    """Easy way to save points (like the com) into a single obj"""
    def __new__(cls, nparray):
        obj = np.asarray(nparray).view(cls)
        obj.row, obj.col = obj
        obj.x = obj.col
        obj.y = obj.row
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.x = getattr(obj, 'x', None)
        self.y = getattr(obj, 'y', None)
        self.col = self.x
        self.row = self.y        
    #def __getstate__(self):
    #    return {'x': self.x, 'y':self.y, 'col': self.col, 'row':self.row}    
    

def move_roi(roiobj, prev_ptobj, next_ptobj, height, width):
    """Returns a new ROI that has moved depending on the com points selected"""
    p_ptr, p_ptc = prev_ptobj
    n_ptr, n_ptc = next_ptobj
    
    move_r = n_ptr - p_ptr
    move_c = n_ptc - p_ptc
    
    #print (move_r, move_c)
    
    if ((roiobj.minr + move_r <=0 ) or (roiobj.maxr + move_r >= height) or
       (roiobj.minc + move_c <= 0) or  (roiobj.maxc + move_c >= width)):
        moved_roi = roiobj   
    else:
        moved_roi = RecROI( np.array([[roiobj.minr + move_r, roiobj.maxr + move_r],     
                                      [roiobj.minc + move_c, roiobj.maxc + move_c]]))
    return moved_roi
                        


if __name__ == "__main__":
    video = FrogFrames(PATH, loop=False, gray=False, eq=False  )
    
    #ROI List
    # | 0 | 1 | 2 | 3 | 4 | ..  <- Frame number
    # | 0 | 0 | 1 | 2 | 3 | ..  <- ROI (where num is the frame num that it was 
    #                                   computed during)
    ROI_list = []
    
    COM_list = []
    
    TIMESTAMP_list = []
    
    cond = True
    while cond:
        frame, idx = video.get_next_im()
        
        #If frame == None, no more files:
        if frame == None:
            cond = False
            break
        
        #Get the ROI
        if idx == 0:   #If first frame, initialize it
            ROI = select_prim_ROI(frame)
            ROI_obj = RecROI(ROI)
            ROI_list.append(ROI_obj) #ROI_list[0] = original ROI

        else: #Otherwise, use previously computed ROI (at index)      
            ROI = ROI_list[idx]    
        
        # Get the point of com
        rows, cols = ROI
        COM, ROI = select_COM(frame,rows, cols)
        
        #If manually moved ROI, reset it
        ROI_obj = RecROI(ROI)
        ROI_list[idx] = ROI_obj
        
        if COM == None: # No point selected (probably out of frame):
            cond = False
            break
            
        COM_obj = Point(COM) 
        #print COM_obj    
        
        #append it to the list
        COM_list.append(COM_obj)
        
        print 'idx = %d, len(COM_list) = %d' %(idx, len(COM_list))
        #if len(COM_list) != idx:
        #    print 'Error Will Robinson'
        
        #record the timstamp
        TIMESTAMP_list.append(idx*DELTA_T)
        
        #we don't generate a new ROI for frame 2:
        if idx == 0:
            #Now, we want frame 1 to have the original ROI as well:
            ROI_list.append(ROI_obj) #ROI_list[1] = original ROI
            continue #We are done with frame 0
        #calculate next roi for frame 1 --> end
        prevpt = COM_list[idx - 1]
        newroi = move_roi(ROI_obj, prevpt, COM_obj, frame.shape[0], frame.shape[1])
        #now we want to save that such that it is in idx +1 slot for next round
        ROI_list.append(newroi)
        
        # if len(ROI_list) != idx + 1:
        #    print 'Wrong >_<'
        print 'idx = %d, len(ROI_list) = %d' %(idx, len(ROI_list))
        
        print idx    
        string = '%s%s' % (video.path, video.list[video.index])
        print string
            
