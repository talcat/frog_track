#!/usr/bin/env python

'''
Watershed segmentation
=========

This program demonstrates the watershed segmentation algorithm
in OpenCV: watershed().

Usage
-----
watershed.py [image filename]

Keys
----
  1-7   - switch marker color
  SPACE - update segmentation
  r     - reset
  a     - toggle autoupdate
  ESC   - exit

'''




import numpy as np
import cv2
from common import Sketcher, RectSelector
from FrogFrames import *

class App:
    def __init__(self, img):
        self.img = img
        h, w = self.img.shape[:2]
        self.markers = np.zeros((h, w), np.int32)
        self.markers_vis = self.img.copy()
        self.cur_marker = 1
        self.colors = np.int32( list(np.ndindex(2, 2, 2)) ) * 255

        self.auto_update = True
        self.sketch = Sketcher('img', [self.markers_vis, self.markers], self.get_colors)

        self.segment = None

    def get_colors(self):
        return map(int, self.colors[self.cur_marker]), self.cur_marker

    def watershed(self):
        m = self.markers.copy()
        cv2.watershed(self.img, m)
        overlay = self.colors[np.maximum(m, 0)]
        self.segment = m
        vis = cv2.addWeighted(self.img, 0.5, overlay, 0.5, 0.0, dtype=cv2.CV_8UC3)
        cv2.imshow('watershed', vis)
        cv2.moveWindow('watershed', vis.shape[1]+10, 90)

    def run(self):
        while True:
            ch = 0xFF & cv2.waitKey(50)
            if ch == 27:
                print 'Exiting'
                break

            if ch >= ord('1') and ch <= ord('7'):
                self.cur_marker = ch - ord('0')
                print 'marker: ', self.cur_marker
            if ch == ord(' ') or (self.sketch.dirty and self.auto_update):
                self.watershed()
                self.sketch.dirty = False
            #if ch in [ord('a'), ord('A')]:
            #    self.auto_update = not self.auto_update
            #    print 'auto_update if', ['off', 'on'][self.auto_update]
            if ch in [ord('r'), ord('R')]:
                self.markers[:] = 0
                self.markers_vis[:] = self.img
                self.segment = None
                self.sketch.show()
            if ch in [ord('a'), ord('A')]: 
                print "Sementation Accepted"
                #cv2.destroyAllWindows()
                segment = self.segment
                return segment, self.markers
            if ch in [ord('z'), ord('Z')]: 
                print "Skipping Frame"
                segment = 's'
                return segment, None
        #cv2.destroyAllWindows()


#if __name__ == '__main__':
    # import sys
    # try:
    #     fn = sys.argv[1]
    # except:
    #     fn = '../cpp/fruits.jpg'
    # print __doc__
    # App(fn).run()
def watershed_all_frames(video, fit='fill'):
    numframes = video.num_frames
    contours = dict()
    ellipses = dict()
    angles = dict()
    for fr in range(numframes):
        im = video.get_frame(fr)
        #convert to bgr
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        
        segment, _ = App(im).run()
        todraw = im.copy()
        #get black/white image from segment.  1 is foregraound
        if segment is None:
            break
        if segment is 's':
            continue # this should skip this frame 
        try:
            mask = np.zeros(segment.shape[:2])
            h, w = segment.shape[:2]
            #mask[segment == -1] = 255
            mask[segment == 1] = 255
                        
            mask = mask.astype('uint8')
            
            #fill = np.transpose(np.nonzero(mask == 255))    

            #get contours from that segmented image
            cont = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #cont[1] is the list of points
            #save list of contour points
            contours[fr] = cont[1]
            #draw them
            cv2.drawContours(todraw, cont[1], -1, (255, 0, 0), 2)

            ###fit an ellipse to the contours
            #ell = cv2.fitEllipse(cont[1][0])
            ##this is the rotated rectange:
            ##  (center x, center y), (width, height), angle from column (90 is straight up, 0 is horizontal)
            #ellipses[fr] = ell
            #angles[fr] = 90 - ell[2]

            #Fitting ellipse to filled contour
            empty = np.zeros(mask.shape[:2]).astype('uint8')
            print 'to draw contour'
            cv2.drawContours(empty, cont[1], -1, 255, -1)
            print 'contour drawn'
            

            fill = np.transpose(np.nonzero(empty == 255))
            fill = map(lambda x: [x[1], x[0]], fill)
            
            ell = cv2.fitEllipse(np.array(fill))

            ellipses[fr] = ell
            angles[fr] = 90 - ell[2]



            #draw the ellipse
            cv2.ellipse(todraw, ell, (0, 255, 0), 2)

            cv2.imshow('todraw', todraw)

            k = cv2.waitKey(1000)
            if k == 27: #escp
                break
        except: 
            print "Something went wrong on this frame"
            pass

    ##fix ellipse    
    #if fit is 'fill':
    #    ell2 = dict()
    #    ang2 = dict()
    #    for k in contours.keys():
    #        ell2[k] = ellipseFit(contours[k])
    #        ang2[k] = 90 - ell2[k]
    #    angles = ang2
    #    ellipses = ell2    
    cv2.destroyAllWindows()

    return contours, ellipses, angles


def ellipseFit(c):
    #fits an ellipse to the filled contour area, instead of the contour
    empty = np.zeros((720, 1280)).astype('uint8')
    cv2.drawContours(empty, c, -1, 255, -1)
    fill = np.transpose(np.nonzero(empty == 255))
    fill = map(lambda x: [x[1], x[0]], fill)
    ell = cv2.fitEllipse(np.array(fill))
    return ell


def get_horizon(vid):
    """Given a vid, calc's avg frame, does Canny Edge detection, and fits a line to the horizon"""

    global rectangle, rect_over, rect
    rectangle = False       # flag for drawing rect
    rect_over = False       # flag to check if rect drawn
    rect = (0, 0, 1, 1)

    def onmouse(event,x,y,flags,param):
        global toshow,vis,drawing,value,mask,rectangle,rect,rect_or_mask,ix,iy,rect_over
        
        BLUE = [255,0,0]        # Draw Rectangle

        if event == cv2.EVENT_RBUTTONDOWN:
            rectangle = True
            ix,iy = x,y
    
        elif event == cv2.EVENT_MOUSEMOVE:
            if rectangle == True:
                vis = toshow.copy()
                cv2.rectangle(vis,(ix,iy),(x,y),BLUE,2)
                rect = (ix,iy,x,y)
                rect_or_mask = 0
    
        elif event == cv2.EVENT_RBUTTONUP:
            rectangle = False
            rect_over = True
            cv2.rectangle(vis,(ix,iy),(x,y),BLUE,2)
            rect = (ix,iy,x, y)

    #Get avg frame:
    if vid.avg is None:
        vid.calc_avg()
    avg = vid.avg
    
    #Set up the windows
    
    global toshow, vis 
    gavg = cv2.cvtColor(avg.astype('uint8'), cv2.COLOR_RGB2GRAY)

    cedge = cv2.Canny(gavg, 90, 200, apertureSize=5)
    sedge = cv2.Sobel(gavg, ddepth = -1, dx=0, dy=1, ksize=-1)
    sedge[sedge <  70] = 0
    avg = cv2.cvtColor(avg.astype('uint8'), cv2.COLOR_RGB2BGR)

    viscedge = np.zeros(avg.shape, dtype='uint8')
    viscedge[:,:,0] = cedge
    viscedge[:,:,1] = cedge
    viscedge[:,:,2] = cedge

    vissedge = np.zeros(avg.shape, dtype='uint8')
    vissedge[:,:,0] = sedge
    vissedge[:,:,1] = sedge
    vissedge[:,:,2] = sedge

    toshow = cv2.addWeighted(avg, .8, viscedge, .2, 0,  dtype=cv2.CV_8UC3)
    toshow2 = cv2.addWeighted(avg, .8, vissedge, .2, 0,  dtype=cv2.CV_8UC3)      
    cv2.imshow('cannyim_full', toshow)
    cv2.imshow('sobelim_full', toshow2)
    vis = toshow.copy()                   
 

    cv2.setMouseCallback('cannyim_full', onmouse)
  

    while True:
        cv2.imshow('cannyim_full', vis)
        k = cv2.waitKey(100)
        if k is ord('a'):
            break;

    #now we have the rectangle region of interest:
    ix, iy, x, y = rect
    print rect
    minx = min(ix, x)
    maxx = max(ix, x)
    miny = min(iy, y)
    maxy = max(iy, y)
    ccrop = cedge[miny:maxy, minx:maxx]
    scrop = sedge[miny:maxy, minx:maxx]
    cv2.imshow('cannyim', ccrop)
    cv2.imshow('sobelim', scrop)
    # CANNY
    idx = [[x, y] for [y, x] in np.transpose(np.nonzero(cedge > 90)) ]
    idx2 = [[x,y] for  [x, y] in idx if x in range(minx, maxx) if y in range(miny, maxy)]
    #print range(min(ix, x), max(ix, x))
    #return np.array(idx2)
    vx, vy, cx, cy = cv2.fitLine(np.float32(idx2), cv2.DIST_L2, 0, .01, .01)
    vis = avg.copy()
    w, h = toshow.shape[:2]
    cv2.line(vis, (int(cx - vx*w), int(cy-vy*w)), (int(cx + vx*w), int(cy + vy*w)), (0, 255, 255), 1)
    
    idx = [[x, y] for [y, x] in np.transpose(np.nonzero(sedge > 90)) ]
    idx2 = [[x,y] for  [x, y] in idx if x in range(minx, maxx) if y in range(miny, maxy)]
    #print range(min(ix, x), max(ix, x))
    #return np.array(idx2)
    #vx, vy, cx, cy = cv2.fitLine(np.float32(idx2), cv2.DIST_L2, 0, .01, .01)
    #cv2.line(vis, (int(cx - vx*w), int(cy-vy*w)), (int(cx + vx*w), int(cy + vy*w)), (0, 0, 255), 1)       


    cv2.imshow('result', vis)
    cv2.destroyAllWindows()
    return vx, vy, cx, cy

def vis_horizon(vid):
    vx, vy, cx, cy = get_horizon(vid)

    for fr in range(vid.num_frames):
        im = vid.get_frame(fr)
        w, h = im.shape[:2]
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.line(im, (int(cx - vx*w), int(cy-vy*w)), (int(cx + vx*w), int(cy + vy*w)), (0, 0, 255), 1)
        cv2.imshow('a', im)
        k = cv2.waitKey(100)
        if k is ord('z'):
            break
    return vx, vy, cx, cy

def get_y_coord(x, vx, vy, cx, cy):
    """Given the output of fitline (vx, vy, cx, cy) and the point at which we want the height,
    returns the height"""
    y = (vy/vx)*x + cy - (vy/vx)*cx
    return y

def get_angle(vx, vy):
    """Ange of line in degrees"""
    return np.arctan(vy/vx)*180/(2 * np.pi)



def draw_rotated_rec(rotrec, img):
    ##this is the rotated rectange:
            ##  (center x, center y), (width, height), angle from column (90 is straight up, 0 is horizontal)
    wid, hei = rotrec[1]
    cx, cy = rotrec[0]
    ang = np.radians(rotrec[2])

    #first nonrotated rec about origin
    x1 = -wid/2
    x2 = wid/2
    y1 = hei/2
    y2 = -hei/2

    rotmatrix = np.array([[np.cos(ang), -np.sin(ang) ], [np.sin(ang), np.cos(ang)]])
    pt1 = rotmatrix.dot(np.array([x1, y1])) + np.array([cx, cy])
    pt2 = rotmatrix.dot(np.array([x2, y1]))+ np.array([cx, cy])
    pt3 = rotmatrix.dot(np.array([x2, y2]))+ np.array([cx, cy])
    pt4 = rotmatrix.dot(np.array([x1, y2]))+ np.array([cx, cy])

    lala = [(int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (int(pt3[0]), int(pt3[1])), (int(pt4[0]), int(pt4[1])), (int(pt1[0]), int(pt1[1]))]

    for i in range(4):
        cv2.line(img, lala[i], lala[i+1], (255, 255, 0), 2 )
    return img