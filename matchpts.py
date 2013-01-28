#matchpts.py  

from matplotlib import use
use('Qt4Agg')
from matplotlib.widgets import  RectangleSelector
from pylab import *
import cv2

def getpts(mask0, im0):
    """ASSUMING SMALL MOVEMENTS, the roi between 2 adjacent frames should be the
    same.  Thus, between two pairs of images, the same mask is used.
    This calculates SURF points for two sequential images, and matches them.  
    An average velocity is then calculated for the movement in the ROI.
    The velocity is returned so the ROI can be updated"""
    
    im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2GRAY)
    mask0 = cv2.cvtColor(mask0, cv2.COLOR_RGB2GRAY)
    
    thesurf = cv2.SIFT()
    
    #uint8:
    im0 = im0*255
    mask0 = mask0*255
    
    kp, descriptors = thesurf.detectAndCompute(im0.astype('uint8'), mask0.astype('uint8'))

    
    return kp, descriptors
   
def filter_matches(kp1, kp2, matches, ratio = 0.75):
    """Copied from OpenCV find_obj python example.   """
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs
    
 
def explore_match(win, img1, img2, kp_pairs, status = None, H = None):
    """Copied from OpenCV find_obj example.
    win = name of window
    kp_pairs = pairs found from matchpts"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
 
    #Color conversion >_<)
    img1 = 255*cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = 255*cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) 
    
 
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1.astype('uint8')
    vis[:h2, w1:w1+w2] = img2.astype('uint8')
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
    p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
            cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)
    vis0 = vis.copy()
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)

    cv2.imshow(win, vis)
    def onmouse(event, x, y, flags, param):
        cur_vis = vis
        if flags & cv2.EVENT_FLAG_LBUTTON:
            cur_vis = vis0.copy()
            r = 8
            m = (anorm(p1 - (x, y)) < r) | (anorm(p2 - (x, y)) < r)
            idxs = np.where(m)[0]
            kp1s, kp2s = [], []
            for i in idxs:
                 (x1, y1), (x2, y2) = p1[i], p2[i]
                 col = (red, green)[status[i]]
                 cv2.line(cur_vis, (x1, y1), (x2, y2), col)
                 kp1, kp2 = kp_pairs[i]
                 kp1s.append(kp1)
                 kp2s.append(kp2)
            cur_vis = cv2.drawKeypoints(cur_vis, kp1s, flags=4, color=kp_color)
            cur_vis[:,w1:] = cv2.drawKeypoints(cur_vis[:,w1:], kp2s, flags=4, color=kp_color)

        cv2.imshow(win, cur_vis)
    cv2.setMouseCallback(win, onmouse)
    return vis
 
 
 
       
def matchpts(kp0, des0, kp1, des1):
    """Uses OpenCV Brute-force descriptor matcher (BFFMatcher) to match SURF 
    points found in im0 and im1.
    A lot of this is copies/modified from opencv's find_obj example script"""
    
    norm = cv2.NORM_L2
    matcher = cv2.BFMatcher(norm)
    
    raw_matches = matcher.knnMatch(des0, trainDescriptors = des1, k = 2) 

    
    p0, p1, kp_pairs = filter_matches(kp0, kp1, raw_matches)
    
    if len(p0) >= 4:
        H, status = cv2.findHomography(p0, p1, cv2.RANSAC, 5.0)
        print '%d / %d  inliers/matched' % (np.sum(status), len(status))
    else:
        H, status = None, None
        print '%d matches found, not enough for homography estimation' % len(p0)
    
    return kp_pairs, status, H
    
    
if __name__ == "__main__":
    mask0 = imread('mask-0.png')
    im0 = imread('im-0.png')
    im1 = imread('im-1.png')
    
    kp0, descriptors0 = getpts(mask0, im0)
    kp1, descriptors1 = getpts(mask0, im1)
    
    kp_pairs, status, H = matchpts(kp0, descriptors0, kp1, descriptors1)
    
    explore_match("test", im0, im1, kp_pairs, status, H)
    
