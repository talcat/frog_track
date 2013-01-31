#matchpts.py  

from matplotlib import use
use('Qt4Agg')
from matplotlib.widgets import  RectangleSelector
from pylab import *
import cv2


def better_mask(mask0, im0):
    """Uses OpenCV's blob detector to try and define a better mask such that
    only points on the frog are found in the feature detector surf/sift/orb"""
    
    #convert im0 to 8bit 
    im0 = 255*cv2.cvtColor(im0, cv2.COLOR_RGB2GRAY)
    mask0 = 255*cv2.cvtColor(mask0.astype('float32'), cv2.COLOR_RGB2GRAY)
    #mask it
    im0 = im0.astype('uint8')
    mask0 = mask0.astype('uint8')
    
    #Create a blob detector
    blob = cv2.FeatureDetector_create('SimpleBlob')
    kp = blob.detect(im0, mask0)
    
    return kp

def init_feature(name):
    """Copied from find_obj in OpenCV python2 examples"""
    chunks = name.split('-')
    if chunks[0] == 'sift':
        detector = cv2.SIFT(nfeatures=5)
        norm = cv2.NORM_L2
    elif chunks[0] == 'surf':
        detector = cv2.SURF(400)
        norm = cv2.NORM_L2
    elif chunks[0] == 'orb':
        detector = cv2.ORB(400)
        norm = cv2.NORM_HAMMING
    else:
        return None, None
    if 'flann' in chunks:
        if norm == cv2.NORM_L2:
            flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else:
            flann_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) #2
        matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    else:
        matcher = cv2.BFMatcher(norm)
    return detector, matcher
    

def getpts(mask0, im0, detector='surf'):
    """ASSUMING SMALL MOVEMENTS, the roi between 2 adjacent frames should be the
    same.  Thus, between two pairs of images, the same mask is used.
    This calculates SURF points for two sequential images, and matches them.  
    An average velocity is then calculated for the movement in the ROI.
    The velocity is returned so the ROI can be updated"""
    
    im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2GRAY)
    mask0 = cv2.cvtColor(mask0.astype('float32'), cv2.COLOR_RGB2GRAY)
    
    detector, _ = init_feature(detector)
    
    #uint8:
    im0 = im0*255
    mask0 = mask0*255
    
    #normalize frog image....
    im0 = cv2.equalizeHist(im0.astype('uint8'))
    
    kp, descriptors = detector.detectAndCompute(im0.astype('uint8'), mask0.astype('uint8'))

    
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
    
 
def explore_match(win, img1, img2, kp_pairs, mask=None, status = None, H = None):
    """Copied from OpenCV find_obj example.
    win = name of window
    kp_pairs = pairs found from matchpts
    edited to allow for the mask
    No onmouse event
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
 
    #Color conversion >_<)
    img1 = 255*cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = 255*cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) 
    if mask != None:
        mask = cv2.cvtColor(mask.astype('float32'), cv2.COLOR_RGB2GRAY)
        img1 = img1*mask
        img2 = img2*mask
    
 
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

    return vis
 
         
def matchpts(kp0, des0, kp1, des1, matcher='surf'):
    """Uses OpenCV Brute-force descriptor matcher (BFFMatcher) to match SURF 
    points found in im0 and im1.
    A lot of this is copies/modified from opencv's find_obj example script"""
    
    _, matcher = init_feature(matcher)
    
    raw_matches = matcher.knnMatch(des0, trainDescriptors = des1, k = 2) 

    
    p0, p1, kp_pairs = filter_matches(kp0, kp1, raw_matches)
    
    if len(p0) >= 4:
        H, status = cv2.findHomography(p0, p1, cv2.RANSAC, 5.0)
        print '%d / %d  inliers/matched' % (np.sum(status), len(status))
    else:
        H, status = None, None
        print '%d matches found, not enough for homography estimation' % len(p0)
    
    return kp_pairs, status, H
   
def get_pts(kp_pairs):
    """Given the matched pairs of KeyPoints, will return 2 numpy arrays (1 for
    each image) of the point locations of matched points"""
    p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) 
    
    return p1, p2
    
def euc_dis(p0, p1):
    """Given a pair of points: [p0r, p0c] and [p1r, p1c] will return the euclidian
    distance between them = i.e. the distance that point moved between two images."""
    
    p0r, p0c = p0
    p1r, p1c = p1
     
    dis = np.sqrt( (p1r - p0r)**2 + (p1c - p0c)**2)
     
    return dis
       
def direc(p0, p1):
    """Given a pair of points [p0r, p0c] and [p1r, p1c] will return the the angle
    AKA direction in radians that the point moved"""
    
    p0r, p0c = p0
    p1r, p1c = p1
    
    delr = p1r - p0r
    delc = p1c - p0c
    #np.arctan2(y, x)
    return np.arctan2(delr, delc)       
       

def remove_outliers(kp_pairs):
    """For kp_pairs, removes the point pairs that are outliers depending on the 
    vector magnitude and vector angle according to 'any poin tthat is more than 
    1.5 IQRs (interquantile range) below the first anf above the 3rd quantile"""
    p0, p1 = get_pts(kp_pairs)
    
    corr = [ (p0[1], p1[i]) for i in range(len(p0))]
    dis = map(lambda (x, y): euc_dis(x, y), corr) 
    ang = map(lambda (x, y): direc(x, y), corr) 
    
    #for distance
    dis2 = np.sort(dis)
    med = np.median(dis2)
    low50 = dis2[dis2<med]
    high50 = dis2[dis2>med]
    dlowmed = np.median(low50)
    dhighmed = np.median(high50)
    
    dIQR = 1.5*(dhighmed - dlowmed)  # 3rd quant - 1st quant
    #dis = np.array(dis)
    #dis = dis[dis <= (highmed + IQR)]
    #dis = dis[dis >= (lowmed - IQR)]
    
    #for direction
    ang2 = np.sort(ang)
    med = np.median(ang2)
    low50 = ang2[ang2<med]
    high50 = ang2[ang2>med]
    alowmed = np.median(low50)
    ahighmed = np.median(high50)
    
    aIQR = 1.5*(ahighmed - alowmed)  # 3rd quant - 1st quant
    #ang = np.array(ang)
    #ang = ang[ang <= (highmed + IQR)]
    #ang = ang[ang >= (lowmed - IQR)]
    
    p0 = np.array(p0)
    p1 = np.array(p1)
    
    p0 = p0[(ang <= (ahighmed + aIQR)) & (ang >= (alowmed - aIQR)) & 
            (dis <= (dhighmed + dIQR)) & (dis >= (dlowmed - dIQR))] 
    p1 = p1[(ang <= (ahighmed + aIQR)) & (ang >= (alowmed - aIQR)) & 
            (dis <= (dhighmed + dIQR)) & (dis >= (dlowmed - dIQR))] 
    
    return p0, p1


def avg_vec(out0, out1):
    """Given a list of points [[p0r, p0c]], [[p1r, p1c]] will calculate the 
    avg vector"""
    
    length = len(out0)
    delta = out1 - out0
    r = mean(delta[:, 0])
    c = mean(delta[:, 1])
    
    return [r, c]
    


if __name__ == "__main__":
    mask0 = imread('mask-0.png')
    im0 = imread('im-0.png')
    im1 = imread('im-1.png')
    
    det = 'sift'
    
    kp0, descriptors0 = getpts(mask0, im0, det)
    kp1, descriptors1 = getpts(mask0, im1, det)
    
    kp_pairs, status, H = matchpts(kp0, descriptors0, kp1, descriptors1, det)
    
    if len(kp_pairs) > 0:
        explore_match("test", im0, im1, kp_pairs, mask0)
    else:
        print 'No matches found :/'

        
    p0, p1 = get_pts(kp_pairs)
    
    #corr = [ (p0[1], p1[i]) for i in range(len(p0))]
    #dis = map(lambda (x, y): euc_dis(x, y), corr) 
    #ang = map(lambda (x, y): direc(x, y), corr) 
    
    out0, out1 = remove_outliers(kp_pairs)
    
    avg = avg_vec(out0, out1)
         
