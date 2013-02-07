#testrun.py

#test script to actually run the moving roi on an entire group of frogs

from updateroi import *
from recsel import *
from matchpts import *
from backgroundsub import FrogFrames
from copy import copy
#from videohelp import *
from cv import CV_FOURCC

DELTA_T = .002
PATH = '/home/talcat/Desktop/Bio Interface/Frogs/frog_frame/Shot2/'
DET = 'surf'


#Get frog videos
video = FrogFrames('/home/talcat/Desktop/Bio Interface/Frogs/frog_frame/Shot2/',
                    loop=True, gray=False)

#video.list = video.list[0:5]
#intialize mask
prev, prev_idx = video.get_next_im()

#Set up videowrite obj
hei, wid, _ = prev.shape
writer1 = cv2.VideoWriter('%strack.mp4' %PATH, CV_FOURCC('D', 'I', 'V', 'X'), fps=25, 
                        frameSize=(wid, hei))
writer2 =cv2.VideoWriter('%sfeature.mp4' %PATH, CV_FOURCC('D', 'I', 'V', 'X'), fps=25, 
                        frameSize=(wid*2, hei))

mask = select_area(prev)
[r0, r1, c0, c1] = returncorners(mask)
height = r1 - r0
width = c1 - c0
avg = [0, 0]
centers =[]
cond = True
deltax = []
deltay = []
time = []

cv2.namedWindow("input")
while(cond):
    
    #show previous image:
    prevtoshow = copy(prev)
    cv2.imshow("input", prevtoshow)

    #draw roi
    #get corners of mask:
    [r0, r1, c0, c1] = returncorners(mask)
    centers.append([c0 + width/2 ,r0 + height/2 ])
    
    #draw them on the copy (do not change and pass original prev)
    cv2.rectangle(prevtoshow, (c0, r0), (c1, r1), (255, 255, 255))
    cv2.polylines(prevtoshow, [np.array(centers, dtype=np.int0)], False, (0, 255, 0))
    cv2.imshow("input", prevtoshow)    

    if writer1.isOpened():
        print 'frame written'
        wrtier1.write(prevtoshow)

    key = cv2.waitKey(30)
    #cv2.imwrite("./pngs/image-"+str(a).zfill(5)+".png", fgmask)
    
    if key ==32:
        cond=False
    
    
    #get next image
    
    next, next_idx = video.get_next_im()
    
    if next == None:
        cond=False
        break
    
   #get the change in posititions:
    
    if prev_idx == 0:
        kp_prev, descriptors_prev = getpts(mask, prev, DET)
    
    kp_next, descriptors_next = getpts(mask, next, DET)
    
    kp_pairs, status, H = matchpts(kp_prev, descriptors_prev, 
                                   kp_next, descriptors_next, DET)

    if len(kp_pairs) > 3:
       print 'Explore!'
       #explore_match("test", prev, next, kp_pairs, mask)
    else:
        print 'No matches found - finding new prev features...'
        kp_prev, descriptors_prev = getpts(mask, prev, DET)
        kp_pairs, status, H = matchpts(kp_prev, descriptors_prev, 
                                         kp_next, descriptors_next, DET)
            
        if len(kp_pairs)!=0:
            print 'Explore!'
            #explore_match("test", prev, next, kp_pairs, mask)
        else:    
            print 'Nope, out of luck'
            break              
    
        
        
    p_prev, p_next = get_xy(kp_pairs)
    
    #if there are <= 5 points to track, do not remove outliers 
    if len(kp_pairs) > 5:
        out_prev, out_next = remove_outliers(kp_pairs)
    else: # no outliers
        out_prev = [kpp[0] for kpp in kp_pairs]
        out_next = [kpp[1] for kpp in kp_pairs]
    
    out_pairs = [(out_prev[i], out_next[i]) for i in range(len(out_prev))]
    vis = explore_match("test", prev, next, out_pairs, mask)
    if writer2.isOpened():
        print 'frame written'
        writer2.write(vis)
    
    avg = avg_vec(out_prev, out_next)
    #avg = [5, 5]
    deltax.append(avg[0])    
    deltay.append(avg[1])
    time.append(DELTA_T*prev_idx)
    
    
    #update mask for next round
    mask = updateroi(mask, avg)
    
    #next --> prev
    prev = next     
    prev_idx = next_idx
    kp_prev = out_next
    descriptors_prev = [descriptors_next[i] for i in range(len(descriptors_next)) 
                        if kp_next[i].pt in map(lambda x: x.pt, out_next)]
    descriptors_prev= np.array(map(list, descriptors_prev))
    
    #print(video.index)
#cv2.destroyAllWindows()
writer1.release()
writer2.release()
