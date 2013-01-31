#testrun.py

#test script to actually run the moving roi on an entire group of frogs

from updateroi import *
from recsel import *
from matchpts import *
from backgroundsub import FrogFrames

#Get frog videos
video = FrogFrames('/home/talcat/Desktop/Bio Interface/Frogs/frog_frame/Shot2/', 
                    loop=False, gray=False)
cv2.namedWindow("input")

#intialize mask
prev = video.get_next_im()
mask = select_area(prev)
avg = [0, 0]

cond = True
while(cond):
    
    #show previous image:
    
    cv2.imshow("input", prev)
    #draw roi
    #get corners of mask:
    [r0, r1, c0, c1] = returncorners(mask)
    
    #draw them
    cv2.rectangle(prev, (c0, r0), (c1, r1), (255, 255, 255))
    cv2.imshow("input", prev)    

    key = cv2.waitKey(30)
    #cv2.imwrite("./pngs/image-"+str(a).zfill(5)+".png", fgmask)
    
    if key ==32:
        cond=False
    
    
    #get next image
    idx = video.index
    next = video.get_next_im()
    
    if next == None:
        cond=False
        break
    
    #get the change in posititions:
   # det = 'sift'
    
   # kp0, descriptors0 = getpts(mask, prev, det)
   # kp1, descriptors1 = getpts(mask, next, det)
    
   # kp_pairs, status, H = matchpts(kp0, descriptors0, kp1, descriptors1, det)
    
   # if len(kp_pairs) > 0:
   #     explore_match("test", prev, next, kp_pairs, mask)
   # else:
   #     print 'No matches found :/'
   #     break
        
    #p0, p1 = get_pts(kp_pairs)
     
    #out0, out1 = remove_outliers(kp_pairs)
    
    #avg = avg_vec(out0, out1)
    avg = [5, 5]
         
    #update mask for next round
    mask = updateroi(mask, avg)
    
    #next --> prev
    prev = next     
    
    #print(video.index)
cv2.destroyAllWindows()
