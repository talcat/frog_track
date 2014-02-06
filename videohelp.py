#videohelp.py

#mostly from http://www.raben.com/book/export/html/6
#defines some nice things to use VideoWriter in cv2

import cv2
import os
import time
import numpy as np


import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

outf = 'test.avi'
rate = 1

cmdstring = ('local/bin/ffmpeg',
             '-r', '%d' % rate,
             '-f','image2pipe',
             '-vcodec', 'png',
             '-i', 'pipe:', outf
             )
p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)

plt.figure()
frames = 10
for i in range(frames):
    plt.imshow(np.random.randn(100,100))
    plt.savefig(p.stdin, format='png')



CV_CAP_PROP_FRAME_WIDTH = 3
CV_CAP_PROP_FRAME_HEIGHT = 4

def CV_FOURCC(c1, c2, c3, c4) :
    return (c1 & 255) + ((c2 & 255) << 8) + ((c3 & 255) << 16) + ((c4 & 255) << 24)
    
def createVideoFile(path, width, height, fps):
    fourcc = CV_FOURCC('P', 'I', 'M', '1')  #MPEG1
    filename = time.strftime("%Y-%m-%dT%Hh%Mm%Ss", time.gmtime(tsec))
    pathname = os.path.join(path,filename) + ".avi"
    print "Create video file "+pathname
    writer = VideoWriter(pathname, fourcc, fps,(width,height))
    return writer;
    
    
def array2cv(a):
  dtype2depth = {
        'uint8':   cv2.cv.IPL_DEPTH_8U,
        'int8':    cv2.cv.IPL_DEPTH_8S,
        'uint16':  cv2.cv.IPL_DEPTH_16U,
        'int16':   cv2.cv.IPL_DEPTH_16S,
        'int32':   cv2.cv.IPL_DEPTH_32S,
        'float32': cv2.cv.IPL_DEPTH_32F,
        'float64': cv2.cv.IPL_DEPTH_64F,
    }
  try:
    nChannels = a.shape[2]
  except:
    nChannels = 1
  cv_im = cv2.cv.CreateImageHeader((a.shape[1],a.shape[0]),
          dtype2depth[str(a.dtype)],
          nChannels)
  cv2.cv.SetData(cv_im, a.tostring(),
             a.dtype.itemsize*nChannels*a.shape[1])
  return cv_im
