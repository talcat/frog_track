#Python video testing

import cv2
import numpy as np

writer = cv2.VideoWriter()

writer.open('square.avi', cv2.VideoWriter_fourcc(*'XVID'), 1, (300, 300), False)

white = 255*np.ones((300, 300)).astype('uint8')
black = np.zeros((300, 300)).astype('uint8')

for k in range(20):
	print k%2
	if k%2 == 0:
		writer.write(white)
	else:
		writer.write(black)

writer.release()

writer.open('non-square.avi', cv2.VideoWriter_fourcc(*'XVID'), 1, (400, 300), False)

white = 255*np.ones((400, 300)).astype('uint8')
black = np.zeros((400, 300)).astype('uint8')

for k in range(20):
	print k%2
	if k%2 == 0:
		writer.write(white)
	else:
		writer.write(black)

writer.release()