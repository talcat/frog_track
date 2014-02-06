from FrogFrames import *
from hand_track import *
from sel_prim_roi import *
import cPickle as pik
import os
import cv2
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
from lookup_vids import *

def get_video(f_num):
	"""Returns Gulo Film clip video object"""
	base = "/home/talcat/Desktop/Bio_Interface/Frogs/frog_frame/GuloFilm"
	folder = base +"/" + f_num
	if f_num not in os.listdir(base):
		print "Not a valid Gulo Film Clip"
		return
	video = FrogFrames(folder, loop=False, wantgray=False, eq=False)
	return video

def do_handtrack(f_num, point_tracked, edit=False, doline=False):
	"""f_num is a string 'F01' -> 'F16' of the Gulo Film clips
	point_tracked is a string detailing what point is being tracked in this run
	the tracked points will be saved in the F folder as point_tracked_PTS.pik and 
	point_tracked_ROI.pik"""

	#base = "/home/talcat/Desktop/Bio_Interface/Frogs/frog_frame/GuloFilm"
	#folder = base +"/" + f_num
	
	if f_num not in lookup.keys():
		print "Not a valid Film Clip"
		return

	folder = lookup[f_num]	
	
	PTS_dest = folder + "/" + point_tracked + "_PTS.pik"
	ROI_dest = folder + "/" + point_tracked + "_ROI.pik"
	PTS_un_dest = folder + '/' + point_tracked + "_PTS_un.pik"
	line_dest = folder +'/' + point_tracked + "_line.pik"
	
	PTS = None
	ROI = None
	PTS_un = None
	line = None

	if edit:
		#load the previous existing _PTS.pik and _ROI.pik to edit
		try:
			with open(PTS_dest, 'r') as f:
				PTS = pik.load(f)
				print "PTS loaded"
			with open(ROI_dest, 'r') as f:
				ROI = pik.load(f)
				print "ROI loaded"
			with open(PTS_un_dest, 'r') as f:
				PTS_un = pik.load(f)
				print "PTS uncertainty loaded"
			
			if doline:
				with open(line_dest, 'r') as f:
					line = pik.load(f)
					print 'line loaded'
		except: pass

	if doline:
		todraw='line'
	else:
		todraw = 'box'
	video = FrogFrames(folder, loop=False, wantgray=False, eq=False)
	#First get initial ROI
	init_ROI = select_prim_ROI(video.get_frame(0))
	(minr, maxr), (minc, maxc) = init_ROI
	#Now, run the track by hand routine
	ROI, PTS, PTS_un, line = hand_track(video, (minr, maxr), (minc, maxc), todraw, ROI, PTS, PTS_un, line)


	with open(PTS_dest, 'w') as f:
		pik.dump(PTS, f)
	with open(ROI_dest, 'w') as f:
		pik.dump(ROI, f)
	with open(PTS_un_dest, 'w') as f:
		pik.dump(PTS_un, f)
	if doline:
		with open(line_dest, 'w') as f:
			pik.dump(line, f)

	return

def contact(f_num, name):
	"""f_num is a string 'F01' -> 'F16' of the Gulo Film clips
	Loads the video, and at each frame you record:
	w: in water (so launchin) - up until feet are free
	c: feet in contact with water
	a: in air (aerial phase)
	will save dict of frame-num (0 index) of w/c/a as contact.pik"""
	def handle_key(key, status, toggle, k):
		if key == ord('a'):
			status[k] = 'a'
			print 'yes, doing ' + name
			return
		elif key == ord('z'):
			print 'not doing anything'	
			return
		elif key == 65363: #right arrow	
			cv2.imshow("%s"%(f_num), cv2.cvtColor(video.get_frame(k - toggle%2), cv2.COLOR_RGB2BGR))
			toggle += 1
			key =cv2.waitKey(-1)
			handle_key(key, status, toggle, k)
			return

		elif key == 27: #escape
			print "Exiting"
			cv2.destroyAllWindows()	
			return
		else: return	

	if f_num not in lookup.keys():
		print "Not a valid Film Clip"
		return

	folder = lookup[f_num]
	
	video = FrogFrames(folder, loop=False, wantgray=False, eq=False)

	status = dict()
	out = False
	toggle = 1
	for k in range(video.num_frames):
		print k
		cv2.imshow("%s"%(f_num), cv2.cvtColor(video.get_frame(k), cv2.COLOR_RGB2BGR))
		key = cv2.waitKey(-1)
		handle_key(key, status, toggle, k)
	
		if key == 27: #escape
			print "Exiting"
			cv2.destroyAllWindows()	
			break	


	out = folder + "/" + name + ".pik"
	with open(out, 'w') as f:
		pik.dump(status, f)
	cv2.destroyAllWindows()	
	return

def draw_pts(f_num, point_tracked):
	base = "/home/talcat/Desktop/Bio_Interface/Frogs/frog_frame/GuloFilm"
	folder = base +"/" + f_num
	if f_num not in os.listdir(base):
		print "Not a valid Gulo Film Clip"
		return
	
	PTS_f = folder + "/" + "%s_PTS.pik"%(point_tracked)
	if "%s_PTS.pik"%(point_tracked) not in os.listdir(folder):
		print "Points never taken"
		return

	with open(PTS_f, 'r') as f:
		PTS = pik.load(f)	

	video = FrogFrames(folder, loop=False, gray=False, eq=False)

	name = folder + "/" + point_tracked + ".avi"
	print name
	#writer = cv2.VideoWriter(name, cv2.cv.CV_FOURCC(*'XVID'), 20, video.get_frame(0).shape[:2], True)
 
	for i in range(video.num_frames):
		img = video.get_frame(i)
		if i in PTS.keys():
			(ptr, ptc) = PTS[i]
			cv2.circle(img, (int(ptc), int(ptr)), 8, (0, 255, 0), 2)
		#writer.write(img)
		cv2.imshow('ack', img)
		k = cv2.waitKey(10)	
		if k == 27: #esc
			break

	cv2.destroyAllWindows()		
	#writer.release()

def draw_line(f_num, point_tracked):
	base = "/home/talcat/Desktop/Bio_Interface/Frogs/frog_frame/GuloFilm"
	folder = base +"/" + f_num
	if f_num not in os.listdir(base):
		print "Not a valid Gulo Film Clip"
		return
	
	line_f = folder + "/" + "%s_line.pik"%(point_tracked)
	if "%s_line.pik"%(point_tracked) not in os.listdir(folder):
		print "line never taken"
		return

	with open(line_f, 'r') as f:
		line = pik.load(f)	

	video = FrogFrames(folder, loop=False, wantgray=False, eq=False)

	name = folder + "/" + point_tracked + ".avi"
	print name
	writer = cv2.VideoWriter()
	writer.open(name, cv2.VideoWriter_fourcc(*'XVID'), 20, video.get_frame(0).shape)
 	#FFMpegWriter = manimation.writers['ffmpeg']
 	#writer = FFMpegWriter(fps=15, metadata = dict(title='Frog Body Angle', artist='Matplotlib', comment="This won't work"))
 	#fig = figure()
 	#ax = plt.axes([0,0,1,1], frameon=0)
 	#fig = ax.get_figure()

 	#with writer.saving(fig, name, video.num_frames):
	for fr in range(video.num_frames):
		img = video.get_frame(fr)
		print fr
		if fr in line.keys():
			(sx, ex), (sy, ey) = line[fr]
			cv2.line(img, (int(sx), int(sy)), (int(ex), int(ey)),(0, 255, 0), 2)
		writer.write(img)
		cv2.imshow('ack', img)
		k = cv2.waitKey(10)	
		#ax.imshow(img, interpolation="nearest")
		#ax.axis('off')
		#writer.grab_frame()
		if k == 27: #esc
			break

	cv2.destroyAllWindows()		
	writer.release()

def plot_angles(f_num,point_tracked):
	base = "/home/talcat/Desktop/Bio_Interface/Frogs/frog_frame/GuloFilm"
	folder = base +"/" + f_num
	if f_num not in os.listdir(base):
		print "Not a valid Gulo Film Clip"
		return
	
	line_f = folder + "/" + "%s_line.pik"%(point_tracked)
	if "%s_line.pik"%(point_tracked) not in os.listdir(folder):
		print "line never taken"
		return

	with open(line_f, 'r') as f:
		line = pik.load(f)	

	#get arrays of (startx, endx), (starty, endy)
	vals = line.values()
	#get arrays of (delta x, delta y)
	vals = map(lambda z: z[:, 1] - z[:, 0], vals)
	#calculate the angle
	angle = map(lambda z: np.arctan(z[1]/z[0]), vals)
	#in degrees
	angle = map(lambda z: 360/2/np.pi * z, angle)

	keys = line.keys()

	return angle, keys


