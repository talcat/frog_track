import matplotlib.pyplot as plt
import numpy as np
import cPickle as pik
from watershed import *
import scipy.interpolate as inter

def load_dic(name):
	with open(name, 'r') as f:
		dic = pik.load(f)

	return dic


def plot_watershed(ellipse, angle, horizon, svl=None, fps=None):
	WID, HI = 1280, 720	

	ell = load_dic(ellipse)
	ang = load_dic(angle)
	line = load_dic(horizon)

	#get angle and heights of horizon
	hor_hei = map(lambda x: HI - get_y_coord(x[0][1], line['vx'], line['vy'], line['cx'], line['cy']), ell.values())
	hor_hei = [x[0] for x in hor_hei] #unwrap from array this is in for some reason
	hor_ang = get_angle(line['vx'], line['vy'])
	hor_ang = hor_ang[0] #unwrap from array

	#unwrap and fix values from watershed based on horizon
	xvalues = ell.keys()
	heights = map(lambda x: HI - x[0][1], ell.values())

	#Distance in pixels
	dis = map(lambda x: x[0][0], ell.values() )

	#THis is the PIXEL height above water surface
	del_hei = [heights[i] - hor_hei[i] for i in range(len(hor_hei))]
	#Angle compared to horizon in degrees
	del_ang = ang.values() - hor_ang


	if svl is not None:
		svl = load_dic(svl)
		#average the svl
		svl = np.mean(svl.values())
	else:
		svl = 1
	if fps is not None:
		xvalues = [float(x)/fps* 1000 for x in xvalues]




	#OK PLOTTING SHIT LETS SEE HOW THIS GOES
	fig, (dis_ax, hei_ax, ang_ax) = plt.subplots(nrows=3, sharex=True)
	xx = np.arange(min(xvalues), max(xvalues), .2)
	
	#dis_ax=  plt.subplot(311, sharex=ang_ax)
	dis_line = dis_ax.plot(xvalues, [d/svl for d in dis], 'o', color='gray', alpha=.5, label='Horizontal Distance')
	dis_fit = inter.UnivariateSpline(xvalues, [d/svl for d in dis])
	dis_ax.plot(xx, dis_fit(xx), 'r-')


	#STD DEV TEST
	#dis_ax.fill_between(xvalues, np.array(dis) - 30, np.array(dis) + 30, facecolor='gray', alpha=0.5, label='Distance Range' )

	adjust_spines(dis_ax, ['left', 'right'])	
	#plt.setp(dis_ax.get_xticklabels(), visible=False)
	if svl != 1:
		dis_ax.set_ylabel('Distance (SVL)')
	else:
		dis_ax.set_ylabel('Distance (px)')

	#hei_ax = plt.subplotclear (312, sharex=ang_ax)
	hei_line = hei_ax.plot(xvalues, [d/svl for d in del_hei], 'o', color='gray', alpha=.5, label='Height')
	#hei_ax.fill_between(xvalues, np.array(del_hei) - 10, np.array(del_hei) + 10, facecolor='gray', alpha=0.5, label='Height Range' )
	hei_fit = inter.UnivariateSpline(xvalues, [d for d in del_hei])
	hei_ax.plot(xx, [d/svl for d in hei_fit(xx)], 'b-')
	
	adjust_spines(hei_ax, ['left', 'right'])	
	#plt.setp(hei_ax.get_xticklabels(), visible=False)
	if svl != 1:
		hei_ax.set_ylabel('Height (SVL)')
	else:
		hei_ax.set_ylabel('Height (px)')

	#ang_ax = plt.subplot(313)
	ang_line = ang_ax.plot(xvalues, del_ang, 'o', color='gray', alpha=.5, label='Angle (Degrees)')
	ang_fit = inter.UnivariateSpline(xvalues, del_ang, s=2*len(del_ang))
	ang_ax.plot(xx, ang_fit(xx), 'g-')

	adjust_spines(ang_ax, ['left', 'bottom', 'right'])
	#ang_ax.fill_between(xvalues, np.array(del_ang) - 5, np.array(del_ang) + 5, facecolor='gray', alpha=0.5, label='Angle Range' )

	ang_ax.set_ylabel('Angle (Degrees)')

	if fps is not None:
		ang_ax.set_xlabel('Time (ms)')
	else:
		ang_ax.set_xlabel('Frame Number')

	#adjust horizontal spacing
	plt.subplots_adjust(hspace=.1)

	#Hide Spines
	#ang_ax.spines['right'].set_visible(False)
	#ang_ax.spines['top'].set_visible(False)
	#hei_ax.spines['bottom'].set_visible(False)
	#hei_ax.spines['top'].set_visible(False)
	#hei_ax.spines['right'].set_visible(False)
	#dis_ax.spines['bottom'].set_visible(False)
	#dis_ax.spines['top'].set_visible(False)
	#dis_ax.spines['right'].set_visible(False)

	#fine ill fucking do it manually:


	plt.show()

	return fig, (dis_ax, hei_ax, ang_ax)

def adjust_spines(ax,spines):
	#"""From matplotlin examples"""
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward',10)) # outward by 10 points
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none') # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_ticks([])


    if 'bottom' in spines:

        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks_position('none')
        ax.xaxis.set_ticks([])	


def gait_diagram(list_of_dics, fps=None, WID=10):
	ld = []
	name = []
	for dic in list_of_dics:
		tmp = load_dic(dic)
		ld.append(tmp)
		ind = dic.rfind('/')
		dic = dic[ind + 1:]
		name.append(dic.rstrip('.pik'))

	ax = plt.subplot(111)

	txrange = len(name)	
	idx = 1

	for dic in ld:
		l = dic.keys()
		l.sort()

		ends = [k for k in range(len(l) - 1) if l[k] != l[k+1] - 1]
		#add the beginning of each run after the ends
		ends = [[k, k+1] for k in ends]
		ends = list(np.array(ends).flatten())
		ends = map(lambda x: l[x], ends)

		ends = [l[0]] + ends + [l[-1]]
		#this is a list of first, last, fist, last, first, last
		runs = [(ends[i], ends[i + 1] - ends[i]) for i in range(len(ends)) if i%2 == 0]
		if fps is not None:
			runs = [(float(x)/fps* 1000, float(y)/fps*1000) for (x, y) in runs]
		ax.broken_barh(runs, (WID*idx + .5, WID-0.5), facecolors='black')
		idx += 1;

	adjust_spines(ax, ['left', 'right', 'bottom'])
	ax.set_yticks([(x + .5)*WID for x in np.arange(1, txrange + 1)])
	ax.set_yticklabels(name)
	
	return ax


def plot_all_the_things(ellipse, angle, horizon, list_of_gaits, svl=None, fps=None):	
	WID, HI = 1280, 720	
	WID2 = 2 #for gait diagram


	ell = load_dic(ellipse)
	ang = load_dic(angle)
	line = load_dic(horizon)

	#get angle and heights of horizon
	hor_hei = map(lambda x: HI - get_y_coord(x[0][1], line['vx'], line['vy'], line['cx'], line['cy']), ell.values())
	hor_hei = [x[0] for x in hor_hei] #unwrap from array this is in for some reason
	hor_ang = get_angle(line['vx'], line['vy'])
	hor_ang = hor_ang[0] #unwrap from array

	#unwrap and fix values from watershed based on horizon
	xvalues = ell.keys()
	heights = map(lambda x: HI - x[0][1], ell.values())

	#Distance in pixels
	dis = map(lambda x: x[0][0], ell.values() )

	#THis is the PIXEL height above water surface
	del_hei = [heights[i] - hor_hei[i] for i in range(len(hor_hei))]
	#Angle compared to horizon in degrees
	del_ang = ang.values() - hor_ang


	if svl is not None:
		svl = load_dic(svl)
		#average the svl
		svl = np.mean(svl.values())
	else:
		svl = 1
	if fps is not None:
		xvalues = [float(x)/fps* 1000 for x in xvalues]




	#OK PLOTTING SHIT LETS SEE HOW THIS GOES
	fig, (gait_ax, dis_ax, hei_ax, ang_ax) = plt.subplots(nrows=4, sharex=True)
	xx = np.arange(min(xvalues), max(xvalues), .2)
	
	#GAIT STUFF
	ld = []
	name = []
	for dic in list_of_gaits:
		tmp = load_dic(dic)
		ld.append(tmp)
		ind = dic.rfind('/')
		dic = dic[ind + 1:]
		name.append(dic.rstrip('.pik'))

	txrange = len(name)	
	idx = 1

	for dic in ld:
		l = dic.keys()
		l.sort()

		ends = [k for k in range(len(l) - 1) if l[k] != l[k+1] - 1]
		ends = [[k, k+1] for k in ends]
		ends = list(np.array(ends).flatten())
		ends = map(lambda x: l[x], ends)

		ends = [l[0]] + ends + [l[-1]]
		#this is a list of first, last, fist, last, first, last
		runs = [(ends[i], ends[i + 1] - ends[i]) for i in range(len(ends)) if i%2 == 0]
		if fps is not None:
			runs = [(float(x)/fps* 1000, float(y)/fps*1000) for (x, y) in runs]
		gait_ax.broken_barh(runs, (WID2*idx + WID2*.05, WID2-WID2*0.05), facecolors='black')
		idx += 1;

	adjust_spines(gait_ax, ['left', 'right'])
	gait_ax.set_yticks([(x + .5)*WID2 for x in np.arange(1, txrange + 1)])
	gait_ax.set_yticklabels(name)


	#dis_ax=  plt.subplot(311, sharex=ang_ax)
	dis_line = dis_ax.plot(xvalues, [d/svl for d in dis], 'o', color='gray', alpha=.5, label='Horizontal Distance')
	dis_fit = inter.UnivariateSpline(xvalues, [d for d in dis])
	dis_ax.plot(xx, [d/svl for d in dis_fit(xx)], 'r-')


	#STD DEV TEST
	#dis_ax.fill_between(xvalues, np.array(dis) - 30, np.array(dis) + 30, facecolor='gray', alpha=0.5, label='Distance Range' )

	adjust_spines(dis_ax, ['left', 'right'])	
	#plt.setp(dis_ax.get_xticklabels(), visible=False)
	if svl != 1:
		dis_ax.set_ylabel('Distance (SVL)')
	else:
		dis_ax.set_ylabel('Distance (px)')

	#hei_ax = plt.subplotclear (312, sharex=ang_ax)
	hei_line = hei_ax.plot(xvalues, [d/svl for d in del_hei], 'o', color='gray', alpha=.5, label='Height')
	#hei_ax.fill_between(xvalues, np.array(del_hei) - 10, np.array(del_hei) + 10, facecolor='gray', alpha=0.5, label='Height Range' )
	hei_fit = inter.UnivariateSpline(xvalues, [d for d in del_hei])
	hei_ax.plot(xx, [d/svl for d in hei_fit(xx)], 'b-')
	
	adjust_spines(hei_ax, ['left', 'right'])	
	#plt.setp(hei_ax.get_xticklabels(), visible=False)
	if svl != 1:
		hei_ax.set_ylabel('Height (SVL)')
	else:
		hei_ax.set_ylabel('Height (px)')

	#ang_ax = plt.subplot(313)
	ang_line = ang_ax.plot(xvalues, del_ang, 'o', color='gray', alpha=.5, label='Angle (Degrees)')
	ang_fit = inter.UnivariateSpline(xvalues, del_ang, s=2*len(del_ang))
	ang_ax.plot(xx, ang_fit(xx), 'g-')

	adjust_spines(ang_ax, ['left', 'bottom', 'right'])
	#ang_ax.fill_between(xvalues, np.array(del_ang) - 5, np.array(del_ang) + 5, facecolor='gray', alpha=0.5, label='Angle Range' )

	ang_ax.set_ylabel('Angle (Degrees)')

	if fps is not None:
		ang_ax.set_xlabel('Time (ms)')
	else:
		ang_ax.set_xlabel('Frame Number')

	#adjust horizontal spacing
	plt.subplots_adjust(hspace=.3)

	return fig, (gait_ax, dis_ax, hei_ax, ang_ax)

def fix_textsize_axis(axes, xory, size):
	"""axes is an axes/subplot, xory is 'x' or 'y' for x or y axis, size is an int"""
	if xory is 'x':
		for text in axes.get_xticklabels():
			text.set_fontsize(size)
		plt.draw()
		return

	elif xory is 'y':
		for text in axes.get_yticklabels():
			text.set_fontsize(size)
		plt.draw()
		return
	else:
		print " 'x' or 'y' axes please"
		return

def fix_textsize_label(axes, xory, size):
	if xory is 'x':
		axes.xaxis.label.set_fontsize(size)
	elif xory is 'y':
		axes.yaxis.label.set_fontsize(size)
	else:
		print 'Error woo'
		return
	plt.draw()
	return

def set_tick_labels(axes, xory, nums):
	"""nums is a list of ticks that already exist, that you want to be the only ones on the graph"""
	if xory is 'x':
		axes.set_xticks(nums)
	elif xory is 'y':
		axes.set_yticks(nums)
	else:
		print 'error woo'
	plt.draw()
	return

def align_ylabels(axes_list, value=-.2, rot=0):
	for axes in axes_list:
		axes.yaxis.set_label_coords(value, .5)
		axes.set_ylabel(axes.get_ylabel(), rotation=rot)
	plt.draw()
	return


