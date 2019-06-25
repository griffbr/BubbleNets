# Script for evaluating J and F of DAVIS videos.
# Brent Griffin 180712

import IPython; import os; import glob; from PIL import Image; import numpy as np
import sys
sys.path.insert(0, './davis_python/lib/davis/measures')
from f_boundary import *; from jaccard import *
from joblib import Parallel, delayed
from multiprocessing import cpu_count

def J_and_F(im_file, ant_file):
	im = np.atleast_3d(Image.open(im_file))[...,0]
	ant = np.atleast_3d(Image.open(ant_file))[...,0]
	return db_eval_iou(im,ant), db_eval_boundary(im,ant)

def segmentation_score(result_dir, annotation_dir):
	# Get list of evaluations present or have preselected list for a specific video.
	video_name = result_dir.split('/')[-1]

	# For video, get J and F score for each frame.
	#seg_files = sorted(glob.glob(os.path.join(result_dir,'*.png')))
	ant_files = sorted(glob.glob(os.path.join(annotation_dir,'*')))
	n_frames = len(ant_files)

	seg_files = [os.path.join(result_dir, f.split('/')[-1]) for f in ant_files]

	print ('\nCalculating J and F for ' + video_name + '.')
	j_video = np.zeros(n_frames)
	f_video = np.zeros(n_frames)
	
	# Multi-core version.
	num_cores = cpu_count()
	parallel_scores = Parallel(n_jobs=num_cores)(delayed(J_and_F)(seg_file, ant_file) for seg_file, ant_file in zip(seg_files, ant_files))
	for i, score in enumerate(parallel_scores):
		j_video[i] = score[0]
		f_video[i] = score[1]

	# Single-core version.
	'''
	for i, filename in enumerate(segmentation_files):
		sys.stdout.write(str(i) + '.'); sys.stdout.flush()
		im = np.atleast_3d(Image.open(filename))[...,0]
		ant = np.atleast_3d(Image.open(annotation_files[i]))[...,0]
		j_video[i] = db_eval_iou(im,ant)
		f_video[i] = db_eval_boundary(im,ant)
	'''

	print ('\nOverall J for ' + video_name + ' is ' + str(np.mean(j_video)))
	print ('Overall F for ' + video_name + ' is ' + str(np.mean(f_video)))
	
	return j_video, f_video
