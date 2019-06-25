'''
Train and evaluate OSVOS segmentation on all frames in example directory.
Providing this as example for generating BubbleNets performance labels.
Brent Griffin, 190624
Questions? griffb@umich.edu
'''
import os
import IPython
import sys
import glob
import cPickle as pickle
import numpy as np
import shutil

import seg_eval
sys.path.insert(0, '../methods/fgbgSeg/')
import osvos_marshal_seg

# Misc. Init.
base_dir = os.getcwd()
PICKLE_FILE = base_dir + '/labels/new_BN_labels.pk'
DATA_DIR = base_dir + '/source_data/'
RESULT_DIR = base_dir + '/temp_results/'
TRAIN_ITER = 500

# Get video list.
vid_list = sorted(next(os.walk(DATA_DIR))[1])

# Get list of prev. generated labels if available (or start from scratch).
if os.path.isfile(PICKLE_FILE):
	pickle_out = pickle.load(open(PICKLE_FILE,'rb'))
	vid_prev = pickle_out.keys()
else:
	pickle_out = {'NA':0}; vid_prev = []

# Generate training labels for BubbleNets.
for i, vid_name in enumerate(vid_list):
	if vid_name in vid_prev:
		print ('%s already evaluated.' % vid_name); continue
	print('Running segmentations for video %i %s.' % (i, vid_name))
	seg_dir = os.path.join(RESULT_DIR, vid_name)
	vid_dir = os.path.join(DATA_DIR, vid_name)
	img_dir = os.path.join(vid_dir, 'src')
	ant_dir = os.path.join(vid_dir, 'usrAnnotate')
	gt_dir = os.path.join(vid_dir, 'ground_truth')
	gt_list = sorted(glob.glob(os.path.join(gt_dir,'*')))
	n_frm = len(gt_list)

	J_all = np.zeros(shape=(n_frm, n_frm))
	F_all = np.zeros(shape=(n_frm, n_frm))
	if not os.path.isdir(ant_dir): os.makedirs(ant_dir)

	# Use all possible frames for annotation and then evaluate.
	for j, ant_file in enumerate(gt_list):
		print('Starting frame %i' % j)
		shutil.rmtree(ant_dir)
		os.makedirs(ant_dir)
		shutil.copy2(ant_file, os.path.join(ant_dir, format(j,'05d') + '.png'))
		# Process training on image j (currently OSVOS segmentation).
		osvos_marshal_seg.osvos_segment(img_dir, ant_dir, RESULT_DIR, vid_name, 
			RESULT_DIR, iters = TRAIN_ITER)
		# Performance evaluation after training on image j (currently J and F).
		J_all[j], F_all[j] = seg_eval.segmentation_score(seg_dir, gt_dir)

	# Final video-wide eval and save results.
	J_mean = np.mean(J_all, axis=1)
	F_mean = np.mean(F_all, axis=1)
	temp_dct = {'J_all': J_all, 'F_all': F_all, 'J_mean': J_mean, 
		'F_mean': F_mean, 'n_frames': n_frm}
	pickle_out[vid_name] = temp_dct
	pickle.dump(pickle_out, open(PICKLE_FILE, 'wb'))
	print('Finished with %s label generation.\n\n' % vid_name)

print('\n\nFinished with all video frame label generation!\n\n')
