# Select annotation frames for Marshal using BubbleNets.
# Brent Griffin, 181128, griffb@umich.edu

import tensorflow as tf
#from tensorflow.contrib import slim
import slim # changed
import IPython
import numpy as np
from copy import deepcopy
import cPickle as pickle
import os; import glob; import time
from BubbleNets import bn_input
from BubbleNets import bn_models
from BubbleNets import bn_utils

def BubbleNets_sort(raw_data_dir, model='BNLF'):
	# Sorting parameters.
	n_frames = 5
	n_ref = n_frames - 2
	n_batch = 5
	n_sorts = 1
	# Generate video list for frame selection.
	video_list = sorted(next(os.walk(raw_data_dir))[1])

	# Prepare the tf input data.
	tf.logging.set_verbosity(tf.logging.INFO)
	input_vector = tf.placeholder(tf.float32, [None, (2048 + 1) * n_frames])
	input_label = tf.placeholder(tf.float32, [None, 1])
	# Select network model.
	if model == 'BNLF':
		ckpt_filename = './methods/annotate_suggest/BubbleNets/BNLF_181030.ckpt-10000000'
		predict, end_pts = bn_models.BNLF(input_vector, is_training=False, n_frames=n_frames)
	else:	
		ckpt_filename = './methods/annotate_suggest/BubbleNets/BN0_181029.ckpt-10000000'
		predict, end_pts = bn_models.BN0(input_vector, is_training=False, n_frames=n_frames)
	
	# Initialize network and select frame.
	init = tf.global_variables_initializer()
	tic = time.time()
	with tf.Session() as sess:
		init_fn = slim.assign_from_checkpoint_fn(ckpt_filename, 
												slim.get_variables_to_restore())
		init_fn(sess)
		# Go through each video in list.
		for j, vid_name in enumerate(video_list):
			# Check if sort selection has aleady been made.
			select_dir = os.path.join(raw_data_dir,vid_name,'frame_selection')
			if not os.path.isdir(select_dir):
				os.makedirs(select_dir)	
			text_out = os.path.join(select_dir, '%s.txt' % model)
			if os.path.isfile(text_out):
				print('%s already has %s frame selection!' %(vid_name,model))
				continue
			print ('\nRunning BubbleNets %s for video %i %s' %(model,j,vid_name))
			# Load ResNet vectors for network input.
			vector_file = os.path.join(raw_data_dir, vid_name, 'ResNet_preprocess.pk')
			input_data = bn_input.BN_Input(vector_file, n_ref=n_ref)
			num_frames = input_data.n_frames
			rank_bn = range(0,num_frames)                                              

			# BubbleNets Deep Sort.
			bubble_step = 1                                                            
			while bubble_step < num_frames * n_sorts:                                            
				a = deepcopy(rank_bn[0])                                                         
				for i in range(1,num_frames):                                          
					b = deepcopy(rank_bn[i])                                         
					batch_vector = input_data.video_batch_n_ref_no_label(a,b,batch=n_batch)
					frame_select = sess.run(predict, feed_dict={input_vector: batch_vector})
					# If frame b is preferred, use frame b for next comparison.        
					if np.mean(frame_select[0]) < 0:
						rank_bn[i-1] = a                                               
						rank_bn[i] = b                                                 
						a = deepcopy(b)                                                          
					else:                                                              
						rank_bn[i-1] = b                                               
						rank_bn[i] = a                                                 
				bubble_step += 1                                                   

			# Write out frame selection to text file.
			select_idx = rank_bn[-1]
			img_file = os.path.basename(sorted(glob.glob(os.path.join(
					raw_data_dir,vid_name,'src','*')))[select_idx])
			statements = [model,'\n',str(select_idx),'\n',img_file,'\n']
			bn_utils.print_statements(text_out, statements)
		sess.close()
	tf.reset_default_graph()

	toc = time.time()
	print('finished selecting all %s frames on list!' % model)
	print('Runtime is ' + str(toc-tic))
