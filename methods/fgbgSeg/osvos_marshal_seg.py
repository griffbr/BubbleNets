# Modified to work with marshal framework using OSVOS as a segmentation function.
# Questions? Ask griffb@umich.edu.

"""
Sergi Caelles (scaelles@vision.ee.ethz.ch)

This file is part of the OSVOS paper presented in:
    Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixe, Daniel Cremers, Luc Van Gool
    One-Shot Video Object Segmentation
    CVPR 2017
Please consider citing the paper if you use this code.
"""
import os
import sys
from PIL import Image
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import matplotlib.pyplot as plt
# Import OSVOS files
root_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(root_folder))
import osvos_mod
from dataset_mod import Dataset

# griffb added
import IPython
import cv2

# griffb tasks

# OSVOS inputs:
# List of images.
# Annotation location.
# Output mask folder.
def osvos_segment(image_dir, annotation_dir, mask_dir, seq_name, data_dir, model_name=''):
	if model_name == '':
		model_name = seq_name
	print '\nRunning OSVOS Segmentation for ' + seq_name + '\n'
	# Set up paths.
	result_path = os.path.join(mask_dir, seq_name)
	result_path_vis = os.path.join(mask_dir+'Visualization', seq_name)
	os.chdir(root_folder)
	parent_path = os.path.join('models', 'OSVOS_parent', 'OSVOS_parent.ckpt-50000')
	logs_path = os.path.join(data_dir,'models', model_name)
	# User defined parameters
	gpu_id = 0
	train_model = True
	# Train parameters
	#max_training_iters = 500
	#print 'max training iters temp reduced!!!!!!!!!!!!!!!'
	#max_training_iters = 100
	#print ('max training iters increased!')
	max_training_iters = 1000
	max_training_iters = 50
	# Define Dataset
	test_frames = sorted(os.listdir(image_dir))
	test_imgs = [os.path.join(image_dir, frame) for frame in test_frames]
	test_img_ext = test_frames[0].split('.')[1]
	# Define annotation images.
	annotate_frames = sorted(os.listdir(annotation_dir))
	if '.DS_Store' in annotate_frames:
		annotate_frames.remove('.DS_Store')
	annotate_imgs = [os.path.join(annotation_dir, frame) for frame in annotate_frames]
	if train_model:
		train_imgs = [os.path.join(image_dir, frame.split('.')[0] + '.' + test_img_ext)  + ' ' + annotate_imgs[i] for i,frame in enumerate(annotate_frames)]
		# train_imgs = [os.path.join(image_dir, frame) + ' ' + annotate_imgs[i] for i,frame in enumerate(annotate_frames)]
		# train_imgs = [os.path.join(image_dir, annotate_frames[0]) + ' ' + annotate_imgs[0]]
		dataset = Dataset(train_imgs, test_imgs, './', data_aug=True)
	else:
	    dataset = Dataset(None, test_imgs, './')
	
	print 'Data structure setup.'
	# Train the network
	if train_model:
	    # More training parameters
	    batch_size = 1 # len(train_imgs)
	    learning_rate = 1e-8
	    save_step = max_training_iters
	    side_supervision = 3
	    display_step = 10
	    with tf.Graph().as_default():
		with tf.device('/gpu:' + str(gpu_id)):
			global_step = tf.Variable(0, name='global_step', trainable=False)
			osvos_mod.train_finetune(dataset, parent_path, side_supervision, learning_rate, 
				logs_path, max_training_iters,save_step, display_step, 
				global_step, iter_mean_grad=1, batch_size=batch_size, ckpt_name=model_name)

	# Test the network
	with tf.Graph().as_default():
	    with tf.device('/gpu:' + str(gpu_id)):
		checkpoint_path = os.path.join(logs_path, model_name+'.ckpt-'+str(max_training_iters))
		osvos_mod.test(dataset, checkpoint_path, result_path)

	# Show results
	overlay_color = [255, 0, 0]
	transparency = 0.6
	plt.ion()

	if not os.path.isdir(result_path_vis):
	    os.makedirs(result_path_vis)

	for img_p in test_frames:
	    frame_num = img_p.split('.')[0]
	    img = np.array(Image.open(os.path.join(image_dir, img_p)))
	    mask = np.array(Image.open(os.path.join(result_path, frame_num+'.png')))
	    mask = mask/np.max(mask)
	    im_over = np.ndarray(img.shape)
	    im_over[:, :, 0] = (1 - mask) * img[:, :, 0] + mask * (overlay_color[0]*transparency + (1-transparency)*img[:, :, 0])
	    im_over[:, :, 1] = (1 - mask) * img[:, :, 1] + mask * (overlay_color[1]*transparency + (1-transparency)*img[:, :, 1])
	    im_over[:, :, 2] = (1 - mask) * img[:, :, 2] + mask * (overlay_color[2]*transparency + (1-transparency)*img[:, :, 2])
	    cv2.imwrite(os.path.join(result_path_vis, frame_num+'.png'),cv2.cvtColor(im_over.astype(np.uint8), cv2.COLOR_RGB2BGR))
	if False: # Show actual images or not.
		plt.imshow(im_over.astype(np.uint8))
		plt.axis('off')
		plt.show()
		plt.pause(0.01)
		plt.clf()
