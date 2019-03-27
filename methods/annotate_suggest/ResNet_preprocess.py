# Script for preprocessing Marshal videos through ResNet.
# griffb@umich, 181128

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np
import tensorflow as tf
import time

from tensorflow.contrib import slim

import sys; import glob; import os; import IPython; import cPickle as pickle

from ResNet import inception_preprocessing
from ResNet import resnet_v2

## Functions ##################################################################
def resnet_process_data_dir(raw_data_dir):
	## Setup ##################################################################
	video_list = sorted(next(os.walk(raw_data_dir))[1])

	## Script #################################################################
	for _, video_name in enumerate(video_list):
		pickle_out = os.path.join(raw_data_dir, video_name, 'ResNet_preprocess.pk')
		if not os.path.isfile(pickle_out):
			print ('ResNet preprocessing for ' + video_name)
			# Image directory info.
			img_dir = os.path.join(raw_data_dir, video_name, 'src')
			img_list = sorted(glob.glob(os.path.join(img_dir, '*')))

			# Pre-process using ResNet.
			img_size = resnet_v2.resnet_v2.default_image_size
			with tf.Graph().as_default():
				processed_images = []
				for i,img in enumerate(img_list):
					image = tf.image.decode_jpeg(tf.read_file(img), channels=3)
					processed_images.append(inception_preprocessing.preprocess_image(
										image, img_size, img_size, is_training=False))
				processed_images = tf.convert_to_tensor(processed_images)

				with slim.arg_scope(resnet_v2.resnet_arg_scope()):
					# Return ResNet 2048 vector.
					logits, _ = resnet_v2.resnet_v2_50(processed_images, 
												num_classes=None, is_training=False)
				init_fn = slim.assign_from_checkpoint_fn(
							'./methods/annotate_suggest/ResNet/resnet_v2_50.ckpt', 
												slim.get_variables_to_restore())

				with tf.Session() as sess:
					init_fn(sess)
					np_images, resnet_vectors = sess.run([processed_images, logits])
					resnet_vectors = resnet_vectors[:,0,0,:]
			
			# Save preprocessed data to pickle file.
			pickle_data = {'frame_resnet_vectors': resnet_vectors}
			pickle.dump(pickle_data, open(pickle_out, 'wb'))
