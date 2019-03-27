# Models tested and used for BubbleNets.

import IPython

import tensorflow as tf
from tensorflow.contrib import slim

def BNLF(inputs, is_training=True, scope='deep_regression', n_frames=5):
	with tf.variable_scope(scope, 'deep_regression', [inputs]):
		end_points = {}
		with slim.arg_scope([slim.fully_connected],
			activation_fn=tf.nn.leaky_relu,
			weights_regularizer=slim.l1_regularizer(2e-6)):

			# Use frame indices later in network.
			input2 = inputs[:,-n_frames:]

			net = slim.fully_connected(inputs, 256, scope='fc1')
			end_points['fc1'] = net

			net = slim.dropout(net, 0.8, is_training=is_training)
			net = tf.concat((net,input2),axis=1)
			
			net = slim.fully_connected(net, 128, scope='fc2')
			end_points['fc2'] = net
			
			net = slim.dropout(net, 0.8, is_training=is_training)
			net = tf.concat((net,input2),axis=1)

			net = slim.fully_connected(net, 64, scope='fc3')
			end_points['fc3'] = net
			
			net = slim.dropout(net, 0.8, is_training=is_training)
			net = tf.concat((net,input2),axis=1)
			
			net = slim.fully_connected(net, 32, scope='fc4')
			end_points['fc4'] = net

			predictions = slim.fully_connected(net, 1, activation_fn=None, 
				scope='prediction', normalizer_fn=None)
			end_points['out'] = predictions

			return predictions, end_points

def BN0(inputs, is_training=True, scope='deep_regression', n_frames=5):
	with tf.variable_scope(scope, 'deep_regression', [inputs]):
		end_points = {}
		with slim.arg_scope([slim.fully_connected],
			activation_fn=tf.nn.leaky_relu,
			weights_regularizer=slim.l1_regularizer(2e-6)):

			# Use frame indices later in network.
			input2 = inputs[:,-n_frames:]

			net = slim.fully_connected(inputs, 256, scope='fc1')
			end_points['fc1'] = net

			net = tf.concat((net,input2),axis=1)
			net = slim.dropout(net, 0.8, is_training=is_training)
			
			net = slim.fully_connected(net, 128, scope='fc2')
			end_points['fc2'] = net
			
			net = tf.concat((net,input2),axis=1)
			net = slim.dropout(net, 0.8, is_training=is_training)

			net = slim.fully_connected(net, 64, scope='fc3')
			end_points['fc3'] = net
			
			net = tf.concat((net,input2),axis=1)
			net = slim.dropout(net, 0.8, is_training=is_training)
			
			net = slim.fully_connected(net, 32, scope='fc4')
			end_points['fc4'] = net

			predictions = slim.fully_connected(net, 1, activation_fn=None, 
				scope='prediction', normalizer_fn=None)
			end_points['out'] = predictions

			return predictions, end_points
