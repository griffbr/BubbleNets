import numpy as np; 
import cv2; 
import IPython; import copy; import glob; import os
import sys; 
sys.path.insert(0, './methods/fgbgSeg/');
# sys.path.insert(0, '.\\methods\\fgbgSeg\\') Works for windows but not unix OS.
from osvos_marshal_seg import * 
# import cPickle as pickle; 

# Brent Griffin, 180205
# Questions? griffb@umich.edu

# Current tasks:
# Create a list of dependencies.

# Future tasks:
# Enable user to generate an annotation on the spot.
# Add post-processing segmentation techniques.

def main():
	main_dir = os.getcwd()
	data_dir = os.path.join(main_dir, 'data')
	raw_data_dir = os.path.join(data_dir, 'rawData')
	
	# Get list of video directories.
	video_list = sorted(next(os.walk(raw_data_dir))[1])
	
	# Cycle through each video.
	for i, video_name in enumerate(video_list):
		# Misc. setup.
		video_dir = os.path.join(raw_data_dir, video_name)
		annotation_dir = os.path.join(video_dir, 'usrAnnotate')
		if os.path.isdir(os.path.join(video_dir, 'srcSegmentation')):
			image_dir = os.path.join(video_dir, 'srcSegmentation')
		else:
			image_dir = os.path.join(video_dir, 'src')
		# Get results file location from parameter file.
		param_file = os.path.join(video_dir, 'videoManipulationParam.txt')
		load_param = open(param_file, 'r')
		# If OS-based error for directory, consider using os.path.join()
		result_dir = os.path.join(main_dir, 'results', 'results_' + 
			load_param.read().splitlines()[13], 'fgbgSeg')
		# Get OSVOS result for each video.
		osvos_segment(image_dir, annotation_dir, result_dir, video_name, data_dir)
		print ('Finished with ' + video_name + ' segmenation.\n\n')

	# Placeholder for post-processing (e.g., n-object hypothesis, n-pixel mask padding).
	print ('\n\nFinished with all segmentations!\n\n')

if __name__ == "__main__":
	main()	
