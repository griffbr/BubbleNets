# Quickly annotate images using BubbleNets suggested frames.
# Brent Griffin, 181127
# Questions? griffb@umich.edu
import numpy as np; import cv2; import IPython; import copy; import glob; import os
import sys; 
cwd = os.getcwd()
sys.path.insert(0, os.path.join(cwd, 'methods', 'preprocess'));
sys.path.insert(0, os.path.join(cwd, 'methods', 'annotate_suggest'));
from grabCutClass import * 
from videoProcessor import *
from annotation_suggester import *
from color_hist_frame_select import *
from ResNet_preprocess import *
from BubbleNets_frame_select import *
from BubbleNets import bn_utils

user_scale = True
user_select = True

def get_user_annotation(videoDir):
	videoName = os.path.basename(videoDir)
	print ('\n\nGenerating user-guided annotation for ') + os.path.basename(videoDir) + ('.\n')
	annotationDir = os.path.join(videoDir, 'usrAnnotate')
	if not os.path.isdir(annotationDir):
		os.makedirs(annotationDir)
	imageDir = os.path.join(videoDir, 'src')
	imageFiles = glob.glob(os.path.join(imageDir,'*'))
	imageFiles.sort()
	# Get list of suggested annotations.
	ant_idx, ant_file = read_annotation_list(os.path.join(videoDir,'frame_selection','all.txt'))
	# Run automated annotation suggestion (just to get folder info).
	suggester = annotation_suggester(videoDir)
	userAnnotating = True
	while userAnnotating:
		antImageFiles = glob.glob(os.path.join(annotationDir,'*'))
		nAntImgs = len(antImageFiles)
		print ('Currently ') + str(nAntImgs) + (' annotation image(s):')
		for i in range(0,nAntImgs):
			ant_name = os.path.basename(antImageFiles[i])
			print (ant_name)
			if ant_name in ant_file:
				del ant_file[ant_file.index(ant_name)]
		print ('Suggested annotation frames remaining:')
		print (ant_file)
		# Ask if there are any other images they would like to annotate?
		response = raw_input('Annotate another image? (y or n)\n')
		if not response in {'y','Y','Yes','yes'}:
			userAnnotating = False
		if userAnnotating:
			# User annotation specific image.
			while True:
				if user_select:
					annotationImageIdx = input('What is preferred annotation image index? (' + str(os.path.basename(imageFiles[0])) + '-' + str(os.path.basename(imageFiles[-1])) + ' possible)\n')
					annotationImageIdx -= suggester.manip_start_idx
				else: annotationImageIdx = ant_idx[0]
				try:
					imageDir = imageFiles[int(annotationImageIdx)]
					annotationImage = cv2.imread(imageDir)
					windowx = 100; windowy = 100
					if user_scale:
						cv2.imshow('Annotation Image', annotationImage)
						cv2.moveWindow('Annotation Image', windowx, windowy)
						cv2.waitKey(20)
						scale = input('What is preferred scale? (e.g., 1, 2, or 0.5)\n')
					else: scale = 1;
					annotationImageScaled = cv2.resize(annotationImage, (0,0), fx=scale, fy=scale)
					cv2.imshow('Scaled Annotation Image', annotationImageScaled)
					cv2.moveWindow('Scaled Annotation Image', windowx, windowy)
					cv2.waitKey(20)
					response = raw_input('Is annotation frame acceptable? (y or n)\n')
					if response in {'y','Y','Yes','yes'}:
						cv2.destroyAllWindows()
						#cv2.waitKey()
						break
				except:
					print ('Image ') + str(annotationImageIdx) + (' does not exist!')
			outputMaskDir = os.path.join(annotationDir,os.path.basename(imageDir).split('.')[0] + '.png')
			# Let user annotate selected image.
			GrabCutter(imageDir, outputMaskDir, windowx, windowy, scale)
			save_extra_image_copy(imageDir, videoDir, nAntImgs)

def save_extra_image_copy(image_dir, video_dir, annotation_frame_num):
	# TODO: add visualization for mask on top of image.
	print ('Saving extra copy of annotation image for development.')
	extra_image_dir = os.path.join(video_dir, 'annotation_imgs')
	if not os.path.isdir(extra_image_dir):
		os.makedirs(extra_image_dir)
	cv2.imwrite(os.path.join(extra_image_dir, format(annotation_frame_num, '02d')
		+ '_annotation_' + os.path.basename(image_dir).split('.')[0] + '.jpg'), 
		cv2.imread(image_dir))

def read_annotation_list(text_file):
	read_list = bn_utils.read_list_file(text_file)
	n_ant = int(read_list[0].split(' ')[0])
	ant_idx = []; ant_file = []
	for i in range(n_ant):
		ant_idx.append(int(read_list[i*2+1]))
		ant_file.append(read_list[i*2+2])
	return ant_idx, ant_file

def main():
	mainDir = os.getcwd()
	dataDir = os.path.join(mainDir, 'data')
	rawDataDir = os.path.join(dataDir, 'rawData')
	# Get list of video directories.
	videoList = sorted(next(os.walk(rawDataDir))[1])
	# Preprocess through ResNet.
	resnet_process_data_dir(rawDataDir)
	# Get BubbleNets suggested annotated frame.
	BubbleNets_sort(rawDataDir, model='BNLF')
	BubbleNets_sort(rawDataDir, model='BN0')
	color_hist_frame_select(rawDataDir, annotate_rate=int(10e6))
	# Cycle through each video.
	for i, videoName in enumerate(videoList):
		# Misc. setup.
		videoDir = os.path.join(rawDataDir, videoName)
		# Check for src folder. If it doesn't exist, make it from video.
		VideoProcessor(videoDir)
		# Check for annotation files, if not there, ask users which frames they would like annotated.
		get_user_annotation(videoDir)
		print ('Finished with ') + videoName + (' annotation.\n\n')
	print ('\n\nFinished with all annotations!\n\n')

if __name__ == "__main__":
	main()	
