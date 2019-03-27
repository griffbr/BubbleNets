# Suggest annotation frames in video using Color Histogram-based approach.
'''
Make color histogram-based annotation frame suggestions for videos.
Brent Griffin, 181130.
Questions? griffb@umich.edu

Input: raw_data_dir
Output: suggested annotation frame text files for each video.
'''
import IPython; import numpy as np
import os; import sys; 
cwd = os.getcwd()
sys.path.insert(0, os.path.join(cwd, 'methods', 'preprocess'));
sys.path.insert(0, os.path.join(cwd, 'methods', 'annotate_suggest'));
from BubbleNets import bn_utils
from annotation_suggester import *

def get_prev_frame_indices(video_dir, models=['BNLF','BN0']):
	indices = np.zeros(0, dtype=int)
	for i, model in enumerate(models):
		text_in = os.path.join(video_dir, 'frame_selection', '%s.txt' % model)
		read_list = bn_utils.read_list_file(text_in)
		if len(read_list) == 3:
			indices = np.append(indices, int(read_list[1]))
	return np.unique(indices)

def color_hist_frame_select(raw_data_dir, annotate_rate = 50):

	# Generate video list.
	video_list = sorted(next(os.walk(raw_data_dir))[1])

	# Generate suggestions for each video.
	for j, vid_name in enumerate(video_list):

		# Check if suggestion has already been made.
		text_out = os.path.join(raw_data_dir,vid_name,'frame_selection','all.txt')
		if os.path.isfile(text_out):
			print('%s already has color histogram frame selection!' % text_out)
			continue
		print('\nRunning color hist frame selections for video %i %s'%(j,vid_name))
		video_dir = os.path.join(raw_data_dir, vid_name)
		suggest = annotation_suggester(video_dir)
		
		# Add frames to suggester that have already been selected.
		suggest.annotation_vector_idx = get_prev_frame_indices(video_dir)

		# Add new annotation frames using color histogram-based selection.
		for i in range(annotate_rate, suggest.n_manip_frames, annotate_rate):
			next_idx, _ = suggest.suggest_next_frame()	
			suggest.annotation_vector_idx = np.append(suggest.annotation_vector_idx,next_idx)
		
		# Write suggestions out to text file.
		img_files = sorted(glob.glob(os.path.join(video_dir,'src','*')))
		statements = ['%i suggested annotations for %s\n' % 
								(len(suggest.annotation_vector_idx),vid_name)]
		for i, idx in enumerate(suggest.annotation_vector_idx):
			statements.append(str(idx)+'\n')
			statements.append(os.path.basename(img_files[int(idx)])+'\n')
		bn_utils.print_statements(text_out, statements)	
