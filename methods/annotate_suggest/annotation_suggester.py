import numpy as np; import cv2; import IPython; import copy; 
import glob; import os; import time; import sys; import cPickle as pickle
from skimage.filters.rank import entropy; from skimage.morphology import disk
# Suggest next annotation frame given current annotations of video. 
# Brent Griffin, 180627

# TODO:
# Should make use of multi-threading.

class annotation_suggester:
	def __init__(self, video_dir):
		self.video_dir = video_dir
		self.types = ('*.mp4', '*.mov', '*.avi', '*.3gp', '*.mxf', '*.mpg', 
			'*.asf', '*.wav', '*.mts')
		self.video_src_dir = os.path.join(self.video_dir, 'src')
		self.annotation_dir = os.path.join(self.video_dir, 'usrAnnotate')
		try:
			self.video_properties()
			self.suggestion = True
		except:
			print ('\nVideo folder has not been initialized. Manipulation' +
									'frame cannot be accurately suggested!\n')
			self.suggestion = False
			return
		self.video_frame_classify()
		
	def video_properties(self):
		self.im_list = glob.glob(os.path.join(self.video_src_dir, '*'))
		self.im_list.sort()
		self.im_start_idx = int(os.path.basename(self.im_list[0]).split('.')[-2])
		self.num_frames = len(self.im_list)
		self.init_img = np.array(cv2.imread(self.im_list[0]), dtype=np.int8)
		self.video_height, self.video_width = self.init_img.shape[:2]
		if os.path.isdir(os.path.join(self.video_dir, 'srcSegmentation')):
			self.seg_im_list = glob.glob(os.path.join(self.video_dir, 'srcSegmentation', '*'))
		else:
			self.seg_im_list = glob.glob(os.path.join(self.video_dir, 'src', '*'))
		self.seg_im_list.sort()
		self.manip_start_idx = int(os.path.basename(self.seg_im_list[0]).split('.')[-2])
		self.manipulation_end_idx = int(os.path.basename(self.seg_im_list[-1]).split('.')[-2])
		self.n_manip_frames = len(self.seg_im_list)
		
	def video_frame_classify(self):
		# TODO: Add check to make sure pickle file is for same number of frames as current video.
		print ('Quantifying video frames.')
		pickle_file = os.path.join(self.video_dir, 'frame_selection', 'frame_classification.pk')
		if not os.path.isfile(pickle_file):
			self.frame_class_vectors = np.zeros(shape=(self.n_manip_frames,512))
			for i in range(0,self.n_manip_frames):
				img = cv2.imread(self.im_list[i])
				self.frame_class_vectors[i] = calculate_image_hist(img)
				sys.stdout.write(str(i + self.manip_start_idx) + ' '); sys.stdout.flush()
			pickle.dump(self.frame_class_vectors, open(pickle_file, 'wb'))
		else:
			self.frame_class_vectors = pickle.load(open(pickle_file, 'rb'))

	def suggest_frame(self):
		if self.suggestion:
			self.find_annotation_vector_idx()
			if self.num_annotations == 0:
				suggested_frame, distance = self.suggest_first_frame()
			else:
				suggested_frame, distance = self.suggest_next_frame()
		else:
			suggested_frame = 0; distance = 0
		return suggested_frame, distance

	def suggest_first_frame(self):
		print ('Finding first suggested annotation frame.')
		median_vector = np.median(self.frame_class_vectors, axis=0)
		self.median_distances = np.zeros(self.n_manip_frames)
		for i in range(0,self.n_manip_frames):
			self.median_distances[i] = chi2_distance(median_vector, 
												self.frame_class_vectors[i])
		return np.argmin(self.median_distances), min(self.median_distances)

	def suggest_next_frame(self):
		print ('Finding next frame to suggest (color histogram).')
		# Calculate distance of each annotation to remaining frames.
		self.annotation_distances = np.ones(self.n_manip_frames)
		# TODO: Make it so distance is only calculated as new annotations are added.
		for i, idx in enumerate(self.annotation_vector_idx):
			for j in range(0,self.n_manip_frames):
				self.annotation_distances[j] *= abs(chi2_distance(
					self.frame_class_vectors[idx], self.frame_class_vectors[j]))
		# Start with hist difference and then consider temporal.
		# Look at annotation histogram differences to video frames.
		return np.argmax(self.annotation_distances), max(self.annotation_distances)

	def find_annotation_vector_idx(self):
		# Identify frames of each annotation relative to classification vector.
		self.annotation_list = glob.glob(os.path.join(self.annotation_dir,'*'))
		self.num_annotations = len(self.annotation_list)
		self.annotation_vector_idx = np.zeros(self.num_annotations, dtype=int)
		for i in range(0,self.num_annotations):
			self.annotation_vector_idx[i] = int(os.path.basename(self.annotation_list[i]).split('.')[0]) - self.manip_start_idx

def calculate_image_hist(image):
	# Calculate image histogram.
	# https://www.pyimagesearch.com/2014/01/27/hobbits-and-histograms-a-how-to-guide-to-building-your-first-image-search-engine-in-python/
	img_hist = cv2.calcHist([image], [0,1,2], None, [8,8,8], [0,256, 0,256, 0,256])
	img_hist = cv2.normalize(img_hist, img_hist)
	return img_hist.flatten()

def chi2_distance(histA, histB, eps = 1e-10):
	# Compute chi-squared distance.
	d = np.sum([((a - b) ** 2) / (a + b + eps)
		for (a, b) in zip(histA, histB)])
	return d

'''
Misc. Notes

		# Frame distance (obvious)
		# w1 Most difference from existing annotation (look at distances of annotation frames to other frames)
		# w2 Perceived segmentation difficulty, distance between histogram for all annotations and each video frame (more distance is easier segmentation)

		# w3 - applicability to other frames (may be difficult or exhaustive to compare every frame to every other frame)

		# Think about some version looking at histogram of annotated objects.
'''	
