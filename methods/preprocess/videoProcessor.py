# Object class for processing and (potentially) characterizing videos.
# Brent Griffin, 180427.
import numpy as np; import cv2; import IPython; import copy; 
import glob; import os; import time
from skimage.filters.rank import entropy; from skimage.morphology import disk

# TODO:
# Should make use of multi-threading.

class VideoProcessor:
	def __init__(self, video_dir):
		self.video_dir = video_dir
		self.types = ('*.mp4', '*.mov', '*.avi', '*.3gp', '*.mxf', '*.mpg', '*.asf', '*.wav', '*.mts')
		self.initializeVideoFolder()

	def initializeVideoFolder(self):
		# If segmentation image directory does not exist, make it now and build directory.
		self.video_seg_dir = os.path.join(self.video_dir, 'src')
		if not os.path.isdir(self.video_seg_dir):
			print 'Initializing video folder ' + self.video_dir
			os.makedirs(self.video_seg_dir)
			videos_grabbed = []
			for files in self.types:
				videos_grabbed.extend(glob.glob(os.path.join(self.video_dir, files)))
			self.video_file = videos_grabbed[0]
			self.buildSegDir()
		self.seg_images = os.listdir(self.video_seg_dir)

	def buildSegDir(self):
		print 'Converting video to image files.'
		vidcap = cv2.VideoCapture(self.video_file)
		count = 0
		success, image = vidcap.read()
		self.video_height, self.video_width = image.shape[:2]
		while success:
			cv2.imwrite(os.path.join(self.video_seg_dir, format(count, '05d') + '.png'), image)
			success, image = vidcap.read()
			count += 1

	def backgroundMovement(self):
		print 'Quantifying background movement.'
		self.video_background_movement = 0.0
		self.im_list = glob.glob(self.video_seg_dir + '*')
		self.im_list.sort()
		self.num_frames = len(self.im_list)
		imgPrev = np.array(cv2.imread(self.im_list[0]), dtype=np.int8)
		self.video_height, self.video_width = imgPrev.shape[:2]
		for i in range(1,self.num_frames):
			img = np.array(cv2.imread(self.im_list[i]), dtype=np.int8)
			img_diff = np.sum(np.absolute(imgPrev - img))
			# print 'Image difference is for frame ' + format(i, '02d') + ' is ' + format(img_diff, '05d')
			# print img[0][0]
			# print imgPrev[0][0]
			imgPrev = img
			self.video_background_movement += img_diff
		self.num_pixels = self.num_frames * self.video_height * self.video_width
		self.video_background_movement /= self.num_pixels
		print 'Average per pixel movement is ' + format(self.video_background_movement, '0.3f')

	def backgroundVariability(self):
		print 'Quantifying background variability.'
		self.video_background_variability = 0.0
		self.frame_check_rate = 300 # Don't have to check every frame to get sense of background variability.
		self.disk_size = 5
		self.im_list = glob.glob(self.video_seg_dir + '*')
		self.num_frames = len(self.im_list)
		imgPrev = np.array(cv2.imread(self.im_list[0]), dtype=np.int8)
		self.video_height, self.video_width = imgPrev.shape[:2]
		for i in range(1,self.num_frames,self.frame_check_rate):
			print 'Checking background variability of frame ' + str(i) + '.'
			img = np.array(cv2.imread(self.im_list[i]), dtype=np.int8)
			img_entropy = entropy(img[:,:,0], disk(self.disk_size))
			img_entropy += entropy(img[:,:,1], disk(self.disk_size))
			img_entropy += entropy(img[:,:,2], disk(self.disk_size))
			self.writeIndexImage(copy.deepcopy(img_entropy), i, 'entropy' + str(self.disk_size))
			self.video_background_variability += np.sum(img_entropy)
		self.num_checked_frames = 1 + (self.num_frames-1)/self.frame_check_rate
		self.num_pixels = self.num_checked_frames * self.video_height * self.video_width
		# self.num_pixels = self.num_frames * self.video_height * self.video_width
		self.video_background_variability /= self.num_pixels
		print 'Average per pixel variability is ' + format(self.video_background_variability, '0.3f')

	def writeIndexImage(self, img, idx, folder_name):
		directory_name = self.video_dir + '/' + folder_name + '/'
		img *= 255/np.max(img)
		if not os.path.isdir(directory_name):
			os.makedirs(directory_name)
		cv2.imwrite(directory_name + format(idx, '05d') + '.png', img)
