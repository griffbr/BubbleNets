# Object class for generating an output mask from a user-guided annotation for a specific image.
# Brent Griffin 180427

import cv2
import IPython
import numpy as np

# TODO: 

class GrabCutter:
	def __init__(self, inputFile, outputFile, windowx, windowy, scale):
		# Given input file, prompt user for input with refinement and then output annotation file.
		self.img = cv2.imread(inputFile)
		self.img = cv2.resize(self.img, (0,0), fx=scale, fy=scale)
		self.refPt = []
		self.cropping = False
		self.drawing = False
		self.erasing = False
		self.selRectPt = []
		self.annotatingImage = True
		self.windowx = windowx; self.windowy = windowy
		self.outputFile = outputFile
		while self.annotatingImage:
			cv2.destroyAllWindows()
			self.user_bounding_box()
			if self.annotatingImage:
				self.initial_grab_cut()
				self.user_refine_cut()
		self.save_output_mask()

	def save_output_mask(self):
		cv2.imwrite(self.outputFile, self.mask2*255)
		cv2.destroyAllWindows()
		cv2.waitKey(20)

	def user_bounding_box(self):
		# Get user provided bounding box around annotation object.
		self.clone = self.img.copy()
		cv2.namedWindow('User Bounding Box Annotation Selection')
		cv2.setMouseCallback('User Bounding Box Annotation Selection', self.click_and_crop)
		cv2.moveWindow('User Bounding Box Annotation Selection', self.windowx, self.windowy)
		print ("Click and drag bounding box over object to annotate.\nType 'r' to restart, 's' to save, or 'n' for no object.")
		while True:
			if not self.cropping:
				cv2.imshow('User Bounding Box Annotation Selection', self.img)
			elif self.cropping and self.selRectPt:
				rectCpy = self.img.copy()
				cv2.rectangle(rectCpy, self.refPt[0], self.selRectPt[0], (255, 0, 0), 1)
				cv2.imshow('User Bounding Box Annotation Selection', rectCpy)

			key = cv2.waitKey(20) & 0xFF
			if key == ord('r'):
				self.img = self.clone.copy()
			elif key == ord('s'):
				self.img = self.clone.copy()
				break
			elif key == ord('n'):
				self.mask2 = np.zeros(self.img.shape[:2], np.uint8) 
				self.save_output_mask()
				self.annotatingImage = False
				return

		if len(self.refPt) == 2:
			self.roiPt = (min(self.refPt[0][0],self.refPt[1][0]),min(self.refPt[0][1],self.refPt[1][1]),max(self.refPt[0][0],self.refPt[1][0]),max(self.refPt[0][1],self.refPt[1][1]))

	def click_and_crop(self, event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			self.img = self.clone.copy()
			self.refPt = [(x,y)]
			self.cropping = True
		elif event == cv2.EVENT_LBUTTONUP:
			self.refPt.append((x,y))
			self.cropping = False
			cv2.rectangle(self.img, self.refPt[0], self.refPt[1], (255,0,0), 2)
			cv2.imshow('User Bounding Box Annotation Selection', self.img)
		elif event == cv2.EVENT_MOUSEMOVE and self.cropping:
			self.selRectPt = [(x,y)]

	def initial_grab_cut(self):
		print ('Performing initial grab cut given user-provided bounding box.')
		# Take initial grab cut given initial user bounding box.
		self.initialIterations = 5
		self.mask = np.zeros(self.img.shape[:2], np.uint8)
		self.bgdModel = np.zeros((1,65), np.float64)
		self.fgdModel = np.zeros((1,65), np.float64)
		cv2.grabCut(self.img,self.mask,self.roiPt,self.bgdModel,self.fgdModel,self.initialIterations,cv2.GC_INIT_WITH_RECT)
		# cv.GC_BGD, cv.GC_FGD, cv.GC_PR_BGD, cv.GC_PR_FGD, or simply pass 0,1,2,3 to image.
		# Refine mask so any area outside of ROI is definite background.
		self.mask[:,:self.roiPt[0]]=0
		self.mask[:self.roiPt[1],:]=0
		self.mask[:,self.roiPt[2]:]=0
		self.mask[self.roiPt[3]:,:]=0
		mask2 = np.where((self.mask==2)|(self.mask==0),0,1).astype('uint8')
		self.grabCutImg0 = self.img//4
		self.grabCutImg0[mask2==1] = self.img[mask2==1]

	def user_refine_cut(self):
		# User refine initial grabcut.
		self.brushWidth = 3
		self.bgBrushWidth = 5
		self.refineIterations = 15
		self.drawImg = self.grabCutImg0.copy()
		self.drawMask = self.mask.copy()
		refining = True
		cv2.destroyAllWindows()
		while refining:
			# User select correct foreground elements.
			self.fgDraw = True
			cv2.namedWindow('User Foreground / Background Refinement')
			cv2.setMouseCallback('User Foreground / Background Refinement', self.click_and_draw)
			cv2.moveWindow('User Foreground / Background Refinement', self.windowx, self.windowy)
			print ("Click and draw lines over foreground object (hold right click to undo, press +/- to change brush size).\nType 'r' to restart or 's' to save.")
			while True:
				cv2.imshow('User Foreground / Background Refinement', self.drawImg)
				key = cv2.waitKey(20) & 0xFF
				if key == ord('r'):
					self.drawImg = self.grabCutImg0.copy()
					self.drawMask = self.mask.copy()
				elif key == ord('s'):
					break
				elif key == ord('+'):
					self.brushWidth += 3
				elif key == ord('-'):
					self.brushWidth -= 3
			# User select correct background elements.
			self.fgDraw = False	
			print ("Click and draw lines over background elements (hold right click to undo, press +/- to change brush size).\nType 'r' to restart or 's' to save.")
			while True:
				cv2.imshow('User Foreground / Background Refinement', self.drawImg)
				key = cv2.waitKey(20) & 0xFF
				if key == ord('r'):
					self.drawImg = self.grabCutImg0.copy()
					print 'Warning: mask does not reset at background drawing stage!'
				elif key == ord('s'):
					break
				elif key == ord('+'):
					self.bgBrushWidth += 5
				elif key == ord('-'):
					self.bgBrushWidth -= 5
			# Refine graph cut with user-provided information.
			print ('Performing final grab cut given user-provided foreground / background annotation.')
			self.mask = self.drawMask
			cv2.grabCut(self.img,self.mask,None,self.bgdModel,self.fgdModel,self.refineIterations,cv2.GC_INIT_WITH_MASK)
			self.mask2 = np.where((self.mask==2)|(self.mask==0),0,1).astype('uint8')
			self.grabCutImg = self.img//4
			self.grabCutImg[self.mask2==1] = self.img[self.mask2==1]
			cv2.destroyAllWindows()
			cv2.imshow('Final Grab Cut Image',self.grabCutImg)
			cv2.moveWindow('Final Grab Cut Image', self.windowx, self.windowy)
			cv2.waitKey(20)
			response = 'a'
			while True:
				response = raw_input("Is annotation acceptable?\nType 'c' to continue refining annotation, 'r' to restart annotation process, or 's' to save.\n")
				if response == 'c':
					self.drawImg = self.grabCutImg.copy()
					break
				elif response == 'r':
					refining = False
					break
				elif response == 's':
					refining = False
					self.annotatingImage = False
					break

	def click_and_draw(self, event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			self.drawing = True
		elif event == cv2.EVENT_LBUTTONUP:
			self.drawing = False
		elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
			if self.fgDraw:
				self.drawImg[y-self.brushWidth:y+self.brushWidth,x-self.brushWidth:x+self.brushWidth,:] = [0,255,0]
				self.drawMask[y-self.brushWidth:y+self.brushWidth,x-self.brushWidth:x+self.brushWidth] = 1
			else:
				self.drawImg[y-self.bgBrushWidth:y+self.bgBrushWidth,x-self.bgBrushWidth:x+self.bgBrushWidth,:] = [0,0,255]
				self.drawMask[y-self.bgBrushWidth:y+self.bgBrushWidth,x-self.bgBrushWidth:x+self.bgBrushWidth] = 0
			cv2.imshow('User Foreground / Background Refinement', self.drawImg)
		# Add ability to erase marks during refinement process.
		elif event == cv2.EVENT_RBUTTONDOWN:
			self.erasing = True
		elif event == cv2.EVENT_RBUTTONUP:
			self.erasing = False
		elif event == cv2.EVENT_MOUSEMOVE and self.erasing:
			if self.fgDraw:
				self.drawImg[y-self.brushWidth:y+self.brushWidth,x-self.brushWidth:x+self.brushWidth,:] = self.grabCutImg0[y-self.brushWidth:y+self.brushWidth,x-self.brushWidth:x+self.brushWidth,:]  
				self.drawMask[y-self.brushWidth:y+self.brushWidth,x-self.brushWidth:x+self.brushWidth] = self.mask[y-self.brushWidth:y+self.brushWidth,x-self.brushWidth:x+self.brushWidth] 
			else:
				self.drawImg[y-self.bgBrushWidth:y+self.bgBrushWidth,x-self.bgBrushWidth:x+self.bgBrushWidth,:] = self.grabCutImg0[y-self.bgBrushWidth:y+self.bgBrushWidth,x-self.bgBrushWidth:x+self.bgBrushWidth,:] 
				self.drawMask[y-self.bgBrushWidth:y+self.bgBrushWidth,x-self.bgBrushWidth:x+self.bgBrushWidth] = self.mask[y-self.bgBrushWidth:y+self.bgBrushWidth,x-self.bgBrushWidth:x+self.bgBrushWidth] 
			cv2.imshow('User Foreground / Background Refinement', self.drawImg)
