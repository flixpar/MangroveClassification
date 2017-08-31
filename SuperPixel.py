import cv2
import numpy as np
import scipy.stats as stats
from scipy import ndimage
from skimage import feature, filters, exposure, img_as_float

# PARAMS:

hog_args = {}
hog_args["block_norm"] = "L2-Hys"
hog_args["pixels_per_cell"] = (8, 8)
hog_args["cells_per_block"] = (3, 3)
hog_args["orientations"] = 9

lbp_args = {}
lbp_args["P"] = 40
lbp_args["R"] = 5
lbp_args["method"] = "ror"

ROTATIONS = 5
ROTATION_SIZE = 20

class SuperPixel:

	def __init__(self, id_num, src_img, lbl_img, mask_img, avg_size):

		self.checkSuperPixel(src_img)

		self.id = id_num
		self.size = avg_size
		self.bounds = self.getBoundingBox(mask_img)

		self.checkBounds()

		self.mask = self.cropMask(mask_img)
		self.features = self.generateFeatures(src_img)

		if lbl_img is not None:
			self.label = self.findLabel(lbl_img)

	# get the min and max coordinates of the superpixel in the input image
	def getBoundingBox(self, mask_img):

		mask = (mask_img == self.id)
		height, width = mask_img.shape

		min_extent = [0,0]
		max_extent = [0,0]

		for i in range(height):
			if (mask[i,:]).any():
				max_extent[0] = i
				if min_extent[0] == 0:
					min_extent[0] = i
			elif max_extent[0] != 0:
				break
		for j in range(width):
			if (mask[:,j]).any():
				max_extent[1] = j
				if min_extent[1] == 0:
					min_extent[1] = j
			elif max_extent[1] != 0:
				break

		del(mask)
		return tuple(min_extent), tuple(max_extent)

	# crop the mask to bounds
	def cropMask(self, mask_img):
		row_min, col_min = self.bounds[0]
		row_max, col_max = self.bounds[1]
		msk = mask_img[row_min:row_max, col_min:col_max]
		mask = (msk == self.id)

		if not mask.any():
			raise ValueError("Empty mask with bounds: {}.".format(self.bounds))

		return mask

	# crop and resize the source image to the correct size for feature description
	def processImg(self, src_img, theta):

		row_min, col_min = self.bounds[0]
		row_max, col_max = self.bounds[1]
		roi = src_img[row_min:row_max, col_min:col_max]
		roi = cv2.resize(roi, self.size)

		rotated = ndimage.rotate(roi, theta, reshape=False)

		return rotated

	def generateFeatures(self, src_img):

		features = []

		rot_start = int(-1 * (ROTATIONS-1) * ROTATION_SIZE / 2)
		rot_end = int((ROTATIONS) * ROTATION_SIZE / 2)
		for theta in range(rot_start, rot_end, ROTATION_SIZE):
			roi = self.processImg(src_img, theta)
			gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
			assert((gray.shape[0]==self.size[1]) and (gray.shape[1]==self.size[0]))

			hog = feature.hog(gray, **hog_args)

			lbp = feature.local_binary_pattern(gray, **lbp_args)
			lbp_n_bins = 256 # int(lbp.max() + 1)
			lbp_hist, _ = np.histogram(lbp, density=True, bins=lbp_n_bins)  #, range=(0, lbp_n_bins))
			assert(lbp_hist.size == lbp_n_bins)

			if not np.isfinite(lbp).all():
				print("Err: lbp not finite")

			if lbp.max() == 0 and not np.isfinite(lbp_hist).all():
				print("Err: bad lbp hist.")
				lbp_hist = np.zeros((lbp_n_bins))
				lbp_hist[0] = lbp.size

			hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
			r_hist, _ = np.histogram(hsv[:,:,0], bins=64)
			g_hist, _ = np.histogram(hsv[:,:,1], bins=64)
			b_hist, _ = np.histogram(hsv[:,:,2], bins=64)
			hist = np.concatenate((r_hist, g_hist, b_hist))
			assert(hist.size == 192)

			features.append(hog)
			features.append(lbp_hist)
			features.append(hist)

		assert(len(features)==3*ROTATIONS)
		result = np.concatenate(features)
		assert(np.isfinite(result).all())
		return result

	# given the image of all labels, find the label for this superpixel
	def findLabel(self, lbl_img):

		row_min, col_min = self.bounds[0]
		row_max, col_max = self.bounds[1]
		roi = lbl_img[row_min:row_max, col_min:col_max]

		roi = roi[np.where(self.mask == True)]
		mode = stats.mode(roi, axis=None)
		mode = mode[0][0]

		del(roi)
		return mode

	#######################
	## HELPER FUNCTIONS: ##
	#######################

	def checkSuperPixel(self, img):
		if not img.any():
			raise ValueError("No input image data given.")
		if len(img.shape) != 3:
			raise ValueError("Misshapen image.")
		if img.shape[0] <= 20:
			raise ValueError("Input image too short.")
		if img.shape[1] <= 20:
			raise ValueError("Input image too narrow.")
		if img.shape[2] != 3:
			raise ValueError("Not enough channels in input image.")

	def checkBounds(self):
		row_min, col_min = self.bounds[0]
		row_max, col_max = self.bounds[1]

		if not (0 <= row_min < row_max):
			raise ValueError("No region found.")
		if not (0 <= col_min < col_max):
			raise ValueError("No region found.")
