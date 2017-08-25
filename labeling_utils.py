import os
import sys
import cv2
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from enum import Enum

# for now we are using the SEEDS algortithm to segment the image into superpixels
def oversegment(img, args):

	num_superpixels, num_levels, iterations = args

	height, width, channels = img.shape
	seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, num_superpixels, num_levels)
	seeds.iterate(img, iterations)
	labels = seeds.getLabels()
	contours = seeds.getLabelContourMask()
	num_superpixels = seeds.getNumberOfSuperpixels()

	return labels, num_superpixels

# takes the superpixel mask and calculates the average size of the superpixels
def calc_avg_size(mask_img, num_superpixels):

	avg_width = 0
	avg_height = 0

	max_label = mask_img.max()

	# for i in tqdm(range(num_superpixels)):
	for i in range(num_superpixels):

		n = random.randint(0, max_label-1)
		mask = (mask_img == n)

		min_extent = [0,0]
		max_extent = [0,0]

		for k in range(mask.shape[0]):
			if (mask[k,:]).any():
				max_extent[0] = k
				if min_extent[0] == 0:
					min_extent[0] = k
			elif max_extent[0] != 0:
				break
		for k in range(mask.shape[1]):
			if (mask[:,k]).any():
				max_extent[1] = k
				if min_extent[1] == 0:
					min_extent[1] = k
			elif max_extent[1] != 0:
				break

		avg_height += (max_extent[0] - min_extent[0])
		avg_width += (max_extent[1] - min_extent[1])

		del(mask)

	avg_height /= num_superpixels
	avg_width /= num_superpixels

	return int(avg_height), int(avg_width)


# plot a confusion matrix to matplotlib and save it
def plotConfusionMatrix(cfm, num_classes, out_fn):

	plt.figure()

	# normalize and display
	cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]
	plt.imshow(cfm, interpolation='nearest', cmap=plt.cm.Blues)

	# setup the title and axes
	plt.title("Normalized Confusion Matrix")
	plt.colorbar()
	tick_marks = np.arange(num_classes)
	plt.xticks(tick_marks, range(num_classes), rotation=45)
	plt.yticks(tick_marks, range(num_classes))

	# label the axes
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

	# save image and display
	plt.savefig(out_fn, dpi=100)
	plt.show()

class Namespace:
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)

class filemode(Enum):
	READ = 0
	WRITE = 1

class writer:
	def __init__(self, *writers):
		self.writers = writers

	def write(self, text):
		for w in self.writers:
			w.write(text)

	def flush(self):
		for w in self.writers:
			w.flush()
