import os
import sys
import gc
import time
from enum import Enum

import cv2
import numpy as np

import glob
import yaml
import pickle
from tqdm import tqdm
from random import sample

import itertools as it
import multiprocessing as mp

from sklearn.svm import SVC as SVM
from sklearn import metrics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from SuperPixel import SuperPixel
from Preprocessor import Preprocessor, Reducers
from labeling_utils import *

def main():

	config = init()
	print("Mangrove Classification")
	print("SVM Classifier")
	print("Classifying with: ", config.hyperparams)
	print("Segmenting with: ", config.segment)
	print("="*50)

	if config.regenerate_features:

		print("Getting data lists...")
		(train_img_list, train_lbl_list), (test_img_list, test_lbl_list) = get_image_paths(config.image_paths)

		print("Computing average SuperPixel size...")
		avg_size = compute_avg_size(train_img_list, config.segment)
		print("Average superpixel size is: {}".format(avg_size))

		print("Generating features...")
		train_features, train_labels = extract_features(train_img_list, train_lbl_list, config, avg_size)
		test_features, test_labels = extract_features(test_img_list, test_lbl_list, config, avg_size)

		print("Preprocessing...")
		preprocessor = get_preprocessor(config.preprocess, train_features)
		train_features = preprocessor.process(train_features)
		test_features = preprocessor.process(test_features)

		print("Saving features...")
		train_file = open(config.save_path["train_features"], 'wb')
		test_file = open(config.save_path["test_features"], 'wb')

		train_save = (train_features, train_labels)
		test_save = (test_features, test_labels)

		pickle.dump(train_save, train_file)
		pickle.dump(test_save, test_file)

		train_file.close()
		test_file.close()

	else:

		print("Loading features...")
		train_file = open(config.save_path["train_features"], 'rb')
		test_file = open(config.save_path["test_features"], 'rb')

		train_features, train_labels = pickle.load(train_file)
		test_features, test_labels = pickle.load(test_file)

		train_file.close()
		test_file.close()

	print("Unique Labels:")
	print(np.unique(train_labels))

	print("Training...")
	start_time = time.time()
	classifier = SVM(**config.hyperparams)
	classifier.fit(train_features, train_labels)
	elapsed_time = time.time() - start_time
	print("Training took {0:.2f} seconds".format(elapsed_time))

	print("Predicting...")
	start_time = time.time()
	pred = classifier.predict(test_features)
	elapsed_time = time.time() - start_time
	print("Predicting took {0:.2f} seconds".format(elapsed_time))

	report, acc, iou, precision, confusion = evaluate(test_labels, pred)
	save_results(report, acc, iou, precision, confusion, config.save_path["results"])

	print(report)
	print()
	print("Accuracy: {0:.4f}".format(acc))
	print("Precision: {0:.4f}".format(precision))
	print("IOU: {0:.4f}".format(iou))
	print()

#####################
## HELPER METHODS: ##
#####################

def get_image_paths(paths):

	train_images_names = sorted(glob.glob(paths["train"] + "/images/*.tif"))
	train_annotation_names = sorted(glob.glob(paths["train"] + "/annotations/*.png"))

	test_images_names = sorted(glob.glob(paths["test"] + "/images/*.tif"))
	test_annotation_names = sorted(glob.glob(paths["test"] + "/annotations/*.png"))

	return (train_images_names, train_annotation_names), (test_images_names, test_annotation_names)

def extract_features(image_paths, mask_paths, config, avg_size):

	threadpool = mp.Pool(config.processors)

	# for debugging:
	image_paths = sample(image_paths, len(image_paths))[:20]
	mask_paths = sample(mask_paths, len(mask_paths))[:20]

	args = zip(image_paths, mask_paths, it.repeat(config.segment), it.repeat(avg_size))

	results = threadpool.starmap(get_features, args)
	features_list = [r[0] for r in results]
	labels_list = [r[1] for r in results]

	features = np.concatenate(features_list)
	labels = np.concatenate(labels_list)

	unique, counts = np.unique(labels, return_counts=True)
	print("\nClassifications found in imagery: {0}\nCounts: {1}\n".format(unique, counts))

	return features, labels

def get_features(img_path, mask_path, config, avg_size):

	start_time = time.time()

	## READ IMAGES: ##
	img = cv2.imread(img_path)
	mask = cv2.imread(mask_path, 0)

	## OVERSEGMENT: ##
	spixel_args = (config["approx_num_superpixels"], config["num_levels"], config["iterations"])
	segment_mask, num_spixels = oversegment(img, spixel_args)

	## EXTRACT SUPERPIXELS: ##
	spixels = [create_spixel(i, img, mask, segment_mask, avg_size) for i in range(num_spixels)]

	## FORMAT DATA: ##
	features = [pixel.features for pixel in spixels if pixel is not None]
	labels = [pixel.label for pixel in spixels if pixel is not None]

	features = np.array(features)
	labels = np.array(labels)

	## FREE NOT NEEDED MEMORY: ##
	del(img)
	del(mask)
	del(spixels)

	## CALCULATE ELAPSED TIME: ##
	elapsed_time = time.time() - start_time
	print("Processing image took {0:.1f} seconds".format(elapsed_time))

	## RETURN RESULTS: ##
	return features, labels

def create_spixel(*args):
	try:
		pixel = SuperPixel(*args)
		return pixel
	except ValueError as err:
		# print("Skipping SuperPixel. " + str(err))
		tqdm.write("Skipping SuperPixel. " + str(err))

def compute_avg_size(images, config):
	num_images = len(images)
	num_samples = int(num_images / 50)
	avg_sizes = []

	for i in tqdm(range(num_samples)):
		indx = np.random.randint(0, num_images-1)
		img = cv2.imread(images[indx])
		args = (config["approx_num_superpixels"], config["num_levels"], config["iterations"])
		mask, _ = oversegment(img, args)
		avg_sizes.append(calc_avg_size(mask, 150))

	avg_size = np.mean(np.asarray(avg_sizes), axis=0).astype(np.int)
	avg_size = tuple(avg_size)
	return avg_size

def get_preprocessor(config, features):
	print("Fitting preprocessor...")
	preprocessor = Preprocessor(normalize=config["normalize"],
								reduce_features=config["reduce_features"],
								reducer_type=config["reducer_type"],
								explained_variance=config["explained_variance"])
	preprocessor.train(features)
	return preprocessor

def evaluate(truth, pred):
	print("Evaluating...")
	report = metrics.classification_report(truth, pred)
	acc = metrics.accuracy_score(truth, pred)
	iou = metrics.jaccard_similarity_score(truth, pred)
	precision = metrics.precision_score(truth, pred, average="weighted")
	confusion = metrics.confusion_matrix(truth, pred)
	return report, acc, iou, precision, confusion

def save_results(report, acc, iou, precision, confusion, filepath):
	print("Saving results...")
	results_file = open(filepath, 'w')
	results_file.write(report)
	results_file.write("\nAccuracy: {0:.4f}".format(acc))
	results_file.write("IOU: {0:.4f}".format(iou))
	results_file.write("Precision: {0:.4f}\n".format(precision))
	results_file.write(np.array2string(confusion))
	results_file.close()

####################
## CONFIGURATION: ##
####################

def init():
	global saved_output
	saved_output = []
	saved_output.append(sys.stdout)
	saved_output.append(sys.stderr)

	config = init_config()

	log_file = open(config.save_path["log"], 'w')
	sys.stdout = writer(sys.stdout, log_file)
	# sys.stderr = writer(sys.stderr, log_file)

	return config

def init_config():

	VERSION = 4
	PROCESSORS = 16
	CLASSES = 4

	# hyper parameters:
	hyperparams = dict (
		C = 4,
		kernel = "rbf",
		probability = False,
		class_weight = "balanced",
		cache_size = 1000,
		verbose = False,
	)

	# image files
	image_paths = dict(
		train = "/home/ubuntu/data/mangrove_rgb/training/",
		test = "/home/ubuntu/data/mangrove_rgb/validation/",
	)

	# superpixels
	segment = dict(
		approx_num_superpixels = 300,
		num_levels = 5,
		iterations = 100
	)

	# preprocessor
	preprocessor = dict(
		normalize = True,
		reduce_features = True,
		reducer_type = Reducers.pca,
		explained_variance = 0.95
	)

	# saving
	regenerate_features = True
	save_path = dict(
		log = "results/classification_v{}_log.txt".format(VERSION),
		results = "results/classification_v{}_results.txt".format(VERSION),
		train_features = "saves/train_features.pkl",
		test_features = "saves/test_features.pkl",
	)

	params = Namespace(
		processors = PROCESSORS,
		classes = CLASSES,
		image_paths = image_paths,
		segment = segment,
		preprocess = preprocessor,
		save_path = save_path,
		hyperparams = hyperparams,
		regenerate_features = regenerate_features,
	)

	return params

##########
## RUN: ##
##########
if __name__ == '__main__':
	main()
	sys.stdout = saved_output[0]
