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

import itertools as it
import multiprocessing as mp

from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn import metrics

from SuperPixel import SuperPixel
from Preprocessor import Preprocessor, Reducers
from labeling_utils import *

def main():

	config = init()
	print("Mangrove Classification")
	print("="*50)
	# print("XGBoost Classifier with: ", config.hyperparams)
	print("Gaussian Naive Bayes Classifier")
	print("Segmenting with: ", config.segment)

	if config.regenerate_features:

		print("Getting data lists...")
		(train_img_list, train_lbl_list), (test_img_list, test_lbl_list) = get_image_paths(config.image_paths)

		print("Generating features...")
		train_features, train_labels, avg_size = extract_features(train_img_list, train_lbl_list, config, mode="train")
		test_features, test_labels = extract_features(test_img_list, test_lbl_list, config, mode="test", avg_size=avg_size)

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

	# Setup data:
	# xg_train = xgb.DMatrix(train_features, label=train_labels)
	# xg_test = xgb.DMatrix(test_features)

	print("Training...")
	start_time = time.time()
	# config.hyperparams["num_class"] = len(train_features)
	# classifier = xgb.train(config.hyperparams, xg_train, num_boost_round=config.rounds, evals=[(xg_test, 'eval')], verbose_eval=True)
	classifier = GaussianNB()
	elapsed_time = time.time() - start_time
	print("Training took {0:.2f} seconds".format(elapsed_time))

	print("Predicting...")
	start_time = time.time()
	pred = classifier.predict(xg_test)
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

	# train_images = [cv2.imread(fn) for fn in train_images_names]
	# train_annotations = [cv2.imread(fn, 0) for fn in train_annotation_names]

	test_images_names = sorted(glob.glob(paths["test"] + "/images/*.tif"))
	test_annotation_names = sorted(glob.glob(paths["test"] + "/annotations/*.png"))

	# test_images = [cv2.imread(fn) for fn in test_images_names]
	# test_annotations = [cv2.imread(fn, 0) for fn in test_annotation_names]

	return (train_images_names, train_annotation_names), (test_images_names, test_annotation_names)

def extract_features(image_paths, mask_paths, config, mode="train", avg_size=None):

	threadpool = mp.Pool(config.processors)

	args = zip(image_paths, mask_paths, it.repeat(config.segment), it.repeat(mode), it.repeat(avg_size))
	# args = tqdm(args)

	if mode == "train":
		features_list, labels_list, avg_sizes = threadpool.starmap(get_features, args)
		avg_size = np.mean(np.array(avg_sizes), axis=0)
	else:
		features_list, labels_list = threadpool.starmap(get_features, args)

	features = np.concatenate(features_list)
	labels = np.concatenate(labels_list)

	if mode == "train":
		return features, labels, avg_size
	else:
		return features, labels

def get_features(img_path, mask_path, config, mode="train", avg_size=None):

	## READ IMAGES: ##
	img = cv2.imread(img_path)
	mask = cv2.imread(mask_path, 0)

	## OVERSEGMENT: ##
	spixel_args = (config["approx_num_superpixels"], config["num_levels"], config["iterations"])
	segment_mask, num_spixels = oversegment(img, spixel_args)

	if mode == "train":
		avg_size = calc_avg_size(segment_mask, int(num_spixels/50))

	## EXTRACT SUPERPIXELS: ##
	spixels = [create_spixel(i, img, mask, segment_mask, avg_size) for i in range(num_spixels)]

	## FORMAT DATA: ##
	features = [pixel.features for pixel in spixels if pixel is not None]
	labels = [pixel.label for pixel in spixels if pixel is not None]

	features = np.array(features)
	labels = np.array(labels)

	## RETURN RESULTS: ##
	if mode == "train":
		return features, labels, avg_size
	else:
		return features, labels

def create_spixel(*args):
	try:
		pixel = SuperPixel(*args)
		return pixel
	except ValueError as err:
		print("Skipping SuperPixel. " + str(err))
		# tqdm.write("Skipping SuperPixel. " + str(err))

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
	global saved_stdout
	saved_stdout = sys.stdout

	config = init_config()

	log_file = open(config.save_path["log"], 'w')
	sys.stdout = writer(saved_stdout, log_file)

	return config

def init_config():

	VERSION = 1
	PROCESSORS = 8
	CLASSES = 4

	# hyper parameters:
	rounds = 5
	hyperparams = dict(
		max_depth = 3,
		eta = 0.2,
		silent = 1,
		objective = "multi:softmax",
		nthread = PROCESSORS,
	)

	# image files
	image_paths = dict(
		train = "/home/ml/felix_ws/data/mangrove_rgb/training/",
		test = "/home/ml/felix_ws/data/mangrove_rgb/validation/",
	)

	# superpixels
	segment = dict(
		approx_num_superpixels = 5000,
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
		rounds = rounds,
		regenerate_features = regenerate_features,
	)

	return params

##########
## RUN: ##
##########
if __name__ == '__main__':
	main()
