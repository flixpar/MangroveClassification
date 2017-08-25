import time
import numpy as np
from enum import Enum

from sklearn import random_projection, feature_selection, decomposition
from sklearn import preprocessing

class Reducers(Enum):
	none = 0
	random_projection = 1
	feature_selection = 2
	pca = 3

## DEFAULTS:
normalize_default = False
reduce_features_defualt = False
reducer_type_default = Reducers.none
explained_variance_default = 0.95

class Preprocessor:

	def __init__(self,
				normalize=normalize_default,
				reduce_features=reduce_features_defualt,
				reducer_type=reducer_type_default,
				explained_variance=explained_variance_default):

		self.normalize = normalize
		self.reduce_features = reduce_features
		self.reducer_type = reducer_type
		self.explained_variance = explained_variance

		if reduce_features:
			assert(reducer_type != Reducers.none)

		self.normalizer = None
		self.reducer = None

	def train(self, features):

		# Setup:
		start_time = time.time()

		# Check feature set:
		assert(np.isfinite(features).all())
		
		# Normalizer:
		if self.normalize:
			standardizer = preprocessing.StandardScaler()
			features = standardizer.fit_transform(features)
			self.normalizer = standardizer

		# Option 1 (Random Projection):
		if self.reducer_type == Reducers.random_projection:
			transformer = random_projection.GaussianRandomProjection()
			transformer.fit(features)
			self.reducer = transformer

		# Option 2 (Feature Selection):
		if self.reducer_type == Reducers.feature_selection:
			threshold = (self.explained_variance) * (1 - self.explained_variance)
			selector = feature_selection.VarianceThreshold(threshold=threshold)
			selector.fit(features)
			self.reducer = selector

		# Option 3 (PCA):
		if self.reducer_type == Reducers.pca:
			pca = decomposition.PCA(n_components=self.explained_variance, svd_solver="full")
			pca.fit(features)
			self.reducer = pca

		# Calculate elapsed time:
		end_time = time.time()
		elapsed_time = end_time - start_time
		print("Training preprocessor took %.2f seconds" % elapsed_time)

	def process(self, features):

		# Setup:
		start_time = time.time()
		initial_feature_size = features.shape[1]

		# Check feature set:
		# assert(np.isfinite(features).all())
		if not np.isfinite(features).all():
			print("Error. Invalid features.")

		# Check args:
		if not self.normalize and not self.reduce_features:
			print("No preprocessing done.")
			return features
		
		# Standardize:
		if self.normalize:
			features = self.normalizer.transform(features)

		# Reduce features:
		if self.reduce_features:
			features = self.reducer.transform(features)

		# Calculate elapsed time:
		end_time = time.time()
		elapsed_time = end_time - start_time

		print("Preprocessing took %.2f seconds" % elapsed_time)
		print("Reduced feature size from %d to %d" % (initial_feature_size, features.shape[1]))

		return features
