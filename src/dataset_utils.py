#!/usr/bin/env python

import numpy as np
import os
import struct
import random
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors

from constants import MNIST_IMAGES_PATH, MNIST_LABELS_PATH

def loadlocal_mnist(images_path=MNIST_IMAGES_PATH, labels_path=MNIST_LABELS_PATH):
	'''
	Read full MNIST dataset from ubyte files. Works for MNIST, Fashion MNIST, Kuzushiji MNIST datasets.
	Source: https://rasbt.github.io/mlxtend/user_guide/data/loadlocal_mnist/#example-1-part-1-downloading-the-mnist-dataset
	Parameters
	----------
	images_path : str
		path to the test or train MNIST ubyte file
	labels_path : str
		path to the test or train MNIST class labels file
	Returns
	--------
	images : [n_samples, n_pixels] numpy.array
		Pixel values of the images.
	labels : [n_samples] numpy array
		Target class labels
	Examples
	-----------
	For usage examples, please see
	http://rasbt.github.io/mlxtend/user_guide/data/loadlocal_mnist/
	'''
	with open(labels_path, 'rb') as lbpath:
		magic, n = struct.unpack('>II',
								 lbpath.read(8))
		labels = np.fromfile(lbpath,
							 dtype=np.uint8)
	with open(images_path, 'rb') as imgpath:
		magic, num, rows, cols = struct.unpack(">IIII",
											   imgpath.read(16))
		images = np.fromfile(imgpath,
							 dtype=np.uint8).reshape(len(labels), 784)

	return images, labels

def loadlocal_mnist_classes(classes, n=None, seed=None, images_path=MNIST_IMAGES_PATH, labels_path=MNIST_LABELS_PATH):
	'''
	Loads specified label classes from MNIST dataset. If n is not None, uniformly downsample filtered
	training data to n samples. Works for MNIST, Fashion MNIST, Kuzushiji MNIST datasets.
	Parameters
	----------
	digits : int list
		list of class labels to load
	n : int, or None, default None
		Max number of samples to return.
	seed : int or None, default None
		If not None, used to seed uniform downsampler.
	images_path : str
		path to the test or train MNIST ubyte file
	labels_path : str
		path to the test or train MNIST class labels file
	Returns
	--------
	images : [n_samples, n_pixels] numpy.array
		Pixel values of the images.
	labels : [n_samples] numpy array
		Target class labels
	Examples
	-----------
	For usage examples, please see
	http://rasbt.github.io/mlxtend/user_guide/data/loadlocal_mnist/
	'''
	# Load full dataset
	X, lab = loadlocal_mnist(images_path=images_path, labels_path=labels_path)

	if seed is not None:
		random.seed(seed)

	in_classes = np.logical_or.reduce([lab==i for i in classes])
	X_classes = X[in_classes]
	lab_classes = lab[in_classes]

	if n is None:
		return X_classes, lab_classes
	else:
		sample = random.sample(range(X_classes.shape[0]), n)
		X_sample = X_classes[sample,:]
		lab_sample = lab_classes[sample]

		return X_sample, lab_sample

def generate_gaussians(mus, covs, n=500, seed=0):
	'''
	Takes a list of multivariate gaussians parameterized by mean and covariance, samples 
	n points from each.
	Parameters
	----------
	mus : vector list
		Means of Gaussian blobs
	covs : matrix list 
		Covariance matrices of Gaussian blobs
	n : int 
		Number of points to sample from each Gaussian
	seed : int or None
		If not None, use this to seed rng.
	Returns
	--------
	images : [n_samples, n_pixels] numpy.array
		Pixel values of the images.
	labels : [n_samples] numpy array
		Target class labels
	'''
	assert(len(mus) == len(covs))

	if seed is not None:
		random.seed(seed)

	X = [np.random.multivariate_normal(mus[i], covs[i], n) for i in range(len(mus))]
	lab = [i * np.ones(n, dtype=np.int32) for i in range(len(mus))]

	X = np.concatenate(X, axis=0)
	lab = np.concatenate(lab, axis=0)

	return X, lab
