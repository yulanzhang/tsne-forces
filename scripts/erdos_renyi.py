#!/usr/bin/env python3

'''
This script runs t-SNE on an Erdos-Renyi graph for various settings of (n,p).
We calculate t-SNE energy and expected energy values from the derivation, along
with their Taylor approximations.
'''

import gc
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import pandas as pd
import random
import sys

from networkx import erdos_renyi_graph, write_adjlist
from networkx.convert_matrix import to_scipy_sparse_matrix

from scipy.sparse import csr_matrix, save_npz
from scipy.spatial.distance import pdist, squareform

SRC_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../src')
sys.path.append(SRC_DIR)

from dataset_utils import generate_gaussians
from graph_tsne import graph_tsne
from tsne_utils import cleanup

from trials_utils import *

# Where to save results
EXPERIMENT_DIR = '/tmp/erdos_renyi'

def _run_trial(X, params, trial_dir, verbose=0, kwargs={}):
	'''
	Generate Erdos-Renyi(n, p) on a single Gaussian cluster of n samples. Save 
	graph, input affinities, and embedding to trial_dir.
	'''
	n, p = params

	if verbose > 0:
		sys.stderr.write("generating graph\n")

	graph = erdos_renyi_graph(n, p)
	# Write graph to file as adjacency list
	write_adjlist(graph, os.path.join(trial_dir, GRAPH_FILE))
	# Convert to csr matrix for graph_tsne
	graph = to_scipy_sparse_matrix(graph)

	if verbose > 0:
		sys.stderr.write("running t-sne\n")

	# Run t-SNE on graph, saving affinities to trial_dir
	Y = graph_tsne(X, graph, save_affinities=trial_dir, kwargs=kwargs)
	np.save(os.path.join(trial_dir, EMBEDDING_FILE), Y)

def run_experiment(params_list, n_trials, experiment_dir, verbose=0):
	if not os.path.isdir(experiment_dir):
		if verbose > 0:
			print('creating experiment dir')
		os.mkdir(experiment_dir)
	if not is_initialized(experiment_dir):
		if verbose > 0:
			print('initializing experiment.')
		initialize_experiment(['n', 'p'], experiment_dir)

	params_ids = register_parameters(params_list, experiment_dir)

	# Gaussian cluster params
	mus = [[0,0]]
	covs = [np.eye(2)]

	for i, params_id in enumerate(params_ids):
		if verbose > 0:
			print('Running trials for params: {}'.format(params_list[i]))

		params_dir = os.path.join(experiment_dir, '{}'.format(params_id))

		# Generate and save single Gaussian cluster dataset. 
		# The values are only used for pca initialization. If we used random initialization, we 
		# would still need to pass in a dataset with n samples in order to run Fit-SNE.
		X, _ = generate_gaussians(mus, covs, n=params_list[i][0])

		kwargs = {'early_exag_coeff':12.0, 'max_iter':1000}
		run_trials(X, params_list[i], n_trials, _run_trial, params_dir, verbose=verbose, kwargs=kwargs)

if __name__ == "__main__":
	ns = [10000, 20000, 30000, 40000]
	ps = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
	n_trials = 10

	# params_list = [(ns[i], ps[j]) for i in range(len(ns)) for j in range(len(ps))]
	# run_experiment(params_list, n_trials, EXPERIMENT_DIR, verbose=1)
	compute_experiment_results(EXPERIMENT_DIR, verbose=1)

	cleanup()

