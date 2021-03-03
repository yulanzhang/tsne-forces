import numpy as np
import os
import scipy.sparse as sp
import struct
import sys

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import euclidean_distances

from constants import ROOT_DIR, MACHINE_EPSILON
from math_utils import normalize
from tsne_utils import save_P_approx, save_P_exact

sys.path.append(os.path.join(ROOT_DIR, '../packages/FIt-SNE'))
from fast_tsne import fast_tsne

def P_from_graph(X, adjacency_matrix, sigma):
	# Calculate P matrix based on adjacencies. Use a fixed
	# sigma for all adjacency calculations.
	pw_distances = euclidean_distances(X, squared=True)
	affinities = csr_matrix(np.multiply(pw_distances, adjacency_matrix))
	denom = 2 * sigma * sigma
	cond_P = -1 * affinities / denom
	np.exp(cond_P.data, out=cond_P.data)
	cond_P = normalize(cond_P)

	# symmetrize joint probability distribution
	P = cond_P + cond_P.T
	P.sort_indices()

	# Normalize the joint probability distribution
	sum_P = np.maximum(P.sum(), MACHINE_EPSILON)
	P /= sum_P
	return P

def graph_tsne(X, adjacency_matrix, save_affinities=None, kwargs={}):
	'''
	Description:
		Calculate t-SNE embedding using input similarities P calculated based on a
		given neighbor graph. Graph is specified in dense adjacency matrix form. 
		Function accepts fixed sigma only (no perplexity-based tuning).

		This function is a wrapper to Fit-SNE. We save P to file and call fast_tsne 
		with load_affinities="load".

	Parameters:
		X : numpy array, shape (n_samples, n_components)
			Input points
		adjacency_matrix : csr matrix, shape (n_samples, n_samples)
			Sparse matrix representation of input adjacency graph. Entries indicate 
			edge weights.
		save_affinities : string or None
			Dir to save input affinities matrix P or None. If directory 
			save_affinities does not exists, creates it. 
		kwargs : dictionary
			Keyword args for fast_tsne. Do not set load_affinities.

	Return:
		Y : numpy array, shape (n_samples, map_dims)
			embedding
		loss : numpy array
			only returned if return_loss is true
	'''
	assert(adjacency_matrix.shape[0] == X.shape[0])
	assert(np.sum(np.absolute(adjacency_matrix - adjacency_matrix.T)) < 1e-15) 
	
	# TEMP: we set input affinity by normalizing the adjacency matrix instead
	# in order to simplify P. This brings p_{ij} in line with the values used 
	# in the expected energy derivation.
	# P = P_from_graph(X, adjacency_matrix, sigma)

	# Set P to be the normalized adjacency matrix.
	P_sum = adjacency_matrix.sum()
	P = adjacency_matrix / P_sum
	
	if 'theta' in kwargs and kwargs['theta'] == 0:
		P = P.toarray()
		# Save P to current working directory. Fit-SNE will load this.
		save_P_exact(P)
		# Save another copy to user specified directory, if given
		if save_affinities is not None:
			save_P_exact(P, dir_name=save_affinities)
	else:
		# Save P to current working directory. Fit-SNE will load this.
		save_P_approx(P)
		# Save another copy to user specified directory, if given
		if save_affinities is not None:
			save_P_approx(P, dir_name=save_affinities)

	
	# K, perplexity params not used when load_affinities="load"
	result = fast_tsne(X, load_affinities="load", **kwargs)

	return result

