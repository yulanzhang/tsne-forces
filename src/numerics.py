'''
p = k/n 
'''

import numpy as np

from scipy.spatial.distance import pdist, squareform

def compute_energy(Y, graph, edge_wt, p):
	'''
	Calculate energy ( p_{ij} = 0 or 1/(2|E(G)|) = edge_wt )

	energy = sum_{i ~ j} edge_wt * log[ sum_{i != j} (1 + |y_i - y_j|^2)^{-1} ] 
						+ sum_{i ~ j} edge_wt * log[ 1 + |y_i - y_j|^2 ]
	'''
	dist = pdist(Y, metric='sqeuclidean')

	left = 1 / (1 + dist)
	# multiply by 2 because left is derived from condensed 
	# distance matrix. diagonal entries are assigned 0 wt.
	left = np.sum(graph) * np.log(2 * np.sum(left))

	right = np.log(1 + dist)
	# convert symmetric graph to vector form condensed matrix.
	right = 2 * np.sum(squareform(graph) * right)

	energy = edge_wt * (left + right)
	return energy

def compute_expected_energy(Y, graph, edge_wt, p):
	'''
	Calculated expected energy (Jensen term) from derivation.

	expected = sum_{i ~ j} edge_wt * log[ sum_{i != j} (1 + |y_i - y_j|^2)^{-1} ]
							+ sum_{i != j} p * edge_wt * log[ 1 + |y_i - y_j|^2 ]
	'''
	dist = pdist(Y, metric='sqeuclidean')

	left = 1 / (1 + dist)
	# multiply by 2 because left is derived from condensed 
	# distance matrix. diagonal entries are assigned 0 wt.
	left = np.sum(graph) * np.log(2 * np.sum(left))

	right = np.log(1 + dist)
	right = 2 * p * np.sum(right)

	expected = edge_wt * (left + right)
	return expected

def compute_variance(Y, edge_wt, p):
	'''
	Variance of energy from derivation.

	variance = sum_{i != j} p * (1-p) * (edge_wt * log(1 + |y_i - y_j|^2))^2
	'''
	var = pdist(Y, metric='sqeuclidean')
	var = edge_wt * np.log(1 + var)
	# multiply by 2 because left is derived from condensed 
	# distance matrix. diagonal entries are assigned 0 wt.
	var = 2 * p * (1-p) * np.sum(var * var)
	return var

def compute_expected_energy_taylor(Y, edge_wt, p):
	'''
	Taylor approximation of expected energy

	expected = sum_{i != j} p/2 * edge_wt * |y_i - y_j|^4 
							- p/(2n^2) * edge_wt * (sum_{i != j} |y_i - y_j|^2)^2
	'''
	dist = pdist(Y, metric='sqeuclidean')

	# condensed array only contains half entries of distance
	# matrix. Since dist is symmetric, this absorbs the 1/2 
	# factor in original expression.
	left = p * edge_wt * np.sum(dist * dist)

	n = Y.shape[0]
	right = 2 * np.sum(dist)
	right = p * edge_wt * right * right
	right /= 2 * n * n

	expected = left - right
	return expected

def compute_variance_taylor(Y, edge_wt, p):
	'''
	Taylor approximation of variance

	variance = p * (1-p) * edge_wt^2 * sum_{i != j} |y_i - y_j|^4
	'''
	var = pdist(Y, metric='sqeuclidean')
	var = p * (1-p) * edge_wt * edge_wt * np.sum(var * var)
	return var 
