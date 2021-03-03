'''
This module contains functions for calculating t-SNE gradient forces. Call these after 
running Fit-SNE with [load_affinities="save"].
'''
import numpy as np

from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from constants import MACHINE_EPSILON
from math_utils import vtheta

from .file import load_P_exact, load_P_approx

def compute_QZ(Y, degrees_of_freedom=1):
	'''
	Get condensed matrix of unnormalized output similarities. Diagonal entries 0.
	Parameters
	----------
	Y : array, shape (n_samples * n_components,)
		Embedding is stored as a dense matrix.
	degrees_of_freedom : int
		Degrees of freedom of t-distribution.
	Returns
	-------
	Q : array, shape (n_samples * (n_samples-1) / 2,)
		Condensed unnormalized output similarities.
	'''
	dist = pdist(Y, "sqeuclidean")
	dist /= degrees_of_freedom
	dist += 1.
	dist **= (degrees_of_freedom + 1.0) / -2.0
	return dist

def compute_Q(Y, degrees_of_freedom=1):
	'''
	Compute joint probabilities q_ij from embedding distances.
	Parameters
	----------
	Y : array, shape (n_samples * n_components,)
		Embedding is stored as a dense matrix.
	degrees_of_freedom : int
		Degrees of freedom of t-distribution.
	Returns
	-------
	Q : array, shape (n_samples * (n_samples-1) / 2,)
		Condensed joint probability matrix.
	'''
	dist = compute_QZ(Y, degrees_of_freedom=degrees_of_freedom)
	Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)
	return Q

def compute_repulsion_forces(Y, degrees_of_freedom=1):
	# Get Q in condensed form.
	Q = compute_Q(Y)

	# Get condensed matrix of unnormalized output similarities
	QZ = compute_QZ(Y, degrees_of_freedom=degrees_of_freedom)

	# Get squareform matrix of repulsion coefficients q_{ij}^2 * Z
	QQZ = squareform(Q * QZ)

	# Compute repulsion force vectors for each data point
	n_samples = Y.shape[0]
	repulsion_forces = [np.dot(QQZ[i], Y[i] - Y) for i in range(n_samples)]
	return repulsion_forces

def compute_repulsion_magnitudes(Y, degrees_of_freedom=1):
	'''
	Calculates magnitude of t-SNE repulsion forces on each point in embedding.

	Parameters
	----------
	Y : array, shape (n_samples, n_components)
		Embedding is stored as a dense matrix.
	degrees_of_freedom : int, default 1
		Degrees of freedom of t-distribution

	Returns
	-------
	repulsion_magnitudes : numpy array, shape (n_samples,)
		Vector of repulsion force magnitudes for embedded point. Indexed
		according to Y.
	'''
	repulsion_forces = compute_repulsion_forces(Y, degrees_of_freedom=degrees_of_freedom)
	repulsion_magnitudes = np.linalg.norm(repulsion_forces, ord=2, axis=1)

	return repulsion_magnitudes

def compute_repulsion_directions(Y, degrees_of_freedom=1):
	repulsion_forces = compute_repulsion_forces(Y, degrees_of_freedom=degrees_of_freedom)
	normalized_forces = normalize(repulsion_forces)
	thetas = vtheta(normalized_forces[:,0], normalized_forces[:,1])
	return thetas

def compute_attraction_forces(Y, degrees_of_freedom=1, load_exact=False):
	n_samples = Y.shape[0]

	# Load P from current working directory, exaggerate.
	if load_exact:
		P = load_P_exact(n_samples)
	else:
		P = load_P_approx(n_samples).toarray()

	# Get condensed matrix of unnormalized output similarities
	QZ = compute_QZ(Y, degrees_of_freedom=degrees_of_freedom)

	# Get squareform matrix of attraction coefficients p_{ij} * q_{ij} * Z
	PQZ = P * squareform(QZ)

	# Compute attraction force vectors for each data point
	attraction_forces = [np.dot(PQZ[i], Y - Y[i]) for i in range(n_samples)]
	return attraction_forces

def compute_attraction_magnitudes(Y, degrees_of_freedom=1, load_exact=False):
	'''
	Calculates magnitude of t-SNE attraction forces on each point in embedding.
	Assumes P is saved in current working directory.

	Parameters
	----------
	Y : array, shape (n_samples, n_components)
		Embedding is stored as a dense matrix.
	degrees_of_freedom : int, default 1
		Degrees of freedom of t-distribution used to calculate Q.
	load_exact : bool, default False
		True if saved P was calculated for exact t-SNE, i.e. P is a dense matrix. 
		False if P was calculated for approximate t-SNE, i.e. P is a sparse matrix.
	Returns
	-------
	attraction_magnitudes : numpy array, shape (n_samples,)
		Vector of attraction force magnitudes for embedded point. Indexed
		according to Y.
	'''
	attraction_forces = compute_attraction_forces(Y, degrees_of_freedom=degrees_of_freedom, load_exact=load_exact)
	attraction_magnitudes = np.linalg.norm(attraction_forces, ord=2, axis=1)

	return attraction_magnitudes

def compute_attraction_directions(Y, degrees_of_freedom=1, load_exact=False):
	attraction_forces = compute_attraction_forces(Y, degrees_of_freedom=degrees_of_freedom, load_exact=load_exact)
	normalized_forces = normalize(attraction_forces)
	thetas = vtheta(normalized_forces[:,0], normalized_forces[:,1])
	return thetas
