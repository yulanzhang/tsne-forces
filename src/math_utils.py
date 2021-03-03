import numpy as np

def normalize(W):
	'''
	Normalize matrix rows to probability distribution.
	Code taken from https://stackoverflow.com/a/59365444.
	'''
	#Find the row scalars as a Matrix_(n,1)
	rowSumW = sp.csr_matrix(W.sum(axis=1))
	rowSumW.data = 1/np.maximum(rowSumW.data, MACHINE_EPSILON)

	#Find the diagonal matrix to scale the rows
	rowSumW = rowSumW.transpose()
	scaling_matrix = sp.diags(rowSumW.toarray()[0])

	return scaling_matrix.dot(W)

def pca(X, n_dims=50):
	'''
	Description:
		Use PCA to reduce dimensionality of X to n_dims.
	Parameters:
		X : numpy array, shape (n_samples, n_features)
			Input dataset
		n_dims : int, default 50
			Number of dimensions to reduce to.
	Returns:
		numpy array, shape (n_samples, n_dims)
	'''
	assert(0 < n_dims and n_dims < X.shape[1])

	X_centered = X - X.mean(axis=0)
	U, s, V = np.linalg.svd(X_centered, full_matrices=False)
	return np.dot(U, np.diag(s))[:,:n_dims]

def theta(x, y):
	'''
	Description:
		Map cartesian coordinates (x,y) to polar angle in [0, 2 * pi).
	Parameters:
		x : float
			x coordinate
		y : float
			y coordinate
	Return: 
		theta : float
			Polar angle of (x, y)
	'''
	if x == 0.0:
	 	theta = np.pi / 2 if y > 0 else 3 * np.pi / 2
	else:
		theta = np.arctan(y / x)
		# Check if (x,y) is in quadrant 2 or 3
		if x < 0:
			theta += np.pi
		# Check if (x,y) is in quadrant 4
		if x > 0 and y < 0:
			theta = 2 * np.pi + theta
	return theta

# vectorized theta
vtheta = np.vectorize(theta)
