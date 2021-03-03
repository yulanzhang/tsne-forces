'''
This module deals with *.dat files saved and/or used by Fit-SNE.
'''

import numpy as np
import os
import struct

from scipy.sparse import csr_matrix

def cleanup():
	'''
	Remove data.dat, P_col.dat, P_row.dat, P_val.dat, P.dat, result.dat from cwd, if
	they exist.
	'''
	if os.path.isfile('data.dat'):
		os.remove('data.dat')
	if os.path.isfile('P_col.dat'):
		os.remove('P_col.dat')
	if os.path.isfile('P_row.dat'):
		os.remove('P_row.dat')
	if os.path.isfile('P_val.dat'):
		os.remove('P_val.dat')
	if os.path.isfile('P.dat'):
		os.remove('P.dat')
	if os.path.isfile('result.dat'):
		os.remove('result.dat')

def load_P_approx(n_samples, dir_name=None):
	'''
	Load input similarities for approximate t-SNE. Matrix must be saved in csr 
	format with row index, col index, and values saved in files P_row.dat, 
	P_col.dat, and P_val.dat respectively. This is the format used by approx 
	Fit-SNE. 

	Parameters
	----------
	n_samples : int
		Number of samples
	dir_name : string or None
		Directory containing P or None. If None, loads P from current working 
		directory.
	Returns
	-------
	P : csr sparse matrix, shape (n_samples, n_samples)
		Condensed joint probability matrix with only nearest neighbors.
	'''
	if dir_name is None:
		dir_name = os.getcwd()
	assert(os.path.isdir(dir_name))

	with open(os.path.join(dir_name, "P_row.dat"), "rb") as f:
		sz = struct.calcsize("=I")
		buf = f.read(sz * (n_samples + 1))
		row_P = np.array([
			struct.unpack_from("=I", buf, sz * offset)[0] for offset in range(n_samples + 1)
		])
	
	numel = row_P[n_samples]
	with open(os.path.join(dir_name, "P_val.dat"), "rb") as f:
		sz = struct.calcsize("=d")
		buf = f.read(sz * numel)
		val_P = np.array([
			struct.unpack_from("=d", buf, sz * offset)[0] for offset in range(numel)
		])
	
	with open(os.path.join(dir_name, "P_col.dat"), "rb") as f:
		sz = struct.calcsize("=I")
		buf = f.read(sz * numel)
		col_P = np.array([
			struct.unpack_from("=I", buf, sz * offset)[0] for offset in range(numel)
		])

	P = csr_matrix((val_P, col_P, row_P), shape=(n_samples, n_samples))
	return P

def load_P_exact(n_samples, dir_name=None):
	'''
	Load input similarities for exact t-SNE. Matrix must be saved in dense 
	format in file P.dat. This is the format used by exact Fit-SNE.

	Parameters
	----------
	n_samples : int
		Number of samples
	dir_name : string or None
		Directory containing P or None. If None, loads P from current working 
		directory.
	Returns
	-------
	P : numpy array, shape (n_samples, n_samples)
		Condensed joint probability matrix with only nearest neighbors.
	'''
	if dir_name is None:
		dir_name = os.getcwd()
	assert(os.path.isdir(dir_name))

	with open(os.path.join(dir_name, "P.dat"), "rb") as f:
		sz = struct.calcsize("=d")
		buf = f.read(sz * n_samples * n_samples)
		P_list = np.array([
			struct.unpack_from("=d", buf, sz * offset)[0] for offset in range(n_samples * n_samples)
		])

	P = np.array(P_list).reshape((n_samples, n_samples), order='C')
	return P

def save_P_approx(P, dir_name=None):
	'''
	Save input similarities for approximate t-SNE to directory.
	Parameters:
		P : csr_matrix
		dir_name : string or None
		Directory containing P or None. If None, loads P from current working 
		directory.
	Return: void
	'''
	if dir_name is None:
		dir_name = os.getcwd()
	if not os.path.isdir(dir_name):
		os.mkdir(dir_name)

	row_P = P.indptr
	col_P = P.indices
	val_P = P.data

	# write affinities to file
	with open(os.path.join(dir_name, "P_row.dat"), "wb") as f:
		for i in row_P.flatten(order='C'):
			f.write(struct.pack("=i", i))

	with open(os.path.join(dir_name, "P_col.dat"), "wb") as f:
		for i in col_P.flatten(order='C'):
			f.write(struct.pack("=i", i))

	with open(os.path.join(dir_name, "P_val.dat"), "wb") as f:
		for d in val_P.flatten(order='C'):
			f.write(struct.pack("=d", d))

def save_P_exact(P, dir_name=None):
	'''
	Save input similarities for exact t-SNE to directory.
	Parameters:
		P : numpy array
		dir_name : string or None
		Directory containing P or None. If None, loads P from current working 
		directory.
	'''
	if dir_name is None:
		dir_name = os.getcwd()
	if not os.path.isdir(dir_name):
		os.mkdir(dir_name)

	with open(os.path.join(dir_name, "P.dat"), "wb") as f:
			for d in P.flatten(order='C'):
				f.write(struct.pack("=d", d))
