import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib import cm, image
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

from grid2d import Grid2D

from .heatmap import create_colormap

def _get_grid_bounds(Y):
	mins = np.amin(Y, axis=0)
	maxes = np.amax(Y, axis=0)
	return mins[0], mins[1], maxes[0], maxes[1]

def partition_indices(Y, cell_width=10, cell_height=10):
	'''
	Create a grid with cell dimensions (cell_width, cell_height) based on Y,
	partition samples in Y into grid.

	Parameters:
		Y : numpy array, shape (n_samples, 2,)
			t-SNE embedding
		cell_width : int
			width of grid cell, default 10
		cell_height : int
			height of grid cell, default 10

	Return:
		grid : Grid2D
			Grid containing sample partition.
	'''
	xmin, ymin, xmax, ymax = _get_grid_bounds(Y)
	grid = Grid2D(xmin, ymin, xmax, ymax, cell_width, cell_height)

	for i,point in enumerate(Y):
		assert(grid.insert(point[0], point[1], i))

	return grid

def _plot_in_cell(Y, plot_in_cell, idx=None, cell_width=10, cell_height=10, fig_scale=0.5, suptitle=None, savepath=None, show=True, show_counts=False, kwargs={}):
	'''
	plot_in_cell can only take in full dataset/embedding as a keyword argument, so if we want to generate 
	plot on a subset of the data, we need to store the indices for the subset elsewhere. idx is the index 
	into the full dataset that indicates this subset. If idx=None, then we plot without subsetting. 
	'''
	# Get grid partition
	grid = partition_indices(Y, cell_width=cell_width, cell_height=cell_height)

	# Create subplot grid of plots
	shape = grid.shape()
	
	if show_counts:
		counts = np.zeros(shape)

	fig = plt.subplots(figsize=(fig_scale * shape[1], fig_scale * shape[0]), squeeze=False)
	for grid_cell, sample_idxs in grid.items():
		ax = plt.subplot2grid(shape, grid_cell)
		# Create plot for cell
		cell_idxs = idx[sample_idxs] if idx is not None else sample_idxs
		plot_in_cell(ax, cell_idxs, grid_cell, grid, **kwargs)

		if show_counts:
			counts[grid_cell[0], grid_cell[1]] = len(cell_idxs)

	if suptitle is not None:
		plt.suptitle(suptitle)
	if savepath is not None:
		plt.savefig(savepath)
	if show:
		plt.show()
	plt.close()

	if show_counts:
		print(counts)

def plot_in_cells(Y, plot_in_cell, cell_width=10, cell_height=10, fig_scale=0.5, suptitle=None, savepath=None, show=True, show_counts=False, kwargs={}):
	'''
	Description:
		1. Create a grid on Y with cell dims (cell_width, cell_height).
		2. Generate a subplot for each grid cell by calling function [plot_in_cell].
	
	Parameters:
		Y : numpy array, shape (n_samples, map_dims)
			t-SNE embedding
		plot_in_cell : function, signature (ax, cell_idxs, grid_cell, grid, **kwargs)
			ax is axis on which to draw plot. cell_idx is indices in lab of elements in cell.
			grid_cell is index of cell in grid. 
		cell_width : int
			width of grid cell, default 10
		cell_height : int
			height of grid cell, default 10
		suptitle : string or None, default None
			If None, set suptitle of each plot to 'Class {c}'. Otherwise, set suptitle to 
			'{suptitle}, {c}'.
		savedir : string or None, default None
			Save plot for each cluster to savedir if not None. Plot is named for label class.
			Attempts to create directory if it does not exist.
		show : bool, default True
			Show generated plots
		kwargs : dictionary, default {}
			Other arguments needed to call plot_in_cell. These should be the same for all cells and 
			classes.

	Returns: None
	'''
	# Generate gridplot for full embedding
	plot_suptitle = '{}, all'.format(suptitle) if suptitle is not None else None
	_plot_in_cell(Y, plot_in_cell, idx=None,
								cell_width=cell_width, cell_height=cell_height,
								fig_scale=fig_scale, suptitle=plot_suptitle, 
								savepath=savepath, show=show, show_counts=show_counts, kwargs=kwargs)

def _plot_reference_grid(Y, color_by, mappable, cell_width=10, cell_height=10, suptitle=None, savepath=None, show=True):
	'''
	Description:
		Plot reference grid on embedding Y with colors c.
	'''
	_, ax = plt.subplots()
	c = mappable.to_rgba(color_by)

	ax.scatter(Y[:, 0], Y[:, 1], c=c, s=2)
	ax.axis('equal')

	# Create and plot grid
	xmin, ymin, xmax, ymax = _get_grid_bounds(Y)
	grid = Grid2D(xmin, ymin, xmax, ymax, cell_width, cell_height)
	grid.plot_gridlines(ax)

	plt.colorbar(mappable, ax=ax)

	if suptitle is not None:
		plt.suptitle(suptitle)
	if savepath is not None:
		plt.savefig(savepath)
	if show:
		plt.show()
	plt.close()

def plot_reference_grid(Y, lab, color_by, cmap='rainbow', cell_width=10, cell_height=10, suptitle=None, savedir=None, show=True):
	'''
	Plot grid over each cluster. Color cluster by values color_by.
	'''
	if savedir is not None and not os.path.isdir(savedir):
			os.mkdir(savedir)

	classes = np.unique(lab)
	mappable = create_colormap(np.amin(color_by), np.amax(color_by), cmap=cmap)

	# Generate reference for full embedding
	savepath = os.path.join(savedir, 'reference.png') if savedir is not None else None
	_plot_reference_grid(Y, color_by, mappable, 
											 cell_width=cell_width, cell_height=cell_height,
											 suptitle=suptitle, savepath=savepath, 
											 show=show)
