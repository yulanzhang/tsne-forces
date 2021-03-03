import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import NearestNeighbors

from .gridplot import plot_in_cells

def _cell_mean_image(ax, cell_idxs, grid_idx, grid, X=None, image_shape=None):
	'''
	Helper for plot_mean_images. Passed to plot_in_cells.
	Parameters:
		ax : Axes
		cell_idxs : int list
		X : numpy array, shape (n_samples, n_components)
			Input dataset. Each row represents an image.
		image_shape : int tuple
			Pixel dimension of image. Length two.
	'''
	# Calculate mean image for grid cell
	n_images = len(cell_idxs)
	mean_image = np.sum(X[cell_idxs, :] / n_images, axis=0).reshape(image_shape)

	# Plot image
	ax.imshow(mean_image, cmap='gray')
	ax.set_xticks([])
	ax.set_yticks([])

def plot_mean_images(X, Y, image_shape, cell_width=10, cell_height=10, fig_scale=0.5, suptitle=None, savepath=None, show_counts=False):
	'''
	Description:
		1. Create a grid on Y with cell dims (cell_width, cell_height).
		2. Plot mean image for each grid cell.
	Parameters:
		image_shape : int tuple
			Pixel dimension of image. Length two.
	'''
	kwargs = {'X':X, 'image_shape':image_shape}
	plot_in_cells(Y, _cell_mean_image, cell_width=cell_width, cell_height=cell_height, fig_scale=fig_scale, suptitle=suptitle, savepath=savepath, show_counts=show_counts, kwargs=kwargs)
