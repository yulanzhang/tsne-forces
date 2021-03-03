'''
General utility functions for plotting heatmaps.
'''

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from math_utils import vtheta

def create_colormap(vmin, vmax, cmap='rainbow'):
	'''
	Description:
		Create a continuous colormap and ScalarMappable object. Sets array for
		ScalarMappable to [].

	Parameters:
		vmin : float
			Minimum colormap value. Used for normalization.
		vmax : float
			Maximum colormap value. Used for normalization.
		cmap : string, default "rainbow"
			String for matplotlib colormap. Available maps listed here:
			https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html

	Return:
		mappable : ScalarMappable
			ScalarMappable for colormap normalized to [vmin, vmax].
	'''
	color_map = cm.get_cmap(cmap)
	mappable = cm.ScalarMappable(norm=Normalize(vmin, vmax), cmap=color_map)
	mappable.set_array([])

	return mappable

def create_discrete_colormap(vmin, vmax, cmap='rainbow'):
	'''
	Description:
		Create a discrete colormap and ScalarMappable object. ColorMap lookup 
		table contains (vmax - vmin + 1) entries. Sets array for ScalarMappable 
		to [].

	Parameters:
		vmin : int
			Minimum colormap value. Used for normalization.
		vmax : int
			Maximum colormap value. Used for normalization.
		cmap : string, default "rainbow"
			String for matplotlib colormap. Available maps listed here:
			https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html

	Return:
		mappable : ScalarMappable
			ScalarMappable for colormap normalized to [vmin, vmax].
	'''
	color_map = cm.get_cmap(cmap, vmax - vmin + 1)
	mappable = cm.ScalarMappable(norm=Normalize(vmin, vmax), cmap=color_map)
	mappable.set_array([])

	return mappable

def plot_heatmap_on_ax(points, lab, title, ax, s=2, cmap="rainbow", mappable=None, colorbar=True, colorbar_fontsize=20, colorbar_kwargs={}):
	'''
	Description:
		Plot a point set (sets) on a specified axis (axes), color according to
		labels. Option to add colobar.

	Parameters:
		points : numpy array, shape (n_samples, 2,) or array list
			Point set or list of point sets to color. If multiple point sets provided,
			we assume that all are indexed by sample
		lab : numpy array, shape (n_samples,)
			Sample labels to use for coloring.
		title : string or string list
			Label for point set or labels for each point set.
		ax : axes or axes array
			Axes to plot heatmap on.
		s : int
			Point size
		cmap : string, default "rainbow"
			String for matplotlib colormap. Used to generate new colormap scaled by lab
			if mappable is None. Available maps listed here:
			https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
		mappable : ScalarMappable or None, default None
			Scaled colormap for heatplot
		colorbar : bool, default True
			Add colorbar to axis (axes).
		colorbar_fontsize : int, default 20
			Font size for colorbar labels.
		colorbar_kwargs : dict, default {}
			Keyword args for generating colorbar. 

	Returns: void
	'''
	# Create colormap based on lab
	if mappable is None:
		mappable = create_colormap(np.amin(lab), np.amax(lab), cmap=cmap)
	ncols = len(points) if isinstance(points, list) else 1
	c = mappable.to_rgba(lab)
	if ncols == 1:
		ax.scatter(points[:,0], points[:,1], c=c, s=s)
		ax.set_title(title)
		ax.axis('equal')
		if colorbar:
			cbar = plt.colorbar(mappable, ax=ax, **colorbar_kwargs)
			for t in cbar.ax.get_yticklabels():
				t.set_fontsize(colorbar_fontsize)
	else:
		assert(isinstance(title, list) and isinstance(ax, np.ndarray))
		assert(len(points) == len(title) and len(points) == ax.size)
		
		for i in range(ncols):
			ax[i].scatter(points[i][:,0], points[i][:,1], c=c, s=s)
			ax[i].set_title(title[i])
			ax[i].axis('equal')
		if colorbar:
			cbar = plt.colorbar(mappable, ax=ax[-1], **colorbar_kwargs)
			for t in cbar.ax.get_yticklabels():
				t.set_fontsize(colorbar_fontsize)

def plot_heatmap(points, lab, title, s=2, cmap="rainbow", colorbar=True, suptitle=None, savepath=None, show=True):
	'''
	Description:
		Plot a 2d point set (sets) and color according to labels. 

	Parameters:
		points : numpy array, shape (n_samples, 2,) or array list
			Point set or list of point sets to color. If multiple point sets provided,
			we assume that all are indexed by sample
		lab : numpy array, shape (n_samples,)
			Sample labels to use for coloring.
		title : string or string list
			Label for point set or labels for each point set.
		cmap : string, default "rainbow"
			String for matplotlib colormap. Available maps listed here:
			https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
		colorbar : bool, default True
			Add colorbar to axis (axes).
		suptitle : string or None
			Figure title or None.
		savepath : string or None
			Saves figure to savepath if not None.
		show : bool, default True
			Show heatmap plot
	Return: void
	'''
	ncols = len(points) if isinstance(points, list) else 1
	fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 5, 5))
	plot_heatmap_on_ax(points, lab, title, ax, s=s, cmap=cmap, colorbar=colorbar)
	if suptitle is not None:
		fig.suptitle(suptitle)
	if savepath is not None:
		plt.savefig(savepath)

	if show:
		plt.show()
	plt.close()

def plot_discrete_heatmap_on_ax(points, lab, title, ax, s=2, cmap="rainbow"):
	'''
	Description:
		[plot_heatmap_on_ax] for categorical labels. Option to add legend. 
	'''
	classes = np.unique(lab)
	mappable = create_discrete_colormap(np.amin(classes), np.amax(classes), cmap=cmap)
	legend_elements = [Line2D([0],[0], marker='o', color='w', label='{}'.format(c), 
														markerfacecolor=mappable.to_rgba(c), markersize=5) 
										 for c in classes]

	ncols = len(points) if isinstance(points, list) else 1
	c = mappable.to_rgba(lab)
	if ncols == 1:
		ax.scatter(points[:,0], points[:,1], c=c, s=s)
		ax.set_title(title)
		ax.axis('equal')
		ax.legend(handles=legend_elements)
	else:
		assert(isinstance(title, list) and isinstance(ax, np.ndarray))
		assert(len(points) == len(title) and len(points) == ax.size)
		
		for i in range(ncols):
			ax[i].scatter(points[i][:,0], points[i][:,1], c=c, s=s)
			ax[i].set_title(title[i])
			ax[i].axis('equal')

		ax[-1].legend(handles=legend_elements)

def plot_discrete_heatmap(points, lab, title, s=2, cmap="rainbow", show=True, suptitle=None, savepath=None):
	'''
	[plot_heatmap] for categorical labels.
	'''
	ncols = len(points) if isinstance(points, list) else 1
	fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 5, 5))
	plot_discrete_heatmap_on_ax(points, lab, title, ax, s=s, cmap=cmap)
	if suptitle is not None:
		fig.suptitle(suptitle)
	if savepath is not None:
		plt.savefig(savepath)
	if show:
		plt.show()
	plt.close()
