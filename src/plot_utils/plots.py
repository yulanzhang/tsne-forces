import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from sklearn.preprocessing import normalize

from tsne_utils import compute_attraction_directions, compute_attraction_forces, compute_attraction_magnitudes, compute_repulsion_forces

from .heatmap import create_colormap, plot_discrete_heatmap_on_ax, plot_heatmap_on_ax

'''
Many of the functions in this file take the same parameters. We describe some 
common ones here.

Parameters
----------
Y : numpy array, shape (n_samples, 2,) or array list
	Point set or list of point sets to color. If multiple point sets provided,
	we assume that all are indexed by sample
s : int
	Point size
cmap : string
	String for matplotlib colormap. Available maps listed here:
	https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
fig_len : int
fig_width : int
ax_off : bool
	If true, plot embedding without axes visible.
show : bool
	If true, show plot
suptitle : string or None
	Figure title or None
savepath : string or None
	Saves figure to path if not None. 
'''

def plot_blind(Y, s=2, fig_len=5, fig_width=5, ax_off=False, show=True, suptitle=None, savepath=None):
	'''
	Plots unlabeled embedding.

	Parameters
	----------
		Y : numpy array, shape (n_samples, 2,) or array list
			Point set or list of point sets to color. If multiple point sets provided,
			we assume that all are indexed by sample

	Returns: void
	'''
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_len, fig_width))
	ax.scatter(Y[:,0], Y[:,1], c='k', s=s)
	ax.set_xlim(left=np.amin(Y[:,0]), right=np.amax(Y[:,0]))
	ax.set_ylim(bottom=np.amin(Y[:,1]), top=np.amax(Y[:,1]))
	ax.axis('equal')
	ax.axis('off')

	if suptitle is not None:
		fig.suptitle(suptitle)
	if savepath is not None:
		plt.savefig(savepath, bbox_inches='tight')
	if show:
		plt.show()
	plt.close()

def plot_ground_truth(Y, lab, s=2, cmap='tab10', fig_len=5, fig_width=5, ax_off=False, show=True, suptitle=None, savepath=None):
	'''
	Plots embedding colored by ground truth labels.

	Parameters
	----------
	Y : numpy array, shape (n_samples, 2,) or array list
		Point set or list of point sets to color. If multiple point sets provided,
		we assume that all are indexed by sample
	lab : numpy array, shape (n_samples,)
		Categorical ground truth labels.

	Returns: void
	'''
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_len, fig_width))
	plot_discrete_heatmap_on_ax(Y, lab, "", ax, s=s, cmap=cmap)
	ax.set_xlim(left=np.amin(Y[:,0]), right=np.amax(Y[:,0]))
	ax.set_ylim(bottom=np.amin(Y[:,1]), top=np.amax(Y[:,1]))
	ax.axis('equal')
	if ax_off:
		ax.axis('off')

	if suptitle is not None:
		fig.suptitle(suptitle)
	if savepath is not None:
		plt.savefig(savepath, bbox_inches='tight')
	if show:
		plt.show()
	plt.close()

def plot_attraction_magnitude(Y, degrees_of_freedom=1, load_exact=False, s=2, fig_len=5, fig_width=5, ax_off=False, show=True, suptitle=None, savepath=None, colorbar_fontsize=10, colorbar_kwargs={}):
	'''
	Plots embedding colored by attraction magnitude. This function assumes that
	P is saved in the current working directory.

	Parameters
	----------
	Y : numpy array, shape (n_samples, 2,) or array list
		Point set or list of point sets to color. If multiple point sets provided,
		we assume that all are indexed by sample
	degrees_of_freedom : int, default 1
		Degrees of freedom of t-distribution used to calculate Q.
	load_exact : bool, default False
		True if saved P was calculated for exact t-SNE, i.e. P is a dense matrix. 
		False if P was calculated for approximate t-SNE, i.e. P is a sparse matrix.
	colorbar_fontsize : int, default 20
		Font size for colorbar labels.
	colorbar_kwargs : dict, default {}
		Keyword args for generating colorbar.

	Returns: void
	'''

	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_len, fig_width))
	attraction_magnitudes = compute_attraction_magnitudes(Y, degrees_of_freedom=degrees_of_freedom)
	plot_heatmap_on_ax(Y, attraction_magnitudes, "", ax, s=s, colorbar_fontsize=colorbar_fontsize, colorbar_kwargs=colorbar_kwargs)
	
	ax.set_xlim(left=np.amin(Y[:,0]), right=np.amax(Y[:,0]))
	ax.set_ylim(bottom=np.amin(Y[:,1]), top=np.amax(Y[:,1]))
	ax.axis('equal')
	if ax_off:
		ax.axis('off')

	if suptitle is not None:
		fig.suptitle(suptitle)
	if savepath is not None:
		plt.savefig(savepath, bbox_inches='tight')
	if show:
		plt.show()
	plt.close()

def plot_attraction_directions(Y, idx=None, degrees_of_freedom=1, load_exact=False, s=2, cmap='hsv', fig_len=5, fig_width=5, ax_off=False, show=True, suptitle=None, savepath=None, colorbar_kwargs={}):
	'''
	Plots embedding colored by attraction direction. This function assumes that
	P is saved in the current working directory.

	Parameters
	----------
	Y : numpy array, shape (n_samples, 2,) or array list
		Point set or list of point sets to color. If multiple point sets provided,
		we assume that all are indexed by sample
	idx : int list or None
		If not None, only plot samples in idx. This could be useful for examining 
		forces on subsets of Y. 
	degrees_of_freedom : int, default 1
		Degrees of freedom of t-distribution used to calculate Q.
	load_exact : bool, default False
		True if saved P was calculated for exact t-SNE, i.e. P is a dense matrix. 
		False if P was calculated for approximate t-SNE, i.e. P is a sparse matrix.
	colorbar_fontsize : int, default 20
		Font size for colorbar labels.
	colorbar_kwargs : dict, default {}
		Keyword args for generating colorbar.
	
	Returns: void
	'''

	color_map = cm.get_cmap(cmap)
	mappable = create_colormap(0, 2 * np.pi, cmap=cmap)

	plt.figure(figsize=(fig_len, fig_width))
	ax = plt.subplot(111)
	attraction_directions = compute_attraction_directions(Y, degrees_of_freedom=degrees_of_freedom, load_exact=load_exact)
	
	if idx is None:
		plot_heatmap_on_ax(Y, attraction_directions, "", ax, s=s, mappable=mappable, colorbar=False, colorbar_kwargs=colorbar_kwargs)
		ax.set_xlim(left=np.amin(Y[:,0]), right=np.amax(Y[:,0]))
		ax.set_ylim(bottom=np.amin(Y[:,1]), top=np.amax(Y[:,1]))
	else:
		plot_heatmap_on_ax(Y[idx,:], attraction_directions[idx], "", ax, s=s, mappable=mappable, colorbar=False, colorbar_kwargs=colorbar_kwargs)
		ax.set_xlim(left=np.amin(Y[idx,0]), right=np.amax(Y[idx,0]))
		ax.set_ylim(bottom=np.amin(Y[idx,1]), top=np.amax(Y[idx,1]))
	ax.axis('equal')
	if ax_off:
		ax.axis('off')
	
	if suptitle is not None:
		fig.suptitle(suptitle)
	if savepath is not None:
		plt.savefig(savepath, bbox_inches='tight')
	if show:
		plt.show()
	plt.close()

def plot_polar_colormap(cmap='hsv', fig_scale=5, show=True, suptitle=None, savepath=None):
	'''
	Plots polar colormap.

	Returns: void
	'''
	fig, ax = plt.subplots(figsize=(fig_scale, fig_scale))
	ax = plt.subplot(111, projection='polar')
	
	color_map = cm.get_cmap(cmap)
	cb = ColorbarBase(ax, cmap=color_map, norm=Normalize(0.0, 2*np.pi), orientation='horizontal')
	cb.outline.set_visible(False)
	
	ax.set_rlim([-1.5,1.0])
	
	thetas = np.linspace(0, 1.75*np.pi, 8)
	r = np.array([0.0,1.0])
	for theta in thetas:
		ax.plot([theta, theta], r, '#bfbfbf')
		
	ticks = np.linspace(0, 1.75*np.pi, 8)
	ax.xaxis.set_ticks(ticks)
	ax.xaxis.set_ticklabels("")
	ax.axis('off')
	
	if suptitle is not None:
		fig.suptitle(suptitle)
	if savepath is not None:
		plt.savefig(savepath)
	if show:
		plt.show()
	plt.close()

def plot_attraction_vectors(Y, idx=None, degrees_of_freedom=1, load_exact=False, s=2, cmap='rainbow', fig_len=5, fig_width=5, ax_off=False, show=True, suptitle=None, savepath=None, colorbar_fontsize=10, colorbar_kwargs={}):
	'''
	Plots embedding colored by attraction direction. This function assumes that
	P is saved in the current working directory.

	Parameters
	----------
	Y : numpy array, shape (n_samples, 2,) or array list
		Point set or list of point sets to color. If multiple point sets provided,
		we assume that all are indexed by sample
	idx : int list or None
		If not None, only plot samples in idx. This could be useful for examining 
		forces on subsets of Y. 
	degrees_of_freedom : int, default 1
		Degrees of freedom of t-distribution used to calculate Q.
	load_exact : bool, default False
		True if saved P was calculated for exact t-SNE, i.e. P is a dense matrix. 
		False if P was calculated for approximate t-SNE, i.e. P is a sparse matrix.
	colorbar_fontsize : int, default 20
		Font size for colorbar labels.
	colorbar_kwargs : dict, default {}
		Keyword args for generating colorbar.
	
	Returns: void
	'''

	plt.figure(figsize=(fig_len, fig_width))
	ax = plt.subplot(111)
	
	vectors = compute_attraction_forces(Y, degrees_of_freedom=degrees_of_freedom, load_exact=load_exact)
	
	magnitudes = np.linalg.norm(vectors, ord=2, axis=1)
	mappable = create_colormap(np.amin(magnitudes), np.amax(magnitudes), cmap=cmap)
	c = mappable.to_rgba(magnitudes)
	
	vectors = normalize(vectors)
	
	if idx is None:
		ax.quiver(Y[:,0], Y[:,1], vectors[:,0], vectors[:,1], color=c)
		ax.set_xlim(left=np.amin(Y[:,0]), right=np.amax(Y[:,0]))
		ax.set_ylim(bottom=np.amin(Y[:,1]), top=np.amax(Y[:,1]))
	else:
		ax.quiver(Y[idx,0], Y[idx,1], vectors[idx,0], vectors[idx,1], color=c)
		ax.set_xlim(left=np.amin(Y[idx,0]), right=np.amax(Y[idx,0]))
		ax.set_ylim(bottom=np.amin(Y[idx,1]), top=np.amax(Y[idx,1]))
	ax.axis('equal')
	if ax_off:
		ax.axis('off')
	cbar = plt.colorbar(mappable, ax=ax, **colorbar_kwargs)
	
	for t in cbar.ax.get_yticklabels():
		t.set_fontsize(colorbar_fontsize)
	
	if suptitle is not None:
		fig.suptitle(suptitle)
	if savepath is not None:
		plt.savefig(savepath, bbox_inches='tight')
	if show:
		plt.show()
	plt.close()

def plot_repulsion_vectors(Y, idx=None, degrees_of_freedom=1, s=2, cmap='rainbow', fig_len=5, fig_width=5, ax_off=False, show=True, suptitle=None, savepath=None, colorbar_fontsize=10, colorbar_kwargs={}):
	'''
	Plots embedding colored by repulsion direction. This function assumes that
	P is saved in the current working directory.

	Parameters
	----------
	Y : numpy array, shape (n_samples, 2,) or array list
		Point set or list of point sets to color. If multiple point sets provided,
		we assume that all are indexed by sample
	idx : int list or None
		If not None, only plot samples in idx. This could be useful for examining 
		forces on subsets of Y. 
	degrees_of_freedom : int, default 1
		Degrees of freedom of t-distribution used to calculate Q.
	colorbar_fontsize : int, default 20
		Font size for colorbar labels.
	colorbar_kwargs : dict, default {}
		Keyword args for generating colorbar.
	
	Returns: void
	'''

	plt.figure(figsize=(fig_len, fig_width))
	ax = plt.subplot(111)
	
	vectors = compute_repulsion_forces(Y, degrees_of_freedom=degrees_of_freedom)
	
	magnitudes = np.linalg.norm(vectors, ord=2, axis=1)
	mappable = create_colormap(np.amin(magnitudes), np.amax(magnitudes), cmap=cmap)
	c = mappable.to_rgba(magnitudes)
	
	vectors = normalize(vectors)
	
	if idx is None:
		ax.quiver(Y[:,0], Y[:,1], vectors[:,0], vectors[:,1], color=c)
		ax.set_xlim(left=np.amin(Y[:,0]), right=np.amax(Y[:,0]))
		ax.set_ylim(bottom=np.amin(Y[:,1]), top=np.amax(Y[:,1]))
	else:
		ax.quiver(Y[idx,0], Y[idx,1], vectors[idx,0], vectors[idx,1], color=c)
		ax.set_xlim(left=np.amin(Y[idx,0]), right=np.amax(Y[idx,0]))
		ax.set_ylim(bottom=np.amin(Y[idx,1]), top=np.amax(Y[idx,1]))
	ax.axis('equal')
	if ax_off:
		ax.axis('off')
	cbar = plt.colorbar(mappable, ax=ax, **colorbar_kwargs)
	
	for t in cbar.ax.get_yticklabels():
		t.set_fontsize(colorbar_fontsize)
	
	if suptitle is not None:
		fig.suptitle(suptitle)
	if savepath is not None:
		plt.savefig(savepath, bbox_inches='tight')
	if show:
		plt.show()
	plt.close()
