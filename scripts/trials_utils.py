import numpy as np
import os
import pandas as pd
import pickle
import shutil
import sys

from networkx import read_adjlist, to_numpy_array

SRC_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../src')
sys.path.append(SRC_DIR)
from numerics import *
from plot_utils import plot_heatmap
from tsne_utils import compute_repulsion_magnitudes

'''
This library provides functions for organizing and running numerics calculations for 
different parameter settings. Runs are organized by

root
 |__ params
        |__ trial # 

Parameter choices are assigned an integer id. The id -> parameter mapping is saved in a 
PARAMETERS_CSV in the root directory. The next available id is stored in NEXT_ID_FILE and 
the parameter to id mapping is pickled in PARAMETER_TO_ID_FILE. The user should not alter
these files.
'''

## File name constants

# Numpy array containing input data. 
INPUT_DATA_FILE = "X.npy"
# Numpy array containing t-SNE embedding
EMBEDDING_FILE = "Y.npy"
# Networkx graph for input affinities, stored as adjacency list
GRAPH_FILE = "graph.dat"
# csv of derivation statistics for experiment
STATS_FILE = "stats.csv"

# csvs describing id to parameter setting mapping
PARAMETERS_CSV = "parameters.csv"

# file containing next available experiment / trial id
NEXT_ID_FILE = ".next_id"
# file containing pickled map of parameters to experiment ids
REGISTERED_PARAMETERS_FILE = ".registered_parameters.pickle"

## Functions for setting up/running numerics trials

def _clear_directory(folder):
	for filename in os.listdir(folder):
		file_path = os.path.join(folder, filename)
		try:
			if os.path.isfile(file_path) or os.path.islink(file_path):
				os.unlink(file_path)
			elif os.path.isdir(file_path):
				shutil.rmtree(file_path)
		except Exception as e:
			print('Failed to delete %s. Reason: %s' % (file_path, e))

def is_initialized(experiment_dir):
	next_id_file = os.path.join(experiment_dir, NEXT_ID_FILE)
	registered_parameters_file = os.path.join(experiment_dir, REGISTERED_PARAMETERS_FILE)
	parameters_csv = os.path.join(experiment_dir, PARAMETERS_CSV)

	return os.path.isfile(next_id_file) and os.path.isfile(registered_parameters_file) and os.path.isfile(parameters_csv)

def initialize_experiment(parameter_names, experiment_dir):
	'''
	Create or reset PARAMETERS_CSV, NEXT_ID_FILE, REGISTERED_PARAMETERS_FILE.
	'''

	next_id_file = os.path.join(experiment_dir, NEXT_ID_FILE)
	registered_parameters_file = os.path.join(experiment_dir, REGISTERED_PARAMETERS_FILE)
	parameters_csv = os.path.join(experiment_dir, PARAMETERS_CSV)

	# write first id to next_id_file
	with open(next_id_file, 'w') as f:
		f.write('1')

	# initialize registered parameters map
	with open(registered_parameters_file, 'wb') as f:
		pickle.dump({}, f, protocol=pickle.HIGHEST_PROTOCOL)

	# write header of parameters_csv 
	with open(parameters_csv, 'w') as f:
		f.write('params_id')
		for param in parameter_names:
			f.write(',{}'.format(param))
		f.write('\n')

def get_next_id(experiment_dir):
	'''
	Gets then increments next available id. Trial ids are integers, begin at 1.
	'''
	next_id_file = os.path.join(experiment_dir, NEXT_ID_FILE)
	with open(next_id_file, 'r+') as f: 
		next_id = int(f.readline())
		f.seek(0)
		f.write('{}'.format(next_id + 1))
		f.truncate()
	return next_id

def get_params_id(params_key, experiment_dir):
	'''
	Get experiment id associated with params. Return None if params
	not registered.

	Parameters:
		params_key: tuple
		experiment_dir: string
	Return:
		params_id : int or None

	'''
	# load registered params
	registered_parameters_file = os.path.join(experiment_dir, REGISTERED_PARAMETERS_FILE)
	with open(registered_parameters_file, 'rb') as f:
		registered_parameters = pickle.load(f)

	if params_key in registered_parameters:
		return registered_parameters[params_key]

	return None

def get_params_dir(params_key, experiment_dir):
	'''
	Get directory containing experiment data for params setting. Return
	None if params not registered.
	'''
	params_id = get_params_id(params_key, experiment_dir)
	if params_id is not None:
		return os.path.join(experiment_dir, params_id)
	return None

def get_params_df(experiment_dir):
	parameters_csv = os.path.join(experiment_dir, PARAMETERS_CSV)
	return pd.read_csv(parameters_csv, dtype={'params_id':np.int32})

def register_parameters(params_keys, experiment_dir):
	'''
	Registers parameter settings, returns list of assigned ids.
	'''
	ids = []

	# Load registered parameters
	registered_parameters_file = os.path.join(experiment_dir, REGISTERED_PARAMETERS_FILE)
	with open(registered_parameters_file, 'rb') as f:
		registered_parameters = pickle.load(f)

	# Register new parameters, get ids of registered parameters.
	parameters_csv = os.path.join(experiment_dir, PARAMETERS_CSV)
	with open(parameters_csv, 'a') as f:
		for params_key in params_keys:
			if params_key in registered_parameters:
				# Params already registered
				ids.append(registered_parameters[params_key])
			else: 
				# Register new params
				next_id = get_next_id(experiment_dir)
				ids.append(next_id)

				# add parameters to csv
				f.write('{}'.format(next_id))
				for param in params_key:
					f.write(',{}'.format(param))
				f.write('\n')

				registered_parameters[params_key] = next_id

	# Update registered parameters
	with open(registered_parameters_file, 'wb') as f:
		pickle.dump(registered_parameters, f, protocol=pickle.HIGHEST_PROTOCOL)

	return ids

def run_trials(X, params, n_trials, run_trial, params_dir, verbose=0, kwargs={}):
	if not os.path.isdir(params_dir):
		os.mkdir(params_dir)
		# write first id to next_id_file
		next_id_file = os.path.join(params_dir, NEXT_ID_FILE)
		with open(next_id_file, 'w') as f:
			f.write('1')

		if verbose > 0:
			print('Created params_dir {}'.format(params_dir))

	for i in range(n_trials):
		if verbose > 0:
			print('Running trial {} of {} for params {}'.format(i+1, n_trials, params))

		trial_id = get_next_id(params_dir)
		trial_dir = os.path.join(params_dir, '{}'.format(trial_id))
		os.mkdir(trial_dir)
		run_trial(X, params, trial_dir, verbose=verbose, kwargs=kwargs)

## Functions for computing experiment results

def has_file(dir_name, file_name):
	path = os.path.join(dir_name, file_name)
	return os.path.isfile(path)

def data_complete(trial_dir):
	'''
	Check whether all data for a trial is present
	'''
	has_graph = has_file(trial_dir, GRAPH_FILE)
	has_Y = has_file(trial_dir, EMBEDDING_FILE)
	has_P_approx = has_file(trial_dir, 'P_col.dat') and has_file(trial_dir, 'P_row.dat') and has_file(trial_dir, 'P_val.dat')

	return has_graph and has_Y and has_P_approx

def plot_repulsion(trial_dir):
	'''
	Generate and save repulsion magnitude heatplot for trial from saved data.
	'''
	Y = np.load(os.path.join(trial_dir, EMBEDDING_FILE))
	# Plot repulsion heatmap
	repulsion_magnitudes = compute_repulsion_magnitudes(Y)
	savepath = os.path.join(trial_dir, "repulsion.png")
	plot_heatmap(Y, repulsion_magnitudes, "repulsion", show=False, savepath=savepath)

def calculate_numerics(n, p, trial_dir):
	'''
	Calculate energy, expected energy, variance, and taylor approximations
	for a trial from saved data.
	'''
	Y = np.load(os.path.join(trial_dir, EMBEDDING_FILE))
	graph = read_adjlist(os.path.join(trial_dir, GRAPH_FILE))
	graph = to_numpy_array(graph)
	edge_wt = 1.0 / (np.sum(graph))

	energy = compute_energy(Y, graph, edge_wt, p)
	expected_energy = compute_expected_energy(Y, graph, edge_wt, p)
	variance = compute_variance(Y, edge_wt, p)
	expected_energy_taylor = compute_expected_energy_taylor(Y, edge_wt, p)
	variance_taylor = compute_variance_taylor(Y, edge_wt, p)

	return energy, expected_energy, variance, expected_energy_taylor, variance_taylor

def compute_experiment_results(experiment_dir, verbose=0):
	'''
	Plot repulsion heatmap and perform numerics calculations for all 
	trials in experiment set. Save values to csv.
	'''
	stats = []
	params_df = get_params_df(experiment_dir)

	# Check Erdos-Renyi parameters are present
	assert('n' in params_df.columns and 'p' in params_df.columns)

	for idx in params_df.index:
		params_id = params_df.loc[idx, 'params_id']

		if verbose > 0:
			print('Processing params_id {}'.format(params_id))

		n = params_df.loc[idx, 'n']
		p = params_df.loc[idx, 'p']

		params_dir = os.path.join(experiment_dir, '{}'.format(params_id))
		for trial_id in os.listdir(params_dir):
			trial_dir = os.path.join(params_dir, trial_id)
			# Path is a trial directory iff it is a subfolder.
			if os.path.isdir(trial_dir) and data_complete(trial_dir):
				# path is trial dir

				if verbose > 0:
					print('Processing trial {} for params {}'.format(trial_id, (n,p)))
				plot_repulsion(trial_dir)
				energy, expected_energy, variance, expected_energy_taylor, variance_taylor = calculate_numerics(n, p, trial_dir)
				stats.append([params_id, int(trial_id), n, p, energy, expected_energy, variance, expected_energy_taylor, variance_taylor])

	columns = ['params_id', 'trial_id', 'n', 'p', 'energy', 'expected_energy', 'variance', 'expected_energy_taylor', 'variance_taylor']
	stats = pd.DataFrame(data=stats, columns=columns)
	stats.to_csv(os.path.join(experiment_dir, STATS_FILE),index=False)
