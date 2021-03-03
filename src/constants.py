import os
import numpy as np

'''
Don't change these constants.
'''

# smallest divisor. avoid numerical error when dividing by small values.
MACHINE_EPSILON = np.finfo(np.double).eps

# project directory. 
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# this is useful for locating datasets within the project.
DATA_DIR = os.path.join(ROOT_DIR, "data")

# this is used to help with module __init__ files.
SRC_DIR = os.path.join(ROOT_DIR, "src")

'''
Change these constants for your local installation.
'''

# Path of FIt-SNE installation
TSNE_PATH = os.path.join(ROOT_DIR, '../packages/FIt-SNE')

# Path to MNIST training images
MNIST_IMAGES_PATH = os.path.join(DATA_DIR, "MNIST/train-images-idx3-ubyte")

# Path to MNIST training labels
MNIST_LABELS_PATH = os.path.join(DATA_DIR, "MNIST/train-labels-idx1-ubyte")