"""
Calculates correlations across the time-series voxels of a matrix.

Usage:
    calc_corr [--step=<STEP>] DATA_MATRIX SAVE_LOCATION

Arguments:
    DATA_MATRIX     Path to the matrix which will be used to calculate the correlations
    SAVE_LOCATION   Where to save the result to

Options:
    --step=<STEP>   How many correlations should be calculated per one step [default: 500]
"""
from __future__ import print_function
import numpy as np
from docopt import docopt
from time import time
from numba import jit


def run(data_location, save_folder, step_size=500):
    X = np.load(data_location)

    n_voxel, n_time = X.shape

    X = X - X.mean(axis=1)[:, np.newaxis]
    X_var = np.sqrt(X.var(axis=1) * n_time)

    calc_correlations(X, X_var, n_voxel, step_size)


#@jit
def calc_correlations(X, X_var, n_voxel, step_size):
    t1 = 0.
    t2 = 0.
    for i in xrange(0, n_voxel, step_size):
        print('{:.2f}'.format(t2 - t1))
        t1 = time()
        x_step = X[i:i + step_size]
        np.tensordot(x_step, X, axes=(1, 1)) / (np.outer(X_var[i: i + step_size], X_var) + 0.00001)
        t2 = time()


def main(args):
    data_location = args['DATA_MATRIX']
    save_folder = args['SAVE_LOCATION']
    step_size = int(args['--step'])
    run(data_location, save_folder, step_size)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    main(arguments)