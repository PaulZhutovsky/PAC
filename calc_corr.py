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
import numpy as np
from docopt import docopt
from time import time
from sys import stdout


def run(data_location, save_folder, step_size=500):
    X = np.load(data_location)

    n_time = X.shape[1]

    X = X - X.mean(axis=1)[:, np.newaxis]
    X_var = np.sqrt(X.var(axis=1) * n_time)

    calc_correlations(X, X_var, step_size)


def calc_correlations(X, X_var, step_size):
    t1 = 0.
    t2 = 0.
    id_run = 1
    n_voxel = X.shape[0]
    num_iter = int(np.ceil(n_voxel/step_size))
    r = []

    for i in xrange(0, n_voxel, step_size):
        stdout.write('{}/{} {:.2f} \r'.format(id_run, num_iter, t2- t1))
        stdout.flush()

        t1 = time()
        tmp = np.triu(np.tensordot(X[i:i + step_size], X[i:], axes=(1, 1)) / (np.outer(X_var[i: i + step_size], X_var[i:])
                                                                                       + 0.00001))
        r.append(tmp[tmp!=0])

        id_run += 1
        t2 = time()
    return r

def main(args):
    data_location = args['DATA_MATRIX']
    save_folder = args['SAVE_LOCATION']
    step_size = int(args['--step'])
    run(data_location, save_folder, step_size)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    main(arguments)