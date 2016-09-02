"""
Calculates SVD decomposition for a correlation matrix   .

Usage:
    calc_corr CORR_MATRIX SAVE_LOCATION

Arguments:
    CORR_MATRIX     Path to the data folder
    SAVE_LOCATION   Where to save the result to
"""
import numpy as np
from docopt import docopt
from fft_transform import ensure_folder
from time import time
import os.path as osp


def run(corr_mat_loc, save_folder):
    t1 = time()
    corr_mat = np.load(corr_mat_loc)
    U, s, V = np.linalg.svd(corr_mat)
    np.savez_compressed(osp.join(save_folder, 'svd_corr.npz'), U=U, s=s, V=V)
    t2 = time()
    print 'Time taken: {:.2f}'.format(t2 - t1)


def main(args):
    corr_location = args['CORR_MATRIX']
    save_folder = args['SAVE_LOCATION']
    ensure_folder(save_folder)
    run(corr_location, save_folder)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    main(arguments)