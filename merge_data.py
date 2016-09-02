"""
Merges data from different sources into one final array

Usage:
    merge_data [--save_loc=<SAVE_LOC>] (DATA_LOC ...)

Arguments:
    DATA_LOC     Location of data to be merged: can be as many locations (file_path) as you want

Options:
    --save_loc=<SAVE_LOC>   where to save the data to [default: .]
"""

import numpy as np
from docopt import docopt
from fft_transform import ensure_folder
from time import time
from sys import stdout
import os.path as osp


def run(data_locations, save_folder):
    t1 = 0.
    t2 = 0.

    for id_data, data_loc in enumerate(data_locations):
        stdout.write('{}/{} {:.2f}\r'.format(id_data + 1, len(data_locations), t2 - t1))
        stdout.flush()
        t1 = time()
        if id_data == 0:
            ftr_mat = np.load(data_loc)
        else:
            ftr_mat = np.concatenate((ftr_mat, np.load(data_loc)), axis=1)
        t2 = time()

    print 'save final features'
    np.savez_compressed(osp.join(save_folder, 'average_corr.npz'), X_ftr=ftr_mat, loaded_pathes=data_locations)


def main(args):
    data_locations = args['DATA_LOC']
    save_folder = args['--save_loc']
    ensure_folder(save_folder)
    run(data_locations, save_folder)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    main(arguments)