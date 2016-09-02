"""
Calculates correlations across the time-series voxels.

Usage:
    calc_corr  DATA_FOLDER SAVE_LOCATION

Arguments:
    DATA_FOLDER     Path to the data folder
    SAVE_LOCATION   Where to save the result to
"""
import numpy as np
from docopt import docopt
from fft_transform import ensure_folder, get_subject_file_path
from calc_psd_band import get_subject_id
from time import time
from sys import stdout
import os.path as osp


def run(subject_files, save_folder, num_voxel=15176):
    t1 = 0.
    t2 = 0.
    X_average = np.zeros((num_voxel, num_voxel))
    N = float(len(subject_files))
    for id_subj, subject_file in enumerate(subject_files):
        stdout.write('{}/{} {:.2f}\r'.format(id_subj + 1, len(subject_files), t2 - t1))
        stdout.flush()
        t1 = time()
        X = np.load(subject_file)
        corr_mat = np.corrcoef(X, rowvar=True)
        X_average += (corr_mat/N)

        subj_id = get_subject_id(subject_file)

        save_subj = subj_id + '_corr.npy'
        np.save(osp.join(save_folder, save_subj), corr_mat)
        t2 = time()
    print 'save average'
    np.save(osp.join(save_folder, 'average_corr.npy'), X_average)

def main(args):
    data_location = args['DATA_FOLDER']
    save_folder = args['SAVE_LOCATION']
    ensure_folder(save_folder)
    subject_files = get_subject_file_path(data_location)
    run(subject_files, save_folder)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    main(arguments)