"""
Calculates the amount of flat vocels per subject and also the variance across the time-series per voxel per subject.
Stores them afterwards.

Usage:
    flat_voxels_investigation FOLDER_DATA SAVE_LOCATION

Arguments:
    FOLDER_DATA     Folder where the data is located
    SAVE_LOCATION   Where to save the power spectra to (folder, will be created if not existing)

"""

import numpy as np
import os.path as osp
from sys import stdout
from time import time
from fft_transform import ensure_folder, get_subject_file_path

from docopt import docopt


def get_flat_voxels(data):
    return np.all(np.diff(data, axis=1) == 0, axis=1)


def get_var_voxels(data):
    return data.var(axis=1)


def run(subject_files, save_loc, num_voxels=228483):
    t1 = 0.
    t2 = 0.
    t_start = time()
    flat_voxels = np.zeros((len(subject_files), num_voxels))
    var_voxels = np.zeros((len(subject_files), num_voxels))

    for id_subj, subject_file in enumerate(subject_files):
        stdout.write('{}/{} {:.2f}\r'.format(id_subj + 1, len(subject_files), t2 - t1))
        stdout.flush()
        
        t1 = time()
        X = np.load(subject_file)

        flat_voxels[id_subj] = get_flat_voxels(X)
        var_voxels[id_subj] = get_var_voxels(X)
        t2 = time()

    print 'Total time: {:.2f}'.format(time() - t_start)

    np.save(osp.join(save_loc, 'flat_voxels_subjects.npy'), flat_voxels)
    np.save(osp.join(save_loc, 'var_voxels_subjects.npy'), var_voxels)


def main(args):
    data_loc = args['FOLDER_DATA']
    save_loc = args['SAVE_LOCATION']
    ensure_folder(save_loc)
    subject_files = get_subject_file_path(data_loc)
    run(subject_files, save_loc)


if __name__ == '__main__':
    arguments =  docopt(__doc__)
    main(arguments)