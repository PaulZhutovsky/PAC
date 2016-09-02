"""
Calculates different statistics across the time-series of voxels for every subject.

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
from scipy.stats import kurtosis, skew
from collections import OrderedDict


def IQR(data, axis=1):
    q75, q25 = np.percentile(data, q=[75, 25], axis=axis)
    return q75 - q25


def max_min_ratio(data, axis=1):
    return np.abs(data.max(axis=axis)/data.min(axis=axis))


def run(subject_files, save_folder, num_voxels=15176):
    t1 = 0.
    t2 = 0.
    fun_dict = OrderedDict([('mean', np.mean), ('var', np.var), ('kurtosis', kurtosis),
                            ('skewness', skew), ('median', np.median), ('IQR', IQR),
                            ('abs(max/min)', max_min_ratio)])
    labels = fun_dict.keys()
    temp_ftr_mat = np.zeros((len(subject_files), num_voxels * len(labels)))

    for id_subj, subject_file in enumerate(subject_files):
        stdout.write('{}/{} {:.2f}\r'.format(id_subj + 1, len(subject_files), t2 - t1))
        stdout.flush()
        t1 = time()
        X = np.load(subject_file)

        for id_label, label in enumerate(labels):
            temp_ftr_mat[id_subj, id_label * num_voxels:(id_label + 1) * num_voxels] = fun_dict[label](X, axis=1)
        t2 = time()
    print 'save features'
    np.savez_compressed(osp.join(save_folder, 'features_temporal.npz'), X_temp=temp_ftr_mat,
                        labels_ftrs=labels, n_voxels=num_voxels)


def main(args):
    data_location = args['DATA_FOLDER']
    save_folder = args['SAVE_LOCATION']
    ensure_folder(save_folder)
    subject_files = get_subject_file_path(data_location)
    run(subject_files, save_folder)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    main(arguments)