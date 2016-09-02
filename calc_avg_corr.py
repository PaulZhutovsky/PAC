"""
Calculates average correlations across specified data

Usage:
    calc_avg_corr DATA_FOLDER CHOSEN_SUBJ SAVE_LOCATION

Arguments:
    DATA_FOLDER     Path to the data folder
    SAVE_LOCATION   Where to save the result to
    CHOSEN_SUBJ     Path to an array of *int* elements of subject_ids to pick
"""
import numpy as np
from docopt import docopt
from fft_transform import ensure_folder, get_subject_file_path
from calc_psd_band import get_subject_id
from time import time
from sys import stdout
import os.path as osp


def run(subject_files, save_folder, chosen_subj, num_voxel=15176):
    t1 = 0.
    t2 = 0.
    corr_average = np.zeros((num_voxel, num_voxel))
    chosen_subj = np.load(chosen_subj)
    N = chosen_subj.size

    for id_subj, subject_file in enumerate(subject_files):
        if get_subject_id(subject_file).isdigit() and np.any(int(get_subject_id(subject_file)) == chosen_subj):
            stdout.write('{}/{} {:.2f}\r'.format(id_subj + 1, len(subject_files), t2 - t1))
            stdout.flush()
            t1 = time()
            corr_average += np.load(subject_file)
            t2 = time()
    corr_average /= N

    print 'save average'
    np.save(osp.join(save_folder, 'average_corr.npy'), corr_average)


def main(args):
    data_location = args['DATA_FOLDER']
    save_folder = args['SAVE_LOCATION']
    chosen_subj = args['CHOSEN_SUBJ']
    ensure_folder(save_folder)
    subject_files = get_subject_file_path(data_location)
    run(subject_files, save_folder, chosen_subj)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    main(arguments)