"""
Calculates the power spectrum of all time series of all subjects in a folder. Will use mklfft if possible!
The FFT will be performed after the mean of the data (across time) is rmoved and the signal is padded to the next power
of 2 length.

Usage:
    fft_transform FOLDER_DATA SAVE_LOCATION

Arguments:
    FOLDER_DATA     Folder where the data is located
    SAVE_LOCATION   Where to save the power spectra to (folder, will be created if not existing)
"""

import numpy as np
import os
import os.path as osp
from glob import glob
from sys import stdout

from docopt import docopt


def ensure_folder(folder_path):
    if not osp.exists(folder_path):
        os.makedirs(folder_path)


def get_subject_file_path(folder_path):
    file_names = sorted(glob(osp.join(folder_path, '*.npy')))

    if not file_names:
        raise RuntimeError('No files in the specified folder! {}'.format(folder_path))

    return [osp.join(folder_path, file_name) for file_name in file_names]


def calc_fft(X_time):
    n = int(np.log2(X_time.shape[1]) + 1)
    X_freq = np.fft.fft(X_time, n=n, axis=-1)

    # take only the positive frequencies, see numpy description of fft
    return X_freq[:, 1:n/2]


def calc_psd(X_freq):
    return np.abs(X_freq)**2


def run(subject_files, save_folder):

    for id_file, subj_file in enumerate(subject_files):
        stdout.write('{}/{}\r'.format(id_file + 1, len(subject_files)))
        stdout.flush()

        X = np.load(subj_file)
        X_fft = calc_fft(X)

        power_spectral_density = calc_psd(X_fft)

        save_file = osp.basename(subj_file)
        save_file = save_file.rpartition('.')[0] + '_psd' + '.npy'
        np.save(osp.join(save_folder, save_file), power_spectral_density)


def main(args):
    data_folder = args['FOLDER_DATA']
    save_folder = args['SAVE_LOCATION']
    ensure_folder(save_folder)
    subject_files = get_subject_file_path(data_folder)
    run(subject_files, save_folder)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    main(arguments)
