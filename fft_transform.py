"""
Calculates the power spectrum of all time series of all subjects in a folder. Will use mklfft if possible!
The FFT will be performed after the mean of the data (across time) is rmoved and the signal is padded to the next power
of 2 length.

Usage:
    fft_transform [--relative] FOLDER_DATA SAVE_LOCATION

Arguments:
    FOLDER_DATA     Folder where the data is located
    SAVE_LOCATION   Where to save the power spectra to (folder, will be created if not existing)

Options:
    --relative      Whether to scale the power spectrum density or not (will not be done by default)
"""

import numpy as np
import os
import os.path as osp
from glob import glob
from sys import stdout
from time import time

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
    n = 2 ** int(np.log2(X_time.shape[1]) + 1)
    X_time = X_time - X_time.mean(axis=1)[:, np.newaxis]
    X_freq = np.fft.fft(X_time, n=n, axis=-1)

    # take only the positive frequencies, see numpy description of fft
    return X_freq[:, 1:n/2]


def calc_psd(X_freq, relative=True):
    psd = np.abs(X_freq)**2

    if relative:
        psd /=(psd.sum(axis=1)[:, np.newaxis] + 0.0001)
    return psd


def run(subject_files, save_folder, scale_psd):
    t1 = 0.
    t2 = 0.
    for id_file, subj_file in enumerate(subject_files):
        stdout.write('{}/{} {:.2f}s\r'.format(id_file + 1, len(subject_files), t2 - t1))
        stdout.flush()
        t1 = time()
        X = np.load(subj_file)
        X_fft = calc_fft(X)

        power_spectral_density = calc_psd(X_fft, relative=scale_psd)

        save_file = osp.basename(subj_file)
        save_file = save_file.rpartition('.')[0] + '_psd' + '.npy'
        np.save(osp.join(save_folder, save_file), power_spectral_density)
        t2 = time()


def main(args):
    data_folder = args['FOLDER_DATA']
    save_folder = args['SAVE_LOCATION']
    scale_psd = args['--relative']
    ensure_folder(save_folder)
    subject_files = get_subject_file_path(data_folder)
    run(subject_files, save_folder, scale_psd)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    main(arguments)
