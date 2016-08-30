"""
Calculates amount of (relative) power in a specific band (default: resting state).

Usage:
    calc_psd_band [--low_thr=<LOW_THR> --high_thr=<HIGH_THR> --relative] FOLDER_DATA DESC_FILE SAVE_LOCATION

Arguments:
    FOLDER_DATA     Folder where the data is located
    DESC_FILE       File path for the description path (to extract the TR (expected to be a csv-file)
    SAVE_LOCATION   Where to save the power spectra to (folder, will be created if not existing)

Options:
    --relative              depending on whether the psd is already scaled or not you can scale it here again
    --low_thr=<LOW_THR>     the lower threshold for band in interest [default: 0.005]
    --high_thr=<HIGH_THR>   the higher threshold for band in interest [default: 0.01]
"""

import os.path as osp
import numpy as np
import pandas as pd
from fft_transform import ensure_folder, get_subject_file_path
from docopt import docopt
from sys import stdout
from time import time


def calc_freq(n, TR):
    # again only return the positive frequency
    return np.fft.fftfreq(n, d=TR)[1:n/2]


def normalize_psd(X_psd):
    return X_psd/(X_psd.sum(axis=1)[:, np.newaxis] + 0.00001)


def calculate_power_in_band(X_psd, freq, low_thresh, high_thresh):
    band_roi = (freq >= low_thresh) & (freq < high_thresh)
    return X_psd[:, band_roi].sum(axis=1)


def run(subject_files, save_folder, desc_file_df, low_thresh, high_thresh, scale_psd):
    t1 = 0.
    t2 = 0.
    for id_file, subj_file in enumerate(subject_files):
        stdout.write('{}/{} {:.2f}s\r'.format(id_file + 1, len(subject_files), t2 - t1))
        stdout.flush()

        t1 = time()
        X_psd = np.load(subj_file)
        subj_id = int(osp.basename(subj_file).partition('.')[0].partition('_')[0])
        TR_subj = float(desc_file_df.loc[desc_file_df.subject_id == subj_id, 'subject_sites_TR'])

        # a bit hacky.. we need to get the original n (with the padding)
        n = (X_psd.shape[1] + 1) * 2
        freq = calc_freq(n, TR_subj)
        if scale_psd:
            X_psd = normalize_psd(X_psd)

        X_band = calculate_power_in_band(X_psd, freq, low_thresh, high_thresh)

        save_file = osp.basename(subj_file)
        save_file = save_file.rpartition('.')[0] + '_band' + '.npy'
        np.save(osp.join(save_folder, save_file), X_band)
        t2 = time()


def get_desc_file(desc_file):
    return pd.read_csv(desc_file)


def main(args):
    data_folder = args['FOLDER_DATA']
    desc_file_path = args['DESC_FILE']
    save_folder = args['SAVE_LOCATION']
    low_thresh = float(args['--low_thr'])
    high_thresh = float(args['--high_thr'])
    scale_psd = args['--relative']
    ensure_folder(save_folder)
    subject_files = get_subject_file_path(data_folder)
    desc_file_df = get_desc_file(desc_file_path)

    run(subject_files, save_folder, desc_file_df, low_thresh, high_thresh, scale_psd)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    main(arguments)