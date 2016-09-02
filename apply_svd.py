"""
Calculates correlations across the time-series voxels.

Usage:
    calc_corr [--nfeats=<NFEATS>] DATA_FOLDER SVD_FILE SAVE_LOCATION

Arguments:
    DATA_FOLDER     Path to the data folder
    SVD_FILE        Path to SVD file
    SAVE_LOCATION   Where to save the result to

Options:
    --nfeats=<NFEATS>     Number of components to use [default: 100]
"""
import numpy as np
from docopt import docopt
from fft_transform import ensure_folder, get_subject_file_path
from time import time
from sys import stdout
import os.path as osp


def run(subject_files, save_folder, svd_file, n_feats):
    tmp = np.load(svd_file)
    U, s, V = tmp['U'], tmp['s'], tmp['V']
    t1 = 0.
    t2 = 0.
    features = np.zeros((len(subject_files), n_feats))

    for id_subject, subject_file in enumerate(subject_files):
        stdout.write('{}/{} {:.2f}\r'.format(id_subject + 1, len(subject_files), t2 - t1))
        stdout.flush()
        t1 = time()
        X_subj = np.load(subject_file)

        for id_M in xrange(n_feats):
            pattern = np.outer(U[:, id_M], V[id_M]) * s[id_M]
            import ipdb; ipdb.set_trace()
            features[id_subject, id_M] = (X_subj * pattern).sum()
        t2 = time()

    np.save(osp.join(save_folder, 'svd_features.npy'), features)


def main(args):
    data_location = args['DATA_FOLDER']
    svd_file = args['SVD_FILE']
    save_folder = args['SAVE_LOCATION']
    n_feats = int(args['--nfeats'])
    ensure_folder(save_folder)
    subject_files = get_subject_file_path(data_location)
    run(subject_files, save_folder, svd_file, n_feats)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    main(arguments)
