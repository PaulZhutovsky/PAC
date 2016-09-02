"""
Removes noise voxels as defined by the union of flat voxels (across subjects) and voxels with high variance (var >98%)

Usage:
    remove_noise FOLDER_DATA ID_FILE SAVE_LOCATION

Arguments:
    FOLDER_DATA     Folder where the data is located
    ID_FILE         Location of id (boolean array) file
    SAVE_LOCATION   Where to save the power spectra to (folder, will be created if not existing)

"""
import numpy as np
import os.path as osp
from sys import stdout
from time import time
from fft_transform import ensure_folder, get_subject_file_path

from docopt import docopt


def run(subject_files, id_file_loc, save_loc):
    t1 = 0.
    t2 = 0.
    id_remove = np.load(id_file_loc)

    for id_subj, subject_file in enumerate(subject_files):
        stdout.write('{}/{} {:.2f}\r'.format(id_subj + 1, len(subject_files), t2 - t1))
        stdout.flush()
        t1 = time()
        X = np.load(subject_file)
        X_remove = X[~id_remove, :]
        subj_id = osp.basename(subject_file).partition('.')[0].partition('_')[0] + '_cleaned.npy'
        np.save(osp.join(save_loc, subj_id), X_remove)
        t2 = time()



def main(args):
    data_loc = args['FOLDER_DATA']
    id_file_loc = args['ID_FILE']
    save_loc = args['SAVE_LOCATION']
    ensure_folder(save_loc)
    subject_files = get_subject_file_path(data_loc)
    run(subject_files, id_file_loc, save_loc)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    main(arguments)