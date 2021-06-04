import os
import sys
import getopt
import numpy as np
import scipy.io as sio


def _load_raw_data(data_path):
    S_RATE = 1000
    dir_list = os.listdir(data_path)

    # mat1: raw_eeg1~raw_eeg17 (removing data: 9, 13)
    # mat2: raw_eeg18~raw_eeg34 (removing data: 25, 34)
    mat1_idx = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16, 17]
    mat2_idx = [18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33]

    labels = [
        0, 2, 1, 4, 3, 5, 6, 7, 1, 2, 3, 6, 7, 4, 5, 0, 1, 5, 6, 2, 2, 1, 7, 6,
        4, 4, 3, 5, 3, 7
    ]

    train_trials = [
        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 23, 25,
        27
    ]
    test_trials = [21, 22, 24, 26, 28, 29, 30]  # 1, 16 are omitted

    # Label information needed to be added
    eeg_all = []
    for d in dir_list:
        eeg_signals = []
        subj_path = os.path.join(data_path, d)
        if os.path.isdir(subj_path):
            print(d)
            fname1 = os.path.join(subj_path, f'{d}_d.mat')
            fname2 = os.path.join(subj_path, f'{d}_dd.mat')

            mat1 = sio.loadmat(fname1)
            mat2 = sio.loadmat(fname2)

            for idx in mat1_idx:
                sig = mat1[f'raw_eeg{idx}']
                eeg_signals.append(sig[:, :S_RATE * 60])

            for idx in mat2_idx:
                sig = mat2[f'raw_eeg{idx}']
                eeg_signals.append(sig[:, :S_RATE * 60])

            eeg_signals = np.stack(eeg_signals, axis=0)
        eeg_all.append(eeg_signals)

    return eeg_all


def main(argv):
    input_path = None
    output_path = None
    try:
        opts, args = getopt.getopt(
            argv, "hd:i:o:s:", ["dataset_type=", "input_path=", "output_path="])
    except getopt.GetoptError:
        print(
            'lmdb_generate_rev.py -i <inputpath> -o <outputpath> -d <dataset_type>'
        )
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(
                'lmdb_generate_rev.py -i <inputpath> -o <outputpath> -d <dataset_type>'
            )
            sys.exit()
        elif opt in ("-i", "--input_path"):
            input_path = arg
        elif opt in ("-o", "--output_path"):
            output_path = arg
        elif opt in ("-d", "--dataset_type"):
            dataset_type = arg

    assert input_path is not None, 'Input path should be specified.'
    assert output_path is not None, 'Output path should be specified.'

    construct_lmdb_dataset(input_path, output_path, dataset_type)