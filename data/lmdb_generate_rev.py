import pickle
import os
import sys
import getopt
import itertools
from pathlib import Path
from sys import platform

import lmdb
import torch.utils.data as data
import scipy.io as sio
import numpy as np
from tqdm import tqdm, trange

from dataset import *

'''
Objective
1. Make K-fold dataset
2. Time, video, subject variable K-fold (possible?)
3. Using K-fold to get bsc_lcs results
'''

def _load_raw_data_deap(data_path):
    N_SUBJ = 32
    S_RATE = 128

    if platform == 'linux':
        data_key = b'data'
        label_key = b'labels'
    else:
        data_key = 'data'
        label_key = 'labels'

    signal_list = []
    label_list = []
    for i in tqdm(range(N_SUBJ), desc='Loading data - subject '):
        file_name = 's{:02d}.dat'.format(i + 1)
        with open(os.path.join(data_path, file_name), 'rb') as f:
            subj_data = pickle.load(f, encoding='bytes')
        subj_signal = subj_data[data_key][:, :32, 3 * S_RATE:]
        subj_label = subj_data[label_key]

        signal_list.append(subj_signal)
        label_list.append(subj_label)

    signal_list = np.stack(signal_list, axis=0)
    label_list = np.stack(label_list, axis=0)

    return signal_list, label_list, [
        'VALENCE', 'AROUSAL', 'DOMINANCE', 'LIKING'
    ]


def _load_raw_data_dreamer(data_path):
    mat_data = sio.loadmat(os.path.join(data_path, 'Processed.mat'))

    valence = mat_data['valence']
    arousal = mat_data['arousal']
    dominance = mat_data['dominance']
    stimuli = mat_data['stimuli']
    baseline = mat_data['baseline']

    eeg_list = stimuli.astype(np.float32)
    label_list = np.stack((valence, arousal, dominance), axis=-1)

    # eeg_list.shape = N_SUBJ, N_TRIAL, N_ELECTRODE, V_LENGTH * S_RATE
    # label_list.shape = N_SUBJ, N_TRIAL, 3(val, ars, dom)

    return eeg_list, label_list, ['VALENCE', 'AROUSAL', 'DOMINANCE']


def divide_signal(signals, labels, s_rate, window, overlap, axis=0):
    # s_rate: samples / sec
    # window: window length in seconds
    #    * actual length: window * s_rate
    # overlap: overlap between windows in seconds
    #    * actual length: overlap * s_rate

    signal_len = signals.shape[-1]
    stride = (window - overlap) * s_rate
    cursor_curr = 0
    cursor_end = cursor_curr + window * s_rate

    signals_divided = []
    labels_divided = []
    while cursor_end <= signal_len:
        signals_divided.append(signals[..., cursor_curr:cursor_end])
        labels_divided.append(labels)
        cursor_curr += stride
        cursor_end += stride

    signals_divided = np.stack(signals_divided, axis=axis)
    labels_divided = np.stack(labels_divided, axis=axis)

    return signals_divided, labels_divided


def construct_lmdb(lmdb_root, lmdb_size, signals, labels, tag=''):
    lmdb_env = lmdb.open(lmdb_root, map_size=lmdb_size)

    n_sub, n_tri, n_win, _, _ = signals.shape
    list_all = itertools.product(range(n_sub), range(n_tri), range(n_win))

    data_index = 0
    with lmdb_env.begin(write=True) as txn:
        for sub, tri, win in tqdm(list_all, desc='Constructing LMDB: ' + tag):
            signal_t = signals[sub, tri, win]
            label_t = labels[sub, tri, win]
            datum = EEGDatum(dshape=np.asarray(signal_t.shape, dtype=np.int32),
                             ddata=signal_t.astype(np.float32),
                             lshape=np.asarray(label_t.shape, dtype=np.int32),
                             ldata=label_t.astype(np.float32), subject=sub,
                             trial=tri)

            str_id = '{:06}'.format(data_index)
            txn.put(str_id.encode('ascii'), datum.encode())
            data_index += 1

    lmdb_env.close()

    return data_index


def construct_lmdb_dataset(input_path, output_path, dataset_type):
    if dataset_type.lower() == 'deap':
        signals_raw, labels_raw, label_name = _load_raw_data_deap(input_path)
        MAP_SIZE_MULTIPLER = 3
    elif dataset_type.lower() == 'dreamer':
        signals_raw, labels_raw, label_name = _load_raw_data_dreamer(input_path)
        MAP_SIZE_MULTIPLER = 3

    # use only valence and arousal in revision
    labels_raw = labels_raw[..., :2]
    label_name = label_name[:2]

    print('-----Signal Statistics-----')
    signals_mean = np.mean(signals_raw)
    signals_std = np.std(signals_raw)
    print('Mean: {}, Stddev: {}'.format(signals_mean, signals_std))
    signals_raw = (signals_raw - signals_mean) / signals_std

    signals_div, labels_div = divide_signal(signals_raw, labels_raw, 128, 3, 2,
                                            axis=0)

    indices_list = [[0, 12, 0, 9], [9, 24, 12, 21], [21, 36, 24, 33],
                    [33, 48, 36, 45], [45, 58, 48, 57]]

    for i, indices in enumerate(indices_list):
        train_l, train_r, test_l, test_r = indices

        train_signal = np.concatenate(
            (signals_div[0:train_l], signals_div[train_r:]), axis=0)
        train_label = np.concatenate(
            (labels_div[0:train_l], labels_div[train_r:]), axis=0)

        test_signal = signals_div[test_l:test_r]
        test_label = labels_div[test_l:test_r]

        # signal.shape = [window, subject, video, electrode, time]
        # label.shape = [window, subject, video, emotions] (valence, arousal, dominance, liking)
        train_signal = np.transpose(train_signal, [1, 2, 0, 3, 4])
        test_signal = np.transpose(test_signal, [1, 2, 0, 3, 4])

        train_label = np.transpose(train_label, [1, 2, 0, 3])
        test_label = np.transpose(test_label, [1, 2, 0, 3])

        print('-----Processed Data Shape-----')
        print('Train data: {}, label: {}'.format(str(train_signal.shape),
                                                 str(train_label.shape)))
        print('Test data: {}, label: {}'.format(str(test_signal.shape),
                                                str(test_label.shape)))

        d_size = train_signal.dtype.itemsize
        l_size = train_label.dtype.itemsize

        train_map_size = train_signal.size * d_size + train_label.size * l_size
        test_map_size = test_signal.size * d_size + test_label.size * l_size

        train_map_size *= MAP_SIZE_MULTIPLER
        test_map_size *= MAP_SIZE_MULTIPLER

        train_root = os.path.join(output_path, f'{i}', 'train//')
        test_root = os.path.join(output_path, f'{i}', 'test//')

        Path(train_root).mkdir(parents=True, exist_ok=True)
        Path(test_root).mkdir(parents=True, exist_ok=True)

        train_records = construct_lmdb(train_root, train_map_size, train_signal,
                                       train_label, 'train')
        test_records = construct_lmdb(test_root, test_map_size, test_signal,
                                      test_label, 'test')

        test_lmdb_dataset(os.path.join(output_path, f'{i}'), batch_size=256)
        test_lmdb_dataset(os.path.join(output_path, f'{i}'), batch_size=256)

        print('-----LMDB Records-----')
        print('Train path: ', train_root)
        print('Test path: ', test_root)
        print('{} - #Train: {}, #Test: {}'.format(i, train_records,
                                                  test_records))


def test_lmdb_dataset(lmdb_root, batch_size=256):
    train_root = os.path.join(lmdb_root, 'train//')
    test_root = os.path.join(lmdb_root, 'test//')

    train_loader = data.DataLoader(EEGDataset(root=train_root, transform=None),
                                   batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(EEGDataset(root=test_root, transform=None),
                                  batch_size=batch_size, shuffle=True)

    train_mean = []
    for signal, label in tqdm(train_loader):
        train_mean.append(signal.mean())

    test_mean = []
    for signal, label in tqdm(test_loader):
        test_mean.append(signal.mean())

    print(np.mean(train_mean))
    print(np.mean(test_mean))


def main(argv):
    input_path = None
    output_path = None
    dataset_type = 'deap'

    try:
        opts, args = getopt.getopt(
            argv, "hd:i:o:s:", ["dataset_type=", "input_path=", "output_path="])
    except getopt.GetoptError:
        print(
            'lmdb_generate_rev.py -i <inputpath> -o <outputpath> -d <dataset_type>')
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


if __name__ == "__main__":
    # Construct LMDB dataset from DEAP.
    main(sys.argv[1:])
