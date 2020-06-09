import pickle
import os
import sys
import getopt
import itertools

import lmdb
import torch.utils.data as data
import scipy.io as sio
import numpy as np
from tqdm import tqdm, trange

from dataset import *


def _load_raw_data_deap(data_path):
    N_SUBJ = 32
    S_RATE = 128

    signal_list = []
    label_list = []
    for i in tqdm(range(N_SUBJ), desc='Loading data - subject '):
        file_name = 's{:02d}.dat'.format(i + 1)
        with open(os.path.join(data_path, file_name), 'rb') as f:
            subj_data = pickle.load(f)
        subj_signal = subj_data['data'][:, :32, 3 * S_RATE:]
        subj_label = subj_data['labels']

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


def divide_dataset(signals, labels, dataset_type, scheme):
    S_RATE = 128

    if 'time' in scheme:
        if scheme == 'time':
            # 00~20: training, 20~25: validation, 25~30: test
            # 30~50: training, 50~55: validation, 55~60: test
            # 3 seconds window, 2 seconds overlap
            # 34 training segments, 6 test segments
            t1, t2, t3, t4, t5, t6 = 20, 25, 30, 50, 55, 60
            window = 3
            overlap = 2
        elif scheme == 'time_nooverlap_w3':
            # 00~24: training, 24~27: validation, 27~30: test
            # 30~54: training, 54~57: validation, 57~60: test
            # 3 seconds window, 0 seconds overlap
            # 16 training segments, 2 valid segments, 2 test
            t1, t2, t3, t4, t5, t6 = 24, 27, 30, 54, 57, 60
            window = 3
            overlap = 0
        elif scheme == 'time_nooverlap_w2':
            # 00~22: training, 22~26: validation, 26~30: test
            # 30~52: training, 52~56: validation, 56~60: test
            # 2 seconds window, 0 seconds overlap
            # 22 training segments, 4 valid segments, 4 test
            t1, t2, t3, t4, t5, t6 = 22, 26, 30, 52, 56, 60
            window = 2
            overlap = 0

        train_s1, train_l1 = divide_signal(signals[..., 0 * S_RATE:t1 * S_RATE],
                                           labels, S_RATE, window, overlap,
                                           axis=2)
        valid_s1, valid_l1 = divide_signal(
            signals[..., t1 * S_RATE:t2 * S_RATE], labels, S_RATE, window,
            overlap, axis=2)
        test_s1, test_l1 = divide_signal(signals[..., t2 * S_RATE:t3 * S_RATE],
                                         labels, S_RATE, window, overlap,
                                         axis=2)

        train_s2, train_l2 = divide_signal(
            signals[..., t3 * S_RATE:t4 * S_RATE], labels, S_RATE, window,
            overlap, axis=2)
        valid_s2, valid_l2 = divide_signal(
            signals[..., t4 * S_RATE:t5 * S_RATE], labels, S_RATE, window,
            overlap, axis=2)
        test_s2, test_l2 = divide_signal(signals[..., t5 * S_RATE:t6 * S_RATE],
                                         labels, S_RATE, window, overlap,
                                         axis=2)

        # n_win, n_subj, n_trial, n_electrode, win_length
        # n_win, n_subj, n_trial, feats

        train_signals = np.concatenate((train_s1, train_s2), axis=2)
        valid_signals = np.concatenate((valid_s1, valid_s2), axis=2)
        test_signals = np.concatenate((test_s1, test_s2), axis=2)

        train_labels = np.concatenate((train_l1, train_l2), axis=2)
        valid_labels = np.concatenate((valid_l1, valid_l2), axis=2)
        test_labels = np.concatenate((test_l1, test_l2), axis=2)
    elif scheme == 'subject':
        if dataset_type == 'deap':
            train_signals = signals[0:24]
            train_labels = labels[0:24]
            valid_signals = signals[24:28]
            valid_labels = labels[24:28]
            test_signals = signals[28:32]
            test_labels = labels[28:32]
        elif dataset_type == 'dreamer':
            train_signals = signals[0:17]
            train_labels = labels[0:17]
            valid_signals = signals[17:20]
            valid_labels = labels[17:20]
            test_signals = signals[20:23]
            test_labels = labels[20:23]

        # DEAP subject scheme
        # DEAP: 24, 4, 4 (0:24, 24:28, 28:32)
        # DREAMER: 17, 3, 3 (0:17, 17:20, 20:23)
        train_signals, train_labels = divide_signal(train_signals, train_labels,
                                                    S_RATE, 3, 2, axis=2)
        valid_signals, valid_labels = divide_signal(valid_signals, valid_labels,
                                                    S_RATE, 3, 2, axis=2)
        test_signals, test_labels = divide_signal(test_signals, test_labels,
                                                  S_RATE, 3, 2, axis=2)

    return (train_signals, valid_signals,
            test_signals), (train_labels, valid_labels, test_labels)


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


def construct_lmdb_dataset(input_path, output_path, dataset_type,
                           scheme='time'):

    if dataset_type == 'deap':
        signals_raw, labels_raw, label_name = _load_raw_data_deap(input_path)
        MAP_SIZE_MULTIPLER = 3
    elif dataset_type == 'dreamer':
        signals_raw, labels_raw, label_name = _load_raw_data_dreamer(input_path)
        MAP_SIZE_MULTIPLER = 3

    print('-----Signal Statistics-----')
    signals_mean = np.mean(signals_raw)
    signals_std = np.std(signals_raw)
    print('Mean: {}, Stddev: {}'.format(signals_mean, signals_std))
    # signals_raw = (signals_raw - signals_mean) / signals_std

    signals_divided, labels_divided = divide_dataset(signals_raw, labels_raw,
                                                     dataset_type, scheme)
    train_signal, valid_signal, test_signal = signals_divided
    train_labels, valid_labels, test_labels = labels_divided

    print('-----Processed Data Shape-----')
    print('Train data: {}, label: {}'.format(str(train_signal.shape),
                                             str(train_labels.shape)))
    print('Valid data: {}, label: {}'.format(str(valid_signal.shape),
                                             str(valid_labels.shape)))
    print('Test data: {}, label: {}'.format(str(test_signal.shape),
                                            str(test_labels.shape)))

    d_size = train_signal.dtype.itemsize
    l_size = train_labels.dtype.itemsize

    train_map_size = train_signal.size * d_size + train_labels.size * l_size
    valid_map_size = valid_signal.size * d_size + valid_labels.size * l_size
    test_map_size = test_signal.size * d_size + test_labels.size * l_size

    train_map_size *= MAP_SIZE_MULTIPLER
    valid_map_size *= MAP_SIZE_MULTIPLER
    test_map_size *= MAP_SIZE_MULTIPLER

    train_root = os.path.join(output_path, 'train//')
    valid_root = os.path.join(output_path, 'valid//')
    test_root = os.path.join(output_path, 'test//')

    train_records = construct_lmdb(train_root, train_map_size, train_signal,
                                   train_labels, 'train')
    valid_records = construct_lmdb(valid_root, valid_map_size, valid_signal,
                                   valid_labels, 'valid')
    test_records = construct_lmdb(test_root, test_map_size, test_signal,
                                  test_labels, 'test')

    print('-----LMDB Records-----')
    print('#Train: {}, #Valid: {}, #Test: {}'.format(train_records,
                                                     valid_records,
                                                     test_records))


def test_lmdb_dataset(lmdb_root, batch_size=256):
    train_root = os.path.join(lmdb_root, 'train//')
    valid_root = os.path.join(lmdb_root, 'valid//')
    test_root = os.path.join(lmdb_root, 'test//')

    train_loader = data.DataLoader(EEGDataset(root=train_root, transform=None),
                                   batch_size=batch_size, shuffle=True)
    valid_loader = data.DataLoader(EEGDataset(root=valid_root, transform=None),
                                   batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(EEGDataset(root=test_root, transform=None),
                                  batch_size=batch_size, shuffle=True)

    train_mean = []
    for signal, label in tqdm(train_loader):
        train_mean.append(signal.mean())
    valid_mean = []
    for signal, label in tqdm(valid_loader):
        valid_mean.append(signal.mean())

    test_mean = []
    for signal, label in tqdm(test_loader):
        test_mean.append(signal.mean())

    print(np.mean(train_mean))
    print(np.mean(valid_mean))
    print(np.mean(test_mean))


def main(argv):
    input_path = None
    output_path = None
    dataset_type = 'deap'

    try:
        opts, args = getopt.getopt(
            argv, "hd:i:o:s:",
            ["dataset_type=", "input_path=", "output_path=", "scheme="])
    except getopt.GetoptError:
        print(
            'lmdb_generate.py -i <inputpath> -o <outputpath> -d <dataset_type> -s <scheme>'
        )
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(
                'lmdb_generate.py -i <inputpath> -o <outputpath> -d <dataset_type> -s <scheme>'
            )
            sys.exit()
        elif opt in ("-i", "--input_path"):
            input_path = arg
        elif opt in ("-o", "--output_path"):
            output_path = arg
        elif opt in ("-d", "--dataset_type"):
            dataset_type = arg
        elif opt in ("-s", "--scheme"):
            scheme = arg

    assert input_path is not None, 'Input path should be specified.'
    assert output_path is not None, 'Output path should be specified.'

    construct_lmdb_dataset(input_path, output_path, dataset_type, scheme)
    test_lmdb_dataset(output_path)


if __name__ == "__main__":
    # Construct LMDB dataset from DEAP.
    main(sys.argv[1:])
