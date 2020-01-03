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
    N_SUBJ = 23
    N_TRIAL = 18
    S_RATE = 128
    V_LENGTH = 60

    IDX_EEG = 2
    IDX_VAL = 4
    IDX_ARS = 5
    IDX_DOM = 6

    mat_data = sio.loadmat(os.path.join(data_path, 'DREAMER.mat'))
    dreamer_data = mat_data['DREAMER'][0][0][0][0]

    eeg_list = []
    label_list = []
    for i in tqdm(range(N_SUBJ), desc='Loading data - subject '):
        subj_data = dreamer_data[i][0][0]
        subj_eeg = subj_data[IDX_EEG]
        subj_val = subj_data[IDX_VAL]
        subj_ars = subj_data[IDX_ARS]
        subj_dom = subj_data[IDX_DOM]

        subj_eeg_list = []
        subj_label_list = []
        for j in tqdm(range(N_TRIAL), desc='Loading data - video '):
            sv_eeg = subj_eeg[0][0][1][j][0][0:V_LENGTH * S_RATE]
            # size: (length x #electrode)
            labels = np.array([subj_val[j], subj_ars[j], subj_dom[j]])

            subj_eeg_list.append(np.transpose(sv_eeg))
            subj_label_list.append(np.squeeze(labels))
        subj_eeg_list = np.stack(subj_eeg_list, axis=0)
        # N_TRIAL, N_ELECTRODE, V_LENGTH * S_RATE
        subj_label_list = np.stack(subj_label_list, axis=0)
        # N_TRIAL, 3(val, ars, dom)

        eeg_list.append(subj_eeg_list)
        label_list.append(subj_label_list)
    eeg_list = np.stack(eeg_list, axis=0)
    label_list = np.stack(label_list, axis=0)

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


def divide_dataset(signals, labels):
    S_RATE = 128

    signals, labels = divide_signal(signals, labels, S_RATE, 3, 2, axis=0)

    signals = signals.transpose(1, 2, 0, 3, 4)
    labels = labels.transpose(1, 2, 0, 3)
    sh = signals.shape
    # Subject, Trial, Window, Object, Length (for signals)
    # Subject, Trial, Window, Feat (for labels)

    n_sub = sh[0]
    n_tri = sh[1]
    n_win = sh[2]
    n_sample = n_sub * n_tri * n_win
    subjects = np.zeros([n_sub, n_tri, n_win], dtype=np.int32)
    trials = np.zeros([n_sub, n_tri, n_win], dtype=np.int32)

    for s in range(n_sub):
        for t in range(n_tri):
            subjects[s, t, :] = s
            trials[s, t, :] = t

    signals = signals.reshape(n_sample, sh[3], sh[4])
    labels = labels.reshape(n_sample, labels.shape[-1])
    subjects = subjects.reshape(-1)
    trials = trials.reshape(-1)

    perm_idx = np.random.permutation(np.arange(n_sample))
    signals = signals[perm_idx]
    labels = labels[perm_idx]
    subjects = subjects[perm_idx]
    trials = trials[perm_idx]

    sample_10p = int(np.floor(n_sample * 0.1))
    train_cursor = n_sample - (2 * sample_10p)
    valid_cursor = n_sample - (1 * sample_10p)

    train_signals = signals[:train_cursor]
    train_labels = labels[:train_cursor]
    train_sub = subjects[:train_cursor]
    train_tri = trials[:train_cursor]

    valid_signals = signals[train_cursor:valid_cursor]
    valid_labels = labels[train_cursor:valid_cursor]
    valid_sub = subjects[train_cursor:valid_cursor]
    valid_tri = trials[train_cursor:valid_cursor]

    test_signals = signals[valid_cursor:]
    test_labels = labels[valid_cursor:]
    test_sub = subjects[valid_cursor:]
    test_tri = trials[valid_cursor:]

    sig_t = (train_signals, valid_signals, test_signals)
    lab_t = (train_labels, valid_labels, test_labels)
    sub_t = (train_sub, valid_sub, test_sub)
    tri_t = (train_tri, valid_tri, test_tri)

    return sig_t, lab_t, sub_t, tri_t


def construct_lmdb(lmdb_root, lmdb_size, signals, labels, sub, tri, tag=''):
    lmdb_env = lmdb.open(lmdb_root, map_size=lmdb_size)

    n_all = signals.shape[0]

    data_index = 0
    with lmdb_env.begin(write=True) as txn:
        for idx in tqdm(range(n_all), desc='Constructing LMDB: ' + tag):
            signal_t = signals[idx]
            label_t = labels[idx]
            datum = EEGDatum(dshape=np.asarray(signal_t.shape, dtype=np.int32),
                             ddata=signal_t.astype(np.float32),
                             lshape=np.asarray(label_t.shape, dtype=np.int32),
                             ldata=label_t.astype(np.float32), subject=sub[idx],
                             trial=tri[idx])

            str_id = '{:06}'.format(data_index)
            txn.put(str_id.encode('ascii'), datum.encode())
            data_index += 1

    lmdb_env.close()

    return data_index


def construct_lmdb_dataset(input_path, output_path, dataset_type):
    MAP_SIZE_MULTIPLER = 1.5

    if dataset_type == 'deap':
        signals_raw, labels_raw, label_name = _load_raw_data_deap(input_path)
    elif dataset_type == 'dreamer':
        signals_raw, labels_raw, label_name = _load_raw_data_dreamer(input_path)

    print('-----Signal Statistics-----')
    signals_mean = np.mean(signals_raw)
    signals_std = np.std(signals_raw)
    print('Mean: {}, Stddev: {}'.format(signals_mean, signals_std))
    signals_raw = (signals_raw - signals_mean) / signals_std

    signals_divided, labels_divided, sub_divided, tri_divided = divide_dataset(
        signals_raw, labels_raw)
    train_signal, valid_signal, test_signal = signals_divided
    train_labels, valid_labels, test_labels = labels_divided
    train_sub, valid_sub, test_sub = sub_divided
    train_tri, valid_tri, test_tri = tri_divided

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
                                   train_labels, train_sub, train_tri, 'train')
    valid_records = construct_lmdb(valid_root, valid_map_size, valid_signal,
                                   valid_labels, valid_sub, valid_tri, 'valid')
    test_records = construct_lmdb(test_root, test_map_size, test_signal,
                                  test_labels, test_sub, test_tri, 'test')

    print('-----LMDB Records-----')
    print('#Train: {}, #Valid: {}, #Test: {}'.format(train_records,
                                                     valid_records,
                                                     test_records))


def test_lmdb_dataset(lmdb_root, batch_size=128):
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

    assert input_path is not None, 'Input path should be specified.'
    assert output_path is not None, 'Output path should be specified.'

    construct_lmdb_dataset(input_path, output_path, dataset_type)
    test_lmdb_dataset(output_path)


if __name__ == "__main__":
    # Construct LMDB dataset from DEAP.
    main(sys.argv[1:])
