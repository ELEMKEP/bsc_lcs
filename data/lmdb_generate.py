import string
import pickle
import os
import sys
import getopt
import ast
import abc

import lmdb
import torch.utils.data as data
import scipy.io as sio
import numpy as np
from tqdm import tqdm

from dataset import *


def construct_deap_lmdb(data_path, lmdb_root, seed=42):
    train_map_size = 32 * (264 * (1024**2))
    valid_map_size = 32 * (33 * (1024**2))
    test_map_size = 32 * (33 * (1024**2))
    # Train 176MB, Valid 22MB, Test 22MB / subject
    # ~220MB for one subject samples
    # (228065280 + 1280 + @) for each subject

    train_root = os.path.join(lmdb_root, 'train//')
    valid_root = os.path.join(lmdb_root, 'valid//')
    test_root = os.path.join(lmdb_root, 'test//')

    train_env = lmdb.open(train_root, map_size=train_map_size)
    valid_env = lmdb.open(valid_root, map_size=valid_map_size)
    test_env = lmdb.open(test_root, map_size=test_map_size)

    data_index = 0
    train_index = 0
    valid_index = 0
    test_index = 0

    for file_idx in tqdm(range(32)):  # TODO: change to range(32)
        data_file = 'windowed_%02d.dat' % (file_idx + 1)
        label_file = 'label_%02d.dat' % (file_idx + 1)
        with open(data_path + data_file, 'rb') as f:
            signals = pickle.load(f)['signal'][:, :, ::2, :]

        with open(data_path + label_file, 'rb') as f:
            labels = pickle.load(f)['labels']
            # [#trial, 4 (#labels)]

        # shuffling signals in window domain
        np.random.seed(seed)
        window_idx = np.random.permutation(np.arange(signals.shape[2]))
        signals_s = signals[:, :, window_idx, :]
        signals_s = np.transpose(signals_s, [0, 2, 1, 3])

        for i in range(signals.shape[0]):
            signals_t = signals_s[i]
            label_t = labels[i]

            if data_index % 5 == 1:  # 47:6:5
                signals_train = signals_t[:47]
                signals_valid = signals_t[47:53]
                signals_test = signals_t[53:]
            elif data_index % 5 == 3:  # 47:5:6
                signals_train = signals_t[:47]
                signals_valid = signals_t[47:52]
                signals_test = signals_t[52:]
            else:  # data_index % 5 = 0, 2, 4 / 46:6:6
                signals_train = signals_t[:46]
                signals_valid = signals_t[46:52]
                signals_test = signals_t[52:]

            with train_env.begin(write=True) as txn:
                for j in range(signals_train.shape[0]):
                    signal = signals_train[j]
                    datum = EEGDatum(
                        dshape=np.asarray(signal.shape, dtype=np.int32),
                        ddata=signal.astype(np.float32),
                        lshape=np.asarray(label_t.shape, dtype=np.int32),
                        ldata=label_t.astype(np.float32), subject=file_idx + 1,
                        trial=i)

                    str_id = '{:06}'.format(train_index)
                    txn.put(str_id.encode('ascii'), datum.encode())
                    train_index += 1
                    data_index += 1

            with valid_env.begin(write=True) as txn:
                for j in range(signals_valid.shape[0]):
                    signal = signals_valid[j]
                    datum = EEGDatum(
                        dshape=np.asarray(signal.shape, dtype=np.int32),
                        ddata=signal.astype(np.float32),
                        lshape=np.asarray(label_t.shape, dtype=np.int32),
                        ldata=label_t.astype(np.float32), subject=file_idx + 1,
                        trial=i)

                    str_id = '{:06}'.format(valid_index)
                    txn.put(str_id.encode('ascii'), datum.encode())
                    valid_index += 1
                    data_index += 1

            with test_env.begin(write=True) as txn:
                for j in range(signals_test.shape[0]):
                    signal = signals_test[j]
                    datum = EEGDatum(
                        dshape=np.asarray(signal.shape, dtype=np.int32),
                        ddata=signal.astype(np.float32),
                        lshape=np.asarray(label_t.shape, dtype=np.int32),
                        ldata=label_t.astype(np.float32), subject=file_idx + 1,
                        trial=i)

                    str_id = '{:06}'.format(test_index)
                    txn.put(str_id.encode('ascii'), datum.encode())
                    test_index += 1
                    data_index += 1

    train_env.close()
    valid_env.close()
    test_env.close()


def construct_dreamer_lmdb(data_path, lmdb_root, seed=42):
    train_map_size = 23 * (56 * (1024**2))
    valid_map_size = 23 * (7 * (1024**2))
    test_map_size = 23 * (7 * (1024**2))
    # ~22.14MB for one subject sample --> 34MB alloc

    train_root = os.path.join(lmdb_root, 'train//')
    valid_root = os.path.join(lmdb_root, 'valid//')
    test_root = os.path.join(lmdb_root, 'test//')

    train_env = lmdb.open(train_root, map_size=train_map_size)
    valid_env = lmdb.open(valid_root, map_size=valid_map_size)
    test_env = lmdb.open(test_root, map_size=test_map_size)

    data_index = 0
    train_index = 0
    valid_index = 0
    test_index = 0

    for file_idx in tqdm(range(23)):  # TODO: change to range(32)
        data_file = 'subject_%d.mat' % (file_idx + 1)
        mat_data = sio.loadmat(os.path.join(data_path, data_file))
        signals = mat_data['data']  # [#trial, #channels, #window, #signals]
        valence = mat_data['valence']  # [#trial, 1]
        arousal = mat_data['arousal']
        dominance = mat_data['dominance']
        labels = np.concatenate((valence, arousal, dominance),
                                axis=1)  # [#trial, 3]

        # shuffling signals in window domain
        np.random.seed(seed)
        window_idx = np.random.permutation(np.arange(signals.shape[2]))
        signals_s = signals[:, :, window_idx, :]
        signals_s = np.transpose(
            signals_s, [0, 2, 1, 3])  # [#trial, #window, #channels, #signals]

        # 60 samples --> 48:6:6
        for i in range(signals.shape[0]):  # Trial-level
            signals_t = signals_s[i]  # [#window, #channels, #signals]
            label_t = labels[i]  # [3,]

            signals_train = signals_t[:48]
            signals_valid = signals_t[48:54]
            signals_test = signals_t[54:]

            with train_env.begin(write=True) as txn:
                for j in range(signals_train.shape[0]):
                    signal = signals_train[j]
                    datum = EEGDatum(
                        dshape=np.asarray(signal.shape, dtype=np.int32),
                        ddata=signal.astype(np.float32),
                        lshape=np.asarray(label_t.shape, dtype=np.int32),
                        ldata=label_t.astype(np.float32), subject=file_idx + 1,
                        trial=i)

                    str_id = '{:06}'.format(train_index)
                    txn.put(str_id.encode('ascii'), datum.encode())
                    train_index += 1
                    data_index += 1

            with valid_env.begin(write=True) as txn:
                for j in range(signals_valid.shape[0]):
                    signal = signals_valid[j]
                    datum = EEGDatum(
                        dshape=np.asarray(signal.shape, dtype=np.int32),
                        ddata=signal.astype(np.float32),
                        lshape=np.asarray(label_t.shape, dtype=np.int32),
                        ldata=label_t.astype(np.float32), subject=file_idx + 1,
                        trial=i)

                    str_id = '{:06}'.format(valid_index)
                    txn.put(str_id.encode('ascii'), datum.encode())
                    valid_index += 1
                    data_index += 1

            with test_env.begin(write=True) as txn:
                for j in range(signals_test.shape[0]):
                    signal = signals_test[j]
                    datum = EEGDatum(
                        dshape=np.asarray(signal.shape, dtype=np.int32),
                        ddata=signal.astype(np.float32),
                        lshape=np.asarray(label_t.shape, dtype=np.int32),
                        ldata=label_t.astype(np.float32), subject=file_idx + 1,
                        trial=i)

                    str_id = '{:06}'.format(test_index)
                    txn.put(str_id.encode('ascii'), datum.encode())
                    test_index += 1
                    data_index += 1

    train_env.close()
    valid_env.close()
    test_env.close()


def test_lmdb_dataset_v2(lmdb_root, batch_size=128):
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
    dataset_type = 'deap_v2'

    try:
        opts, args = getopt.getopt(
            argv, "hd:i:o:", ["dataset_type=", "input_path=", "output_path="])
    except getopt.GetoptError:
        print('dataset_lmdb.py -i <inputpath> -o <outputpath>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('dataset_lmdb.py -i <inputpath> -o <outputpath>')
            sys.exit()
        elif opt in ("-i", "--input_path"):
            input_path = arg
        elif opt in ("-o", "--output_path"):
            output_path = arg
        elif opt in ("-d", "--dataset_type"):
            dataset_type = arg

    assert input_path is not None, 'Input path should be specified.'
    assert output_path is not None, 'Output path should be specified.'

    if dataset_type == "deap":
        print('Dataset_type: DEAP')
        construct_deap_lmdb(data_path=input_path, lmdb_root=output_path)
        print('Test dataset...')
        test_lmdb_dataset_v2(lmdb_root=output_path)
    elif dataset_type == 'dreamer':
        print('Dataset type: DREAMER')
        construct_dreamer_lmdb(data_path=input_path, lmdb_root=output_path)
        print('Test dataset...')
        test_lmdb_dataset_v2(lmdb_root=output_path)


if __name__ == "__main__":
    # Construct LMDB dataset from DEAP.
    main(sys.argv[1:])
