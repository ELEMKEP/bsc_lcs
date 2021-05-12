import os

import numpy as np
import scipy
import scipy.signal
import torch
from torch.utils.data import DataLoader

from data.dataset import EEGDataset


def load_lmdb_dataset(lmdb_root, dataset=EEGDataset, batch_size=128,
                      transform=None, shuffle=True, print_dataset=False):
    train_root = os.path.join(lmdb_root, 'train//')
    valid_root = os.path.join(lmdb_root, 'valid//')
    test_root = os.path.join(lmdb_root, 'test//')

    train_dataset = dataset(root=train_root, transform=transform)
    valid_dataset = dataset(root=valid_root, transform=transform)
    test_dataset = dataset(root=test_root, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                              shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if print_dataset:
        print(train_dataset)
        print(valid_dataset)
        print(test_dataset)

    return train_loader, valid_loader, test_loader
    # Library function for LMDBloading


def load_lmdb_kfold_dataset(lmdb_root, dataset=EEGDataset, batch_size=128,
                            transform=None, shuffle=True, print_dataset=False):
    lmdb_list = os.listdir(lmdb_root)

    for d in lmdb_list:
        fold_path = os.path.join(lmdb_root, d)

        train_root = os.path.join(fold_path, 'train/')
        test_root = os.path.join(fold_path, 'test/')

        train_dataset = dataset(root=train_root, transform=transform)
        test_dataset = dataset(root=test_root, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False)

        if print_dataset:
            print(train_dataset)
            print(test_dataset)

        yield train_loader, test_loader


def transform_deap_data_raw(datum):
    data_r = np.reshape(datum.ddata, datum.dshape)
    data_r = data_r / 32.1079  # mean is close to 0, global std
    data_r = torch.clamp(torch.FloatTensor(data_r[..., np.newaxis]), -10., 10.)
    return data_r


def transform_deap_label_video(datum):
    # type(datum) = deap_lmdb.DEAPDatum
    # label is video!
    labels_r = np.zeros([40], dtype=float)
    labels_r[datum.trial] = 1.
    label = torch.FloatTensor(labels_r)
    return label


def transform_dreamer_data_raw(datum):
    data_r = np.reshape(datum.ddata, datum.dshape)
    data_r = (data_r - 4251.808) / 197.3314  # global mean and std
    data_r = torch.clamp(torch.FloatTensor(data_r[..., np.newaxis]), -10., 10.)
    return data_r


def transform_dreamer_label_video(datum):
    labels_r = np.zeros([18], dtype=float)
    labels_r[datum.trial] = 1.
    label = torch.FloatTensor(labels_r)
    return label


def transform_deap_label_valence(datum):
    labels_r = np.reshape(datum.ldata, datum.lshape)
    label = [1, 0] if labels_r[0] >= 5 else [0, 1]

    # TODO: check whether 5 is valid
    return torch.FloatTensor(label)


def transform_deap_label_arousal(datum):
    labels_r = np.reshape(datum.ldata, datum.lshape)
    label = [1, 0] if labels_r[1] >= 5 else [0, 1]

    # TODO: check whether 5 is valid
    return torch.FloatTensor(label)


def transform_dreamer_label_valence(datum):
    labels_r = np.reshape(datum.ldata, datum.lshape)
    label = [1, 0] if labels_r[0] >= 3 else [0, 1]

    # TODO: check whether 3 is valid
    return torch.FloatTensor(label)


def transform_dreamer_label_arousal(datum):
    labels_r = np.reshape(datum.ldata, datum.lshape)
    label = [1, 0] if labels_r[1] >= 3 else [0, 1]

    # TODO: check whether 3 is valid
    return torch.FloatTensor(label)


def transform_chebnet_permutation(data, perm):
    N, T = data.size()

    perm = np.array(perm, dtype=np.int32)
    NN = len(perm)
    data_p = torch.zeros([NN, T], dtype=torch.float32)

    select_idx = np.where(perm < N)
    idx_dst = select_idx
    idx_src = perm[select_idx]

    data_p[idx_dst] = data[idx_src]

    return data_p


def transform_eeg_specent(datum, band_to_node=True):
    # input: [32, 384]
    data_r = np.reshape(datum.ddata, datum.dshape)

    f, Pxx = scipy.signal.welch(data_r, window='boxcar', fs=128.0, nperseg=384,
                                nfft=512, scaling='density')
    Pxx = np.divide(Pxx, np.sum(Pxx, axis=-1, keepdims=True))

    band_freqs = [[0, 4], [4, 7], [8, 10], [10, 12.5], [12.5, 16.5], [16.5, 20],
                  [21, 29], [29, 50]]
    # calculate spectral entropy for each band <<
    # https://www.mathworks.com/help/signal/ref/pentropy.html

    ent_list = []
    for low, high in band_freqs:
        in_freq = ((f >= low) * (f < high))
        inf_indices = np.nonzero(in_freq)

        bp = Pxx[:, inf_indices]  # [32, 384]
        abp = np.sum(bp, axis=-1, keepdims=True)  # [32, 1]
        power_dist = np.divide(bp, abp)  # [32, ??]

        ent = scipy.stats.entropy(power_dist, base=2, axis=-1)  # [32, 1]

        ent_list.append(ent)
    data_r = np.concatenate(ent_list, axis=1)  # [32, 8]

    if band_to_node:
        data_r = np.reshape(data_r, [-1, 1])
    data_r = torch.FloatTensor(data_r.astype(np.float32))

    return data_r  # [???, 1]