import os

import numpy as np
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
