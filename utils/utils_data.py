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
    data_r = torch.clamp(torch.FloatTensor(data_r[..., np.newaxis]), -100., 100.)
    return data_r

def transform_deap_data_raw_v1(datum):
    data_r = np.reshape(datum.ddata, datum.dshape)
    data_r = torch.clamp(torch.FloatTensor(data_r[..., np.newaxis]), -10., 10.)
    return data_r

def transform_deap_data_raw_v2(datum):
    data_r = np.reshape(datum.ddata, datum.dshape)
    data_r = torch.clamp(torch.FloatTensor(data_r[..., np.newaxis]), -100., 100.)
    return data_r

def transform_deap_data_raw_v3(datum):
    data_r = np.reshape(datum.ddata, datum.dshape)
    data_r = data_r / 32.1079  # mean is close to 0, global std
    data_r = torch.clamp(torch.FloatTensor(data_r[..., np.newaxis]), -10., 10.)
    return data_r

def transform_deap_data_raw_v4(datum):
    data_r = np.reshape(datum.ddata, datum.dshape)
    m = np.array([ 0.0159, -0.0085, -0.0240,  0.0077, -0.0153, -0.0100, -0.0396,  0.0149,
         0.0286, -0.0058, -0.0032,  0.0044,  0.0020, -0.0032, -0.0078,  0.0144,
         0.0346,  0.0176, -0.0192,  0.0289, -0.0049, -0.0253,  0.0155, -0.0101,
        -0.0226,  0.0297, -0.0172, -0.0121, -0.0072,  0.0010,  0.0098,  0.0111])
    s = np.array([28.7735, 40.3026, 37.2894, 51.0681, 35.1658, 25.5328, 28.3683, 24.5622,
        33.4079, 21.4483, 18.7165, 25.9457, 26.8693, 41.1877, 34.8378, 20.2221,
        45.2038, 37.7310, 33.4042, 38.3248, 30.2907, 31.5689, 33.2135, 30.2887,
        29.5745, 36.2773, 31.3165, 23.2145, 29.4930, 20.3454, 35.4327, 24.4941])

    data_r = data_r.transpose()
    data_r = (data_r - m) / s 
    data_r = data_r.transpose()

    data_r = torch.clamp(torch.FloatTensor(data_r[..., np.newaxis]), -100., 100.)
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
    data_r = torch.clamp(torch.FloatTensor(data_r[..., np.newaxis]), -100., 100.)
    return data_r


def transform_dreamer_data_raw_v1(datum):
    data_r = np.reshape(datum.ddata, datum.dshape)
    data_r = (data_r - 4251.808) / 197.3314  # global mean and std
    data_r = torch.FloatTensor(data_r[..., np.newaxis])
    # data_r = torch.clamp(torch.FloatTensor(data_r[..., np.newaxis]), -100., 100.)
    return data_r

def transform_dreamer_data_raw_v2(datum):
    data_r = np.reshape(datum.ddata, datum.dshape)
    data_r = (data_r - 4251.808) / 197.3314  # global mean and std
    data_r = torch.clamp(torch.FloatTensor(data_r[..., np.newaxis]), -100., 100.)
    return data_r

def transform_dreamer_data_raw_v3(datum):
    data_r = np.reshape(datum.ddata, datum.dshape)
    data_r = (data_r - 4251.808) # / 197.3314  # global mean and std
    data_r = torch.FloatTensor(data_r[..., np.newaxis])
    # data_r = torch.clamp(torch.FloatTensor(data_r[..., np.newaxis]), -100., 100.)
    return data_r

def transform_dreamer_data_raw_v4(datum):
    data_r = np.reshape(datum.ddata, datum.dshape)
    data_r = (data_r - 4251.808) # / 197.3314  # global mean and std
    data_r = torch.clamp(torch.FloatTensor(data_r[..., np.newaxis]), -100., 100.)
    return data_r

def transform_dreamer_data_raw_v5(datum):
    data_r = np.reshape(datum.ddata, datum.dshape)
    m = np.array([4375.7383, 4100.0757, 4164.5312, 4409.0967, 4306.3530, 4360.1978,
        4443.3169, 3944.3589, 4304.4380, 4267.0244, 3934.6753, 4448.5601,
        4293.9170, 4172.0767])
    s = np.array([135.1736, 143.7671, 111.2396, 114.5694, 149.6347,  57.3782,  89.6525,
         91.5363,  77.9824, 159.8216, 162.0061, 171.7061, 160.3590, 201.1384])

    data_r = data_r.transpose()
    data_r = (data_r - m) / s 
    data_r = data_r.transpose()

    data_r = torch.FloatTensor(data_r[..., np.newaxis])
    # data_r = torch.clamp(torch.FloatTensor(data_r[..., np.newaxis]), -100., 100.)
    return data_r

def transform_dreamer_data_raw_v6(datum):
    data_r = np.reshape(datum.ddata, datum.dshape)
    m = np.array([4375.7383, 4100.0757, 4164.5312, 4409.0967, 4306.3530, 4360.1978,
        4443.3169, 3944.3589, 4304.4380, 4267.0244, 3934.6753, 4448.5601,
        4293.9170, 4172.0767])
    s = np.array([135.1736, 143.7671, 111.2396, 114.5694, 149.6347,  57.3782,  89.6525,
         91.5363,  77.9824, 159.8216, 162.0061, 171.7061, 160.3590, 201.1384])

    data_r = data_r.transpose()
    data_r = (data_r - m) / s 
    data_r = data_r.transpose()
    
    data_r = torch.clamp(torch.FloatTensor(data_r[..., np.newaxis]), -100., 100.)
    return data_r


def transform_dreamer_label_video(datum):
    labels_r = np.zeros([18], dtype=float)
    labels_r[datum.trial] = 1.
    label = torch.FloatTensor(labels_r)
    return label
