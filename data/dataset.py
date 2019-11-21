import os
import string
import pickle
import abc
import ast

import lmdb
import torch.utils.data as data
import numpy as np


class _BaseDatum(abc.ABC):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def encode(self):
        pass

    @abc.abstractmethod
    def decode(self, bytestring):
        pass

    @abc.abstractmethod
    def get_data(self):
        pass


class EEGDatum(_BaseDatum):
    # DSHAPE, DATA, LSHAPE, LABEL is mendatory.
    # SUBJ and TRIAL can be not specified.

    DSHAPE = 0
    DATA = 1
    LSHAPE = 2
    LABEL = 3
    SUBJ = 4
    TRIAL = 5

    def __init__(self, dshape=None, ddata=None, lshape=None, ldata=None,
                 subject=None, trial=None):
        self.dshape = dshape
        self.ddata = ddata
        self.lshape = lshape
        self.ldata = ldata

        if subject is None:
            subject = -1

        if trial is None:
            trial = -1

        self.subject = np.asarray(subject, dtype=np.int32)
        self.trial = np.asarray(trial, dtype=np.int32)

    def encode(self):
        if not self._valid_check():
            return None

        encode_tuple = (self.dshape.tostring(), self.ddata.tostring(),
                        self.lshape.tostring(), self.ldata.tostring(),
                        self.subject.tostring(), self.trial.tostring())
        encode_str = str(encode_tuple).encode('ascii')
        return encode_str

    def decode(self, bytestring):
        if bytestring is None:
            print('Invalid bytestring')
            return
        else:
            # decode_tuple = eval(bytestring.decode('ascii'))
            decode_tuple = ast.literal_eval(bytestring.decode('ascii'))

        self.dshape = np.fromstring(decode_tuple[EEGDatum.DSHAPE],
                                    dtype=np.int32)
        self.ddata = np.fromstring(decode_tuple[EEGDatum.DATA],
                                   dtype=np.float32)
        self.lshape = np.fromstring(decode_tuple[EEGDatum.LSHAPE],
                                    dtype=np.int32)
        self.ldata = np.fromstring(decode_tuple[EEGDatum.LABEL],
                                   dtype=np.float32)
        self.subject = np.fromstring(decode_tuple[EEGDatum.SUBJ],
                                     dtype=np.int32)
        self.trial = np.fromstring(decode_tuple[EEGDatum.TRIAL], dtype=np.int32)

    def get_data(self):
        if not self._valid_check():
            return None

        data_r = np.reshape(self.ddata, self.dshape)
        label_r = np.reshape(self.ldata, self.lshape)

        return data_r, label_r

    def _valid_check(self):
        if (self.dshape is None) or (self.ddata is None) or (
                self.lshape is None) or (self.ldata is None):
            print('All data should be filled')
            return False
        return True


class EEGDataset(data.Dataset):
    """ Dataset class for valence classification. """

    def __init__(self, root, transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform

        self.env = lmdb.open(root, max_readers=4, readonly=True, lock=False,
                             readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

        cache_name = '_cache_' + ''.join(
            c for c in root if c in string.ascii_letters)
        cache_file = os.path.join(root, cache_name)

        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key, _ in txn.cursor()]
            pickle.dump(self.keys, open(cache_file, "wb"))

    def __getitem__(self, index):
        signal, label = None, None

        env = self.env
        with env.begin(write=False) as txn:
            raw_datum = txn.get(self.keys[index])

            datum = EEGDatum()
            datum.decode(raw_datum)

            if self.transform is not None:
                output = self.transform(datum)
            else:
                signal, label = datum.get_data()
                signal = signal[..., np.newaxis]
                output = (signal, label)

        return output

    def __len__(self):
        return self.length

    def __repr__(self):
        return '{} ({}) / Entries: {}'.format(self.__class__.__name__,
                                              self.root, self.length)