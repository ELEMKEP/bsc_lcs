import os
import time
import datetime
import pickle
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

from tqdm import tqdm
from tensorboardX import SummaryWriter
import sklearn.svm, sklearn.neighbors, sklearn.neighbors

import arguments_traditional
import utils.utils_data
from utils.utils_data import *


def _construct_loaders(args):

    def transform(datum):
        # Data
        if args.dataset == 'deap':
            if args.data == 'raw':
                data_t = transform_deap_data_raw(datum)
            elif args.data == 'specent':
                data_t = transform_eeg_specent(datum)
            elif args.data == 'de':
                data_t = transform_data_de(datum, n_band=5)
        elif args.dataset == 'dreamer':
            if args.data == 'raw':
                data_t = transform_dreamer_data_raw(datum)
            elif args.data == 'specent':
                data_t = transform_eeg_specent(datum)
            elif args.data == 'de':
                data_t = transform_data_de(datum, n_band=5)

        # Label
        label_func_str = f'transform_{args.dataset}_label_{args.label}'
        label_func = getattr(utils.utils_data, label_func_str)
        label_t = label_func(datum)

        return data_t, label_t

    loaders_kfold = load_lmdb_kfold_dataset(lmdb_root=args.data_path,
                                            batch_size=args.batch_size,
                                            transform=transform, shuffle=True)

    return loaders_kfold


def _construct_model(args):

    if args.model == 'svm':
        # --lr: C-value in SVC
        model = sklearn.svm.LinearSVC(C=args.lr)
        param_dict = {'C': str(args.lr)}
    elif args.model == 'knn':
        # --encoder-hidden: k-value in KNN
        n_neighbors = args.encoder_hidden
        model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbors,
                                                       n_jobs=8)
        param_dict = {'k': str(n_neighbors)}
    elif args.model == 'rf':
        # --encoder-hidden: n_estimator
        # --encoder-heads: max_depth (None if -1)
        # --decoder-hidden: min_samples_split (None if -1)
        n_estimator = args.n_estimator
        max_depth = None if args.max_depth == -1 else args.max_depth
        min_samples_split = 2 if args.min_samples_split == -1 else args.min_samples_spli

        model = sklearn.ensemble.RandomForestClassifier(
            n_estimators=n_estimator, max_depth=max_depth,
            min_samples_split=min_samples_split, n_jobs=8, verbose=10,
            random_state=args.seed)
        param_dict = {
            'n_estimator': str(n_estimator),
            'max_depth': str(max_depth),
            'min_samples_split': str(min_samples_split)
        }

    return model, param_dict


def sklearn_baseline(clf, train_data, train_labels, test_data, test_labels,
                     args):
    """Train various classifiers to get a baseline."""

    t_start = time.process_time()
    clf.fit(train_data, train_labels)
    train_pred = clf.predict(train_data)
    test_pred = clf.predict(test_data)

    train_acc = 100 * sklearn.metrics.accuracy_score(train_labels, train_pred)
    test_acc = 100 * sklearn.metrics.accuracy_score(test_labels, test_pred)
    train_f1 = 100 * sklearn.metrics.f1_score(train_labels, train_pred,
                                              average='weighted')
    test_f1 = 100 * sklearn.metrics.f1_score(test_labels, test_pred,
                                             average='weighted')
    exec_time = time.process_time() - t_start

    with redirect_stdout(open(args.out, 'a')):
        print('Train accuracy:      {:5.2f}'.format(train_acc))
        print('Test accuracy:       {:5.2f}'.format(test_acc))
        print('Train F1 (weighted): {:5.2f}'.format(train_f1))
        print('Test F1 (weighted):  {:5.2f}'.format(test_f1))
        print('Execution time:      {}'.format(exec_time))

    return train_acc, test_acc, train_f1, test_f1, train_pred, test_pred, exec_time


def main():
    """
    Main function for training.
    """
    args = arguments_traditional.parse()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    param_dict = dict()

    # Save model and meta-data. Always saves in a new sub-folder.
    if args.save_folder:
        # Define log folder and make the directory
        timestamp = datetime.datetime.now().isoformat().replace(':', '-')
        save_folder = '{}/{}_{}/'.format(args.save_folder, args.out, timestamp)
        Path(save_folder).mkdir(parents=True, exist_ok=True)

        meta_file = os.path.join(save_folder, 'metadata.pkl')
        model_file = os.path.join(save_folder, 'model.pt')
        log = open(os.path.join(save_folder, 'log.txt'), 'w')

        pickle.dump({'args': vars(args)}, open(meta_file, "wb"))

        param_dict.update({
            'save_folder': save_folder,
            'model_file': model_file,
            'log': log
        })
    else:
        print("WARNING: No save_folder provided!" +
              "Testing (within this script) will throw an error.")
        log = open(args.out, 'w')

    # TensorboardX support
    if args.tensorboard:
        import socket
        if not args.save_folder:
            tb_log_dir = 'logs/tensorboard'
        else:
            tb_log_dir = save_folder
        writer = SummaryWriter(logdir=tb_log_dir)
        param_dict['writer'] = writer

    # Load dataset
    loaders_kfold = _construct_loaders(args)

    # Loop saving
    train_acc_list = []
    test_acc_list = []
    train_f1_list = []
    test_f1_list = []
    exec_time_list = []
    for kf, (train_loader, test_loader) in enumerate(loaders_kfold):
        loaders = (train_loader, test_loader)

        train_data = []
        train_label = []
        for (data, label) in tqdm(train_loader, desc='train'):
            train_data.append(data.squeeze().numpy())
            train_label.append(label.argmax(dim=1, keepdim=False).numpy())

        test_data = []
        test_label = []
        for (data, label) in tqdm(test_loader, desc='test'):
            test_data.append(data.squeeze().numpy())
            test_label.append(label.argmax(dim=1, keepdim=False).numpy())

        train_data = np.concatenate(train_data, axis=0)
        train_label = np.concatenate(train_label, axis=0)
        test_data = np.concatenate(test_data, axis=0)
        test_label = np.concatenate(test_label, axis=0)

        train_data = train_data.reshape(train_data.shape[0], -1)
        train_label = train_label.reshape(train_label.shape[0],)
        test_data = test_data.reshape(test_data.shape[0], -1)
        test_label = test_label.reshape(test_label.shape[0],)

        model, param_add = _construct_model(args)

        param_dict.update({
            'loaders': loaders,
            'model': model,
            'fold': kf,
        })
        param_dict.update(param_add)

        output = sklearn_baseline(model, train_data, train_label, test_data,
                                  test_label, args)
        train_acc, test_acc, train_f1, test_f1, train_pred, test_pred, exec_time = output

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        train_f1_list.append(train_f1)
        test_f1_list.append(test_f1)
        exec_time_list.append(exec_time)

    train_acc_mean = np.mean(train_acc_list)
    test_acc_mean = np.mean(test_acc_list)
    train_f1_mean = np.mean(train_f1_list)
    test_f1_mean = np.mean(test_f1_list)
    exec_time_mean = np.mean(exec_time_list)

    if writer is not None:
        metric_dict = {
            'Train Accuracy': '{:8.6f}'.format(train_acc_mean),
            'Test Accuracy': '{:8.6f}'.format(test_acc_mean),
            'Train F1': '{:8.6f}'.format(train_f1_mean),
            'Test F1': '{:8.6f}'.format(test_f1_mean),
            'Dataset': str(args.data),
            'Elapsed time': f'{str(exec_time_mean):8.2f}s'
        }
        metric_dict.update(param_dict)

        writer.add_text('metrics', str(metric_dict), 0)
        writer.close()
    log.close()


if __name__ == "__main__":
    main()
