import os
import datetime
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from tqdm import tqdm
from tensorboardX import SummaryWriter
import sklearn.svm, sklearn.neighbors, sklearn.neighbors

import arguments_traditional
from utils.utils_loss import label_accuracy, label_cross_entropy
import utils.utils_data
from utils.utils_data import *
from utils.utils_miscellaneous import get_coarsening_map
'''
Revision code for valence and arousal
'''

def _construct_loaders(args, perm):

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


def _construct_model(args, graphs):
    if args.label == 'video':
        if args.dataset == 'deap':
            decoder_out_dim = 40
        elif args.dataset == 'dreamer':
            decoder_out_dim = 18
    else:
        decoder_out_dim = 2

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

    return model

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
    graphs, perm = load_graph_file(args)
    loaders_kfold = _construct_loaders(args, perm)

    # Loop saving
    acc_test_list = []
    ent_test_list = []
    loss_test_list = []

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

        model = _construct_model(args)
        if args.cuda:
            model.cuda()

        optimizer, scheduler = _construct_optimizer(model, args)

        param_dict.update({
            'loaders': loaders,
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'fold': kf,
        })

        # Train model
        best_val_loss = np.inf
        best_epoch = 0
        for epoch in range(args.epochs):
            val_loss = train(epoch, best_val_loss, args, param_dict)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch

        with redirect_stdout(log):
            print(f'Fold {kf} - Best epoch: {best_epoch+1:04d}')
            if args.save_folder:
                print(f'Fold {kf} - Best epoch: {best_epoch+1:04d}', file=log)
                log.flush()

        acc_test, ent_test, loss_test = test(args, param_dict)

        acc_test_list.append(acc_test)
        ent_test_list.append(ent_test)
        loss_test_list.append(loss_test)

    acc_fold_m = np.mean(acc_test_list)
    ent_fold_m = np.mean(ent_test_list)
    loss_fold_m = np.mean(loss_test_list)

    metric_dict = {
        'accuracy': acc_fold_m,
        'loss': loss_fold_m,
        'list_accuracy': acc_test_list,
        'list_loss': loss_test_list
    }

    if (writer is not None):
        arg_dict = vars(args)

        key_list = [
            'dataset',
            'label',
            'epochs',
            'batch_size',
            'lr',
            'beta1',
            'beta2',
            'lr_decay',
            'gamma',
            'hidden',
            'dropout',
        ]
        filtered_arg_dict = {k: arg_dict[k] for k in key_list}

        writer.add_text('parameters', str(filtered_arg_dict), 0)
        writer.add_text('metrics', str(metric_dict), 0)
        writer.close()

    if log is not None:
        print(save_folder)
        log.close()


if __name__ == "__main__":
    main()
