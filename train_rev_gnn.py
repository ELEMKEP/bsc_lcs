import os
import time
import pickle
import datetime
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from tqdm import tqdm
from tensorboardX import SummaryWriter

import arguments_gnn
from model.module_dgcnn import DGCNN, DGCNN_V2, DGCNN_V2_Reverse
from model.module_chebnet import DEAP_ChebNet
from model.module_stgcn import STGCN
from utils.utils_loss import label_accuracy, label_cross_entropy
from utils.utils_data import *
'''
Revision code for valence and arousal
'''


def load_graph_file(args):
    if args.model in ['chebnet']:
        band_str = 'B8_coarsen'
    elif args.pre_graph_expand:
        band_str = 'B8'
    else:
        band_str = 'B1'

    binarize_str = '_binarize' if args.pre_graph_binarized else ''
    conn_str = 'K' + (str(args.pre_graph_k) if args.pre_graph_k > 0 else 'all')
    graph_file = '{}_{}_{}_{}_{}{}.dat'.format(args.pre_graph_prefix,
                                               args.dataset,
                                               args.pre_graph_type, conn_str,
                                               band_str, binarize_str)

    with open(os.path.join(args.pre_graph_path, graph_file), 'rb') as f:
        graph_file = pickle.load(f)
    graphs = graph_file['graphs']
    perm = graph_file['perms']

    return graphs, perm


def _construct_loaders(args, perm):

    def transform(datum):
        # Data
        if args.model in ['chebnet']:
            data_t = transform_eeg_specent(datum)
            data_t = transform_chebnet_permutation(data_t, perm)
        else:
            if args.dataset == 'deap':
                data_t = transform_deap_data_raw(datum)
                if args.label == 'video':
                    label_t = transform_deap_label_video(datum)
            elif args.dataset == 'dreamer':
                data_t = transform_dreamer_data_raw(datum)
                if args.label == 'video':
                    label_t = transform_dreamer_label_video(datum)

        # Label
        if args.dataset == 'deap':
            if args.label == 'valence':
                label_t = transform_deap_label_valence(datum)
            elif args.label == 'arousal':
                label_t = transform_deap_label_arousal(datum)
        elif args.dataset == 'dreamer':
            if args.label == 'valence':
                label_t = transform_dreamer_label_valence(datum)
            elif args.label == 'arousal':
                label_t = transform_dreamer_label_arousal(datum)

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

    F_in = args.timesteps

    if 'dgcnn' in args.model:
        if args.model == 'dgcnn':
            model_type = DGCNN
        elif args.model == 'dgcnn_v2':
            model_type = DGCNN_V2
        elif args.model == 'dgcnn_v2_rev':
            model_type = DGCNN_V2_Reverse
        model = model_type(F_in, args.hidden, decoder_out_dim, graphs[0], 4,
                           args.cuda, True)
    elif args.model == 'chebnet':
        MM = [512, decoder_out_dim]
        FF = [32, 64]
        KK = [4, 9]
        PP = [2, 2]

        model = DEAP_ChebNet(graphs, FF, KK, PP, MM, Fin=args.dims)
    elif 'stgcn' in args.model:
        # model keyword should be stgcn_<size>. <size> = ['full', 'medium', 'small']
        model_split = args.model.split('_')
        model = STGCN(1, decoder_out_dim, graphs[0], False,
                      model_size=model_split[1])

    if args.load_folder:
        model_file = os.path.join(args.load_folder, 'model.pt')
        model.load_state_dict(torch.load(model_file))

        args.save_folder = False

    return model


def _construct_optimizer(model, args):
    if 'dgcnn' in args.model:
        params = list(model.layer.parameters()) + list(model.fc.parameters())
    else:
        params = list(model.parameters())

    optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                    gamma=args.gamma)

    return optimizer, scheduler


def _calculate_l1(model, args):
    if 'dgcnn' in args.model:
        params = list(model.layer.parameters()) + list(model.fc.parameters())
    else:
        params = list(model.parameters())

    l1 = 0

    for param in params:
        l1 += torch.norm(param, p=args.reg_order)

    return l1 * args.reg


def _make_cuda(models, tensors, args):
    if args.cuda:
        for model in models:
            model.cuda()

        cuda_tensors = []
        for tensor in tensors:
            cuda_tensors.append(tensor.cuda())

    return tuple(cuda_tensors)


def train(epoch, best_val_loss, args, params):
    """
    Main train function for a single epoch.
    """
    train_loader, test_loader = params['loaders']
    model = params['model']
    optimizer = params['optimizer']
    scheduler = params['scheduler']

    model_file = params['model_file']
    writer = params.get('writer', None)
    use_tensorboard = (writer is not None)

    t_start = time.time()

    acc_train = []
    ent_train = []
    loss_train = []

    model.train()
    scheduler.step()

    train_iterable = tqdm(train_loader)
    for index, (data, labels) in enumerate(train_iterable):
        train_iterable.set_description("Epoch{:3} Train".format(epoch + 1))

        # data = data[:, :, :args.timesteps, :]
        if args.cuda:
            data = data.cuda()
            labels = labels.cuda()
        optimizer.zero_grad()

        output = model(data)

        loss_ent = label_cross_entropy(output, labels)
        label_acc = label_accuracy(output, labels)
        loss_reg = _calculate_l1(model, args)
        loss = loss_ent + loss_reg

        loss.backward()
        optimizer.step()

        acc_train.append(label_acc.detach().item())
        ent_train.append(loss_ent.detach().item())
        loss_train.append(loss.detach().item())

    acc_test = []
    ent_test = []
    loss_test = []

    model.eval()

    test_iterable = tqdm(test_loader)
    for index, (data, labels) in enumerate(test_iterable):
        test_iterable.set_description("Epoch{:3} Valid".format(epoch + 1))

        # data = data[:, :, :args.timesteps, :]
        if args.cuda:
            data = data.cuda()
            labels = labels.cuda()

        with torch.no_grad():
            output = model(data)
            loss_ent = label_cross_entropy(output, labels)
            label_acc = label_accuracy(output, labels)
            loss_reg = _calculate_l1(model, args)
            loss = loss_ent + loss_reg

        acc_test.append(label_acc.detach().item())
        ent_test.append(loss_ent.detach().item())
        loss_test.append(loss.detach().item())

    acc_train_m = np.mean(acc_train)
    ent_train_m = np.mean(ent_train)
    loss_train_m = np.mean(loss_train)

    acc_test_m = np.mean(acc_test)
    ent_test_m = np.mean(ent_test)
    loss_test_m = np.mean(loss_test)

    if use_tensorboard:
        writer.add_scalar('train/Accuracy', acc_train_m, epoch + 1)
        writer.add_scalar('train/CrossEntropy', ent_train_m, epoch + 1)
        writer.add_scalar('train/TotalLoss', loss_train_m, epoch + 1)

        writer.add_scalar('test/Accuracy', acc_test_m, epoch + 1)
        writer.add_scalar('test/CrossEntropy', ent_test_m, epoch + 1)
        writer.add_scalar('test/TotalLoss', loss_test_m, epoch + 1)

    with redirect_stdout(params['log']):
        out_string = ''.join(
            ('Epoch: {:04d}\n', 'acc_train: {:.6f}, ', 'ent_train: {:.6f}, ',
             'loss_train: {:.6f}, ', 'acc_test: {:.6f}, ', 'ent_test: {:.6f}, ',
             'loss_test: {:.6f}, ', 'time: {:.4f}s'))
        print(
            out_string.format(epoch, acc_train_m, ent_train_m, loss_train_m,
                              acc_test_m, ent_test_m, loss_test_m,
                              time.time() - t_start))
        if args.save_folder and loss_test_m < best_val_loss:
            torch.save(model.state_dict(), model_file)

    return loss_test_m


def test(args, params):
    """
    Main test function after the training finishes.
    """
    train_loader, test_loader = params['loaders']
    model = params['model']

    model_file = params['model_file']

    acc_test = []
    ent_test = []
    loss_test = []

    model.eval()
    model.load_state_dict(torch.load(model_file))

    for index, (data, labels) in enumerate(tqdm(test_loader)):
        # data = data[:, :, :args.timesteps, :]
        if args.cuda:
            data = data.cuda()
            labels = labels.cuda()

        with torch.no_grad():
            output = model(data)

            loss_ent = label_cross_entropy(output, labels)
            label_acc = label_accuracy(output, labels)
            loss_reg = _calculate_l1(model, args)
            loss = loss_ent

        acc_test.append(label_acc.detach().item())
        ent_test.append(loss_ent.detach().item())
        loss_test.append(loss.detach().item())

    acc_test_m = np.mean(acc_test)
    ent_test_m = np.mean(ent_test)
    loss_test_m = np.mean(loss_test)

    return acc_test_m, ent_test_m, loss_test_m


def main():
    """
    Main function for training.
    """
    args = arguments_gnn.parse()
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

        model = _construct_model(args, graphs)
        model.cuda()

        optimizer, scheduler = _construct_optimizer(model, args)

        param_dict.update({
            'loaders': loaders,
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
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
