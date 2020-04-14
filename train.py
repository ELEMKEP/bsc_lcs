import os
import time
import pickle
import datetime
from contextlib import redirect_stdout

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from tqdm import tqdm
from tensorboardX import SummaryWriter

import arguments
from model.modules import MLPEncoder, Feat_GNN
from utils.utils_math import encode_onehot, sample_graph
from utils.utils_loss import label_accuracy, label_cross_entropy
from utils.utils_miscellaneous import graph_to_image
from utils.utils_data import *


def _construct_model(args):
    if isinstance(args.encoder_hidden, list):
        args.encoder_hidden = args.encoder_hidden[0]
    encoder = MLPEncoder(args.timesteps * args.dims, args.encoder_hidden,
                         args.edge_types, args.encoder_dropout)

    if isinstance(args.decoder_hidden, list):
        args.decoder_hidden = args.decoder_hidden[0]

    if args.dataset == 'deap':
        decoder_out_dim = 40
    elif args.dataset == 'dreamer':
        decoder_out_dim = 18

    decoder = Feat_GNN(
        n_in_node=args.dims,
        n_time=args.timesteps,
        n_obj=args.num_objects,
        edge_types=args.edge_types,
        msg_hid=args.decoder_hidden,
        msg_out=args.decoder_hidden,
        n_hid=args.decoder_hidden,
        n_out=decoder_out_dim,  # binary classification for valence
        do_prob=args.decoder_dropout,
        skip_first=args.skip_first)

    if args.load_folder:
        encoder_file = os.path.join(args.load_folder, 'encoder.pt')
        encoder.load_state_dict(torch.load(encoder_file))
        decoder_file = os.path.join(args.load_folder, 'decoder.pt')
        decoder.load_state_dict(torch.load(decoder_file))

        args.save_folder = False

    return encoder, decoder


def _construct_optimizer(models, args):
    params = []
    for model in models:
        params = params + list(model.parameters())

    optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                    gamma=args.gamma)

    return optimizer, scheduler


def _construct_auxiliary_parameters(args):
    off_diag = np.ones([args.num_objects, args.num_objects]) - np.eye(
        args.num_objects)

    rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
    rel_rec = torch.FloatTensor(rel_rec)
    rel_send = torch.FloatTensor(rel_send)

    return rel_rec, rel_send


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
    train_loader, valid_loader, _ = params['loaders']
    encoder = params['encoder']
    decoder = params['decoder']
    optimizer = params['optimizer']
    scheduler = params['scheduler']
    rel_rec = params['rel_rec']
    rel_send = params['rel_send']

    encoder_file = params['encoder_file']
    decoder_file = params['decoder_file']
    writer = params.get('writer', None)
    use_tensorboard = (writer is not None)

    t_start = time.time()

    acc_train = []
    ent_train = []
    loss_train = []

    encoder.train()
    decoder.train()
    scheduler.step()

    train_iterable = tqdm(train_loader)
    for index, (data, labels) in enumerate(train_iterable):
        train_iterable.set_description("Epoch{:3} Train".format(epoch + 1))

        data = data[:, :, :args.timesteps, :]
        if args.cuda:
            data = data.cuda()
            labels = labels.cuda()
        optimizer.zero_grad()

        logits = encoder(data, rel_rec, rel_send)
        edges = sample_graph(logits, args)
        output = decoder(data, edges, rel_rec, rel_send)

        loss_ent = label_cross_entropy(output, labels)
        label_acc = label_accuracy(output, labels)
        loss = loss_ent

        loss.backward()
        optimizer.step()

        acc_train.append(label_acc.detach().item())
        ent_train.append(loss_ent.detach().item())
        loss_train.append(loss.detach().item())

    acc_valid = []
    ent_valid = []
    loss_valid = []

    encoder.eval()
    decoder.eval()

    valid_iterable = tqdm(valid_loader)
    for index, (data, labels) in enumerate(valid_iterable):
        valid_iterable.set_description("Epoch{:3} Valid".format(epoch + 1))

        data = data[:, :, :args.timesteps, :]
        if args.cuda:
            data = data.cuda()
            labels = labels.cuda()

        with torch.no_grad():
            logits = encoder(data, rel_rec, rel_send)
            edges = sample_graph(logits, args)
            prob = F.softmax(logits, dim=-1)

            if index == 0 and use_tensorboard:
                edge_image = graph_to_image(edges, args.skip_first,
                                            args.num_objects, 64)
                prob_image = graph_to_image(prob, args.skip_first,
                                            args.num_objects, 64)

                writer.add_image('val_edge', edge_image, epoch + 1)
                writer.add_image('val_prob', prob_image, epoch + 1)

            output = decoder(data, edges, rel_rec, rel_send)
            loss_ent = label_cross_entropy(output, labels)
            label_acc = label_accuracy(output, labels)
            loss = loss_ent

        acc_valid.append(label_acc.detach().item())
        ent_valid.append(loss_ent.detach().item())
        loss_valid.append(loss.detach().item())

    acc_train_m = np.mean(acc_train)
    ent_train_m = np.mean(ent_train)
    loss_train_m = np.mean(loss_train)

    acc_valid_m = np.mean(acc_valid)
    ent_valid_m = np.mean(ent_valid)
    loss_valid_m = np.mean(loss_valid)

    if use_tensorboard:
        writer.add_scalar('train/Accuracy', acc_train_m, epoch + 1)
        writer.add_scalar('train/CrossEntropy', ent_train_m, epoch + 1)
        writer.add_scalar('train/TotalLoss', loss_train_m, epoch + 1)

        writer.add_scalar('valid/Accuracy', acc_valid_m, epoch + 1)
        writer.add_scalar('valid/CrossEntropy', ent_valid_m, epoch + 1)
        writer.add_scalar('valid/TotalLoss', loss_valid_m, epoch + 1)

    with redirect_stdout(open(args.out, 'a')):
        out_string = ''.join(
            ('Epoch: {:04d}\n', 'acc_train: {:.6f}, ', 'ent_train: {:.6f}, ',
             'loss_train: {:.6f}, ', 'acc_valid: {:.6f}, ',
             'ent_valid: {:.6f}, ', 'loss_valid: {:.6f}, ', 'time: {:.4f}s'))
        print(
            out_string.format(epoch, acc_train_m, ent_train_m, loss_train_m,
                              acc_valid_m, ent_valid_m, loss_valid_m,
                              time.time() - t_start))
        if args.save_folder and loss_valid_m < best_val_loss:
            torch.save(encoder.state_dict(), encoder_file)
            torch.save(decoder.state_dict(), decoder_file)
            print('Best model so far, saving...')
    return loss_valid_m


def test(args, params):
    """
    Main test function after the training finishes.
    """
    _, _, test_loader = params['loaders']
    encoder = params['encoder']
    decoder = params['decoder']
    rel_rec = params['rel_rec']
    rel_send = params['rel_send']

    encoder_file = params['encoder_file']
    decoder_file = params['decoder_file']

    acc_test = []
    ent_test = []
    loss_test = []

    encoder.eval()
    decoder.eval()
    encoder.load_state_dict(torch.load(encoder_file))
    decoder.load_state_dict(torch.load(decoder_file))

    for index, (data, labels) in enumerate(tqdm(test_loader)):
        data = data[:, :, :args.timesteps, :]
        if args.cuda:
            data = data.cuda()
            labels = labels.cuda()

        with torch.no_grad():
            logits = encoder(data, rel_rec, rel_send)
            edges = sample_graph(logits, args)
            output = decoder(data, edges, rel_rec, rel_send)

            loss_ent = label_cross_entropy(output, labels)
            label_acc = label_accuracy(output, labels)
            loss = loss_ent

        acc_test.append(label_acc.detach().item())
        ent_test.append(loss_ent.detach().item())
        loss_test.append(loss.detach().item())

    metric_dict = {}

    acc_test_m = np.mean(acc_test)
    ent_test_m = np.mean(ent_test)
    loss_test_m = np.mean(loss_test)

    metric_dict['acc_test'] = acc_test_m
    metric_dict['xent_test'] = ent_test_m
    metric_dict['loss_test'] = loss_test_m

    with redirect_stdout(open(args.out, 'a')):
        print('--------------------------------')
        print('------------Testing-------------')
        print('--------------------------------')
        print('acc_test: {:.10f}\n'.format(acc_test_m),
              'ent_test: {:.10f}\n'.format(ent_test_m),
              'loss_test: {:.10f}\n'.format(loss_test_m))

    return metric_dict


def main():
    """
    Main function for training.
    """
    args = arguments.parse()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    param_dict = dict()

    # Save model and meta-data. Always saves in a new sub-folder.
    if args.save_folder:
        timestamp = datetime.datetime.now().isoformat().replace(':', '-')
        save_folder = '{}/exp{}_{}/'.format(args.save_folder, timestamp,
                                            args.out)
        os.mkdir(save_folder)
        meta_file = os.path.join(save_folder, 'metadata.pkl')
        encoder_file = os.path.join(save_folder, 'encoder.pt')
        decoder_file = os.path.join(save_folder, 'decoder.pt')

        log_file = os.path.join(save_folder, 'log.txt')
        log = open(log_file, 'w')

        pickle.dump({'args': args}, open(meta_file, "wb"))

        param_dict.update({
            'save_folder': save_folder,
            'encoder_file': encoder_file,
            'decoder_file': decoder_file
        })
    else:
        print("WARNING: No save_folder provided!" +
              "Testing (within this script) will throw an error.")

    def transform(datum):
        if args.dataset == 'deap':
            data_t = transform_deap_data_raw(datum)
            label_t = transform_deap_label_video(datum)
        elif args.dataset == 'dreamer':
            data_t = transform_dreamer_data_raw(datum)
            label_t = transform_dreamer_label_video(datum)

        return data_t, label_t

    loaders = load_lmdb_dataset(lmdb_root=args.data_path,
                                batch_size=args.batch_size, transform=transform,
                                shuffle=True)

    encoder, decoder = _construct_model(args)
    rel_rec, rel_send = _construct_auxiliary_parameters(args)
    rel_rec, rel_send = _make_cuda((encoder, decoder), (rel_rec, rel_send),
                                   args)
    optimizer, scheduler = _construct_optimizer((encoder, decoder), args)

    param_dict.update({
        'loaders': loaders,
        'encoder': encoder,
        'decoder': decoder,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'rel_rec': rel_rec,
        'rel_send': rel_send
    })

    if args.tensorboard:
        import socket
        log_dir = os.path.join(
            'runs', timestamp + '_' + args.out + '_' + socket.gethostname())
        writer = SummaryWriter(logdir=log_dir)
        param_dict['writer'] = writer

    # Train model
    best_val_loss = np.inf
    best_epoch = 0
    for epoch in range(args.epochs):
        val_loss = train(epoch, best_val_loss, args, param_dict)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

    with redirect_stdout(open(args.out, 'a')):
        print("Optimization Finished!")
        print("Best Epoch: {:04d}".format(best_epoch + 1))
        if args.save_folder:
            print("Best Epoch: {:04d}".format(best_epoch + 1), file=log)
            log.flush()

    # Test model
    metric_dict = test(args, param_dict)
    if log is not None:
        print(save_folder)
        log.close()

    if writer is not None:
        if args.deterministic_sampling:
            gsample = 'DET'
        elif args.hard:
            gsample = 'STO'
        else:
            gsample = 'CON'

        hparam_dict = {
            'lr': args.lr,
            'sampling': gsample,
            'enc_hid': args.encoder_hidden,
            'dec_hid': args.decoder_hidden,
        }
        writer.add_text('parameters', str(hparam_dict), 0)
        writer.add_text('metrics', str(metric_dict), 0)
        writer.close()


if __name__ == "__main__":
    main()
