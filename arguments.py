import argparse
import os
import sys
from contextlib import redirect_stdout

import numpy as np


def _add_data_arguments(parser):
    # Data arguments
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=-1, help='Random seed.')

    ### Data
    parser.add_argument('--dataset', type=str, default='deap',
                        help='Dataset type (deap, dreamer)')
    parser.add_argument('--label', type=str, default='video',
                        help='Label type (video, valence, arousal)')
    parser.add_argument('--data-path', type=str, default='E:\\lmdb\\',
                        help='Path for data.')
    parser.add_argument(
        '--save-folder', type=str, default='logs',
        help='Where to save the trained model, leave empty to not save anything.'
    )
    parser.add_argument(
        '--load-folder', type=str, default='',
        help='Where to load the trained model if finetunning. ' +
        'Leave empty to train from scratch')


def _add_learning_arguments(parser):
    ### Training hyper parameters
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Number of samples per batch.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Initial learning rate.')

    ### Learning parameters
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Adam beta factor for mean')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Adam beta factor for standard deviation')
    parser.add_argument(
        '--lr-decay', type=int, default=200,
        help='After how epochs to decay LR by a factor of gamma.')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='LR decay factor.')

    parser.add_argument('--num-objects', type=int, default=32,
                        help='Number of objects.')
    parser.add_argument(
        '--dims', type=int, default=1,
        help='The number of input dimensions (position + velocity).')
    parser.add_argument('--timesteps', type=int, default=384,
                        help='The number of time steps per sample.')


def _add_graph_arguments(parser):
    ### Graph Definition
    parser.add_argument('--edge-types', type=int, default=3,
                        help='Number of edge types.')
    parser.add_argument(
        '--skip-first', action='store_true', default=False,
        help='Skip first edge type in decoder, i.e. it represents no-edge.')

    ### Graph sampling method ###
    parser.add_argument('--deterministic-sampling', action='store_true',
                        default=False, help='Use deterministic sampling.')
    parser.add_argument(
        '--hard', action='store_true', default=False,
        help='Uses discrete samples in training forward pass in Gumbel-softmax.'
    )
    parser.add_argument('--temp', type=float, default=0.5,
                        help='Temperature for Gumbel softmax.')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for deterministic sampling.')


def _add_model_arguments(parser):
    ### Encoder and decoder
    parser.add_argument('--encoder-hidden', type=int, nargs='+', default=256,
                        help='Number of hidden units in encoder.')
    parser.add_argument('--decoder-hidden', type=int, nargs='+', default=256,
                        help='Number of hidden units.')
    parser.add_argument('--encoder-dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--decoder-dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')


def _add_miscellaneous_arguments(parser):
    ### Miscellaneous
    parser.add_argument('--out', type=str, default='out.txt',
                        help='Output file.')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='To print debug messages')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='Use TensorboardX for PyTorch.')


def parse():
    """
    parse arguments for common models.
    """
    parser = argparse.ArgumentParser()
    _add_data_arguments(parser)
    _add_learning_arguments(parser)
    _add_graph_arguments(parser)
    _add_model_arguments(parser)
    _add_miscellaneous_arguments(parser)

    args = parser.parse_args()

    if args.seed == -1:
        args.seed = int.from_bytes(os.urandom(4), sys.byteorder)
    else:
        args.seed = args.seed

    with open(args.out, 'a') as f:
        with redirect_stdout(f):
            print(args)
    print(args)

    return args
