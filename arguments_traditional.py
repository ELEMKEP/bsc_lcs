import argparse
import os
import sys


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
    parser.add_argument('--data', type=str, default='raw',
                        help='data type (raw, de, power)')
    parser.add_argument('--data-path', type=str, default='E:\\lmdb\\',
                        help='Path for data.')


def _add_model_arguments(parser):
    ### Encoder and decoder
    parser.add_argument('--model', type=str, default='dgcnn', help='Model type')


def _add_knn_arguments(parser):
    parser.add_argument('--neighbor', type=int, default=15,
                        help='Number of neighbors')
    parser.add_argument('--n_jobs', type=int, default=8,
                        help='Number of kNN workers')


def _add_rf_arguments(parser):
    parser.add_argument('--n-estimator', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=-1)
    parser.add_argument('--min-samples-split', type=int, default=-1)


def _add_svm_arguments(parser):
    parser.add_argument('--C', type=float, default=1e-1)


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
    _add_model_arguments(parser)
    _add_knn_arguments(parser)
    _add_rf_arguments(parser)
    _add_svm_arguments(parser)
    _add_miscellaneous_arguments(parser)

    args = parser.parse_args()

    if args.seed == -1:
        args.seed = int.from_bytes(os.urandom(4), sys.byteorder)
    else:
        args.seed = args.seed

    print(args)

    return args
