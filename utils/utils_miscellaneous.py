import time
import torch
import torchvision.utils as vutils

from utils.utils_math import offdiag_to_mat
from sklearn import datasets, naive_bayes, linear_model, svm, neighbors, ensemble


def graph_to_image(graph, skip_first, num_atoms, max_cut):
    graph_mat = offdiag_to_mat(graph, num_atoms)

    if graph.size(0) > 64:
        graph_mat = graph_mat[:max_cut]
    else:
        graph_mat = graph_mat

    if skip_first:
        graph_mat = graph_mat[..., 1:]

    g_size = graph_mat.size()

    add_dim = 3 - g_size[-1]
    add_mat = torch.zeros(g_size[:-1] + (add_dim,)).to(graph_mat.device)

    graph_mat = torch.cat((graph_mat, add_mat), dim=(len(g_size) - 1))
    graph_mat = graph_mat.permute(0, 3, 1, 2)

    graph_image = vutils.make_grid(graph_mat, normalize=False, scale_each=False,
                                   pad_value=0.5)
    return graph_image


def baseline_multiclass(train_data, train_labels, test_data, test_labels, args):
    """Train various classifiers to get a baseline."""
    clf, train_accuracy, test_accuracy, train_f1, test_f1, exec_time = [], [], [], [], [], []
    clf.append(sklearn.neighbors.KNeighborsClassifier(n_neighbors=15, n_jobs=8))
    clf.append(
        sklearn.ensemble.RandomForestClassifier(n_jobs=8, verbose=10,
                                                random_state=args.seed))
    for i, c in enumerate(clf):
        t_start = time.process_time()
        c.fit(train_data, train_labels)
        train_pred = c.predict(train_data)
        test_pred = c.predict(test_data)
        train_accuracy.append('{:5.2f}'.format(
            100 * sklearn.metrics.accuracy_score(train_labels, train_pred)))
        test_accuracy.append('{:5.2f}'.format(
            100 * sklearn.metrics.accuracy_score(test_labels, test_pred)))
        train_f1.append('{:5.2f}'.format(100 * sklearn.metrics.f1_score(
            train_labels, train_pred, average='weighted')))
        test_f1.append('{:5.2f}'.format(100 * sklearn.metrics.f1_score(
            test_labels, test_pred, average='weighted')))
        exec_time.append('{:5.2f}'.format(time.process_time() - t_start))
    print('Train accuracy:      {}'.format(' '.join(train_accuracy)))
    print('Test accuracy:       {}'.format(' '.join(test_accuracy)))
    print('Train F1 (weighted): {}'.format(' '.join(train_f1)))
    print('Test F1 (weighted):  {}'.format(' '.join(test_f1)))
    print('Execution time:      {}'.format(' '.join(exec_time)))
    return train_accuracy, test_accuracy, train_f1, test_f1, exec_time, clf
