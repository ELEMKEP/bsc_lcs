import numpy as np
import torch
import torch.nn.functional as F


def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/YongfeiYan/Gumbel_Softmax_VAE/blob/master/gumbel_softmax_vae.py

    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return -torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, temp=1, eps=1e-10, dim=-1):
    """
    NOTE: Stolen from https://github.com/YongfeiYan/Gumbel_Softmax_VAE/blob/master/gumbel_softmax_vae.py

    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + gumbel_noise
    return F.softmax(y / temp, dim=dim)


def gumbel_softmax(logits, temp=1, hard=False, eps=1e-10, dim=-1):
    """
    NOTE: Stolen from https://github.com/YongfeiYan/Gumbel_Softmax_VAE/blob/master/gumbel_softmax_vae.py

    Added dimension selection feature.

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, temp=temp, eps=eps, dim=dim)

    if hard:
        shape = logits.size()
        _, idx = y_soft.max(dim=dim, keepdim=True)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros_like(y_soft)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()

        y_hard = y_hard.zero_().scatter_(dim, idx, 1.0)
        y = (y_hard - y_soft).detach() + y_soft
    else:
        y = y_soft
    return y


def threshold_sampling(logits, threshold=0.5):
    """
    Omit Gumbel sampling for deterministic sampling.
    """
    y_soft = torch.sigmoid(logits)

    y_hard = y_soft.ge(threshold).to(y_soft.device, dtype=torch.float32)
    y = (y_hard - y_soft).detach() + y_soft

    return y


def binary_accuracy(output, labels):
    preds = output > 0.5
    correct = preds.type_as(labels).eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {
        c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)
    }
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def get_triu_indices(num_nodes):
    """Linear triu (upper triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    triu_indices = (ones.triu() - eye).nonzero().t()
    triu_indices = triu_indices[0] * num_nodes + triu_indices[1]
    return triu_indices


def get_tril_indices(num_nodes):
    """Linear tril (lower triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    tril_indices = (ones.tril() - eye).nonzero().t()
    tril_indices = tril_indices[0] * num_nodes + tril_indices[1]
    return tril_indices


def get_offdiag_indices(num_nodes):
    """Linear off-diagonal indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    offdiag_indices = (ones - eye).nonzero().t()
    offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
    return offdiag_indices


def get_triu_offdiag_indices(num_nodes):
    """Linear triu (upper) indices w.r.t. vector of off-diagonal elements."""
    triu_idx = torch.zeros(num_nodes * num_nodes)
    triu_idx[get_triu_indices(num_nodes)] = 1.
    triu_idx = triu_idx[get_offdiag_indices(num_nodes)]
    return triu_idx.nonzero()


def get_tril_offdiag_indices(num_nodes):
    """Linear tril (lower) indices w.r.t. vector of off-diagonal elements."""
    tril_idx = torch.zeros(num_nodes * num_nodes)
    tril_idx[get_tril_indices(num_nodes)] = 1.
    tril_idx = tril_idx[get_offdiag_indices(num_nodes)]
    return tril_idx.nonzero()


def mat_to_offdiag(inputs, num_atoms, num_edge_types):
    # change attention to 3-dimensional [batch_size, num_edges, num_edge_types]
    # inputs.shape = [batch_size, sum(n_heads), num_atoms, num_atoms]
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms]).astype(np.int32)
    num_edges = (num_atoms * num_atoms) - num_atoms

    if not inputs.is_contiguous():
        inputs = inputs.contiguous()
    inputs = inputs.view(-1, num_edge_types, num_atoms * num_atoms)
    inputs = torch.transpose(inputs, 2, 1)
    off_diag_idx = torch.LongTensor(off_diag_idx)

    if inputs.is_cuda:
        off_diag_idx = off_diag_idx.cuda()

    mat_offdiag = torch.index_select(inputs, 1, off_diag_idx).contiguous()
    return mat_offdiag


def offdiag_to_mat(inputs, num_nodes):
    # change attention to 3-dimensional [batch_size, num_edges, num_edge_types]
    # inputs.shape = [batch_size, num_nodes, num_nodes, num_edge_types]
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)),
        [num_nodes, num_nodes]).astype(np.int32)
    batch_size = inputs.size(0)
    edge_types = inputs.size(2)

    output = torch.zeros((batch_size, num_nodes * num_nodes, edge_types))
    if inputs.is_cuda:
        output = output.cuda()

    output[:, off_diag_idx, :] = inputs
    output = output.view(batch_size, num_nodes, num_nodes, edge_types)

    return output


def sample_graph(logits, args):
    if args.bernoulli_sampling:
        edges = bernoulli_sampling(logits)
    elif args.deterministic_sampling:
        edges = threshold_sampling(logits, threshold=args.threshold)
    elif args.fully_connected_graph:
        edges = threshold_sampling(logits, threshold=0.)
    else:
        edges = gumbel_softmax(logits, temp=args.temp, hard=args.hard)

    return edges
