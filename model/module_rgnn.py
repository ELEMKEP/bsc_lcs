import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class RGNN(nn.Module):

    def __init__(self, n_in, n_out, n_obj, n_hid, A_init, K=2, dropout=0.7,
                 domain_adaptation="RevGrad", train_graph=True):
        super(RGNN, self).__init__()

        A_temp = A_init.astype(np.float32).toarray()
        self.A = nn.Parameter(torch.FloatTensor(A_temp),
                              requires_grad=train_graph)
        self.n_obj = n_obj
        self.K = K

        self.n_in = n_in
        self.n_out = n_out

        self.domain_adaptation = domain_adaptation

        self.sgc_fc = nn.Linear(n_in, n_hid)
        self.sgc_dropout = nn.Dropout(p=dropout)
        self.out_fc = nn.Linear(n_hid, n_out)

        if self.domain_adaptation in ["RevGrad"]:
            self.domain_fc = nn.Linear(n_hid, 2)

    @staticmethod
    def norm(A, improved=False):
        indices = np.diag_indices(A.size(0))
        A_val = A[indices]

        fill_value = torch.tensor(1 if not improved else 2, dtype=torch.float32,
                                  device=A_val.device)
        A_val[torch.abs(A_val) == 0] += 1

        deg = torch.sum(torch.abs(A), dim=0, keepdim=False)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.
        deg_inv_mat = torch.diag(deg_inv_sqrt)

        A_norm = torch.matmul(torch.matmul(deg_inv_mat, A), deg_inv_mat)

        return A_norm

    def forward(self, x, alpha=0):
        # it seems that "data" in authors' file have x, edge_index, batch
        # what's the meaning of "batch"?

        # x.shape = [B, N, T] in the case of my experiment
        # S.shape = [N, N]
        S = RGNN.norm(self.A)
        Sk = torch.eye(self.n_obj, device=x.device)
        for _ in range(self.K):
            Sk = torch.matmul(Sk, S)
        # print(Sk)
        x = torch.matmul(torch.unsqueeze(Sk, dim=0), x)
        x = F.relu(self.sgc_fc(x))

        # domain classification
        domain_out = None
        if self.domain_adaptation in ["RevGrad"]:
            reverse_x = ReverseLayerF.apply(x, alpha)
            domain_out = self.domain_fc(reverse_x)

        # p = softmax (pool(ReLU(Z_i)) * W^O)
        x_pool = torch.sum(x, dim=1, keepdim=False)  # pool
        x_drop = self.sgc_dropout(x_pool)  # dropout?
        x_out = self.out_fc(x_drop)  # Apply W^O

        if self.training:
            return x_out, domain_out
        else:
            return x_out

    def graph_l1_loss(self):
        return torch.sum(torch.abs(self.A))
