import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils_math import get_offdiag_indices, gumbel_softmax
from chebnet_lib import coarsening, graph
from deap_modules import Conv1DInception

_EPS = 1e-10


class ChebyshevLayer_DGCNN(nn.Module):

    def __init__(self, K, Fin, Fout, is_cuda):
        super(ChebyshevLayer_DGCNN, self).__init__()
        self.K = K
        self.Fin = Fin
        self.Fout = Fout
        self.is_cuda = is_cuda

        self.conv = nn.Conv1d(Fin, Fout, kernel_size=1, bias=False)

        self.theta = nn.Parameter(
            torch.empty(1, 1, K, dtype=torch.float32, requires_grad=True))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data)
        nn.init.xavier_normal_(self.theta)
        nn.init.xavier_normal_(self.W)

    def _laplacian(self, W):
        W = F.relu(W)
        L = torch.diag_embed(torch.sum(W, dim=1)) - W

        val, _ = torch.symeig(L, eigenvectors=True, upper=True)
        L = L * 2 / val[-1] - torch.eye(L.size(0), device=L.device)

        return L

    def forward(self, inputs, W):
        B, N, Fin = inputs.size()  # should know the meaning of N, M, Fin
        # assert Fin == self.Fin, 'Given Fin of input is not same to self.Fin'

        L = self._laplacian(self.W)
        x0 = inputs.permute(1, 2, 0).contiguous().view(
            N, -1)  # [N, Fin, B] => [N, Fin*B]

        x_list = [x0]
        for k in range(1, self.K):
            if k == 1:
                x1 = torch.matmul(L, x0)
                x_list.append(x1)
            else:
                x2 = 2 * torch.matmul(L, x1) - x0
                x_list.append(x2)
                x0, x1 = x1, x2

        x = torch.stack(x_list, dim=-1)  # [N, Fin*B, K]
        x = torch.sum(torch.mul(x, self.theta), dim=-1)  # [N, Fin * B]

        x = self.conv(x.view(N, Fin, B))  # [N, Fout, B]
        x = x.permute(2, 0, 1).contiguous()
        x = F.relu(x)  # [B, N, Fout]
        return x  # [B, N, Fout]


class DGCNN(nn.Module):

    def __init__(self, F_in, F_hid, F_out, W_init, K, is_cuda=True,
                 train_graph=True):
        super(DGCNN, self).__init__()

        W_temp = W_init.astype(np.float32).toarray()
        self.W = nn.Parameter(torch.FloatTensor(W_temp),
                              requires_grad=train_graph)
        self.n_obj = self.W.size(0)
        self.is_cuda = is_cuda
        self.device = torch.device('cuda') if is_cuda else torch.device('cpu')

        self.layer = ChebyshevLayer_DGCNN(K, F_in, F_hid, is_cuda)
        self.fc1 = nn.Linear(self.n_obj * F_hid, F_hid)
        self.fc2 = nn.Linear(F_hid, F_out)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.)

    def forward(self, inputs):
        # Regularize the matrix W
        # x.shape = [B, N, Fin] (vector feature)
        # Chebyshev layer: B, N, Fin

        s = inputs.size()
        if len(s) == 3:
            B, N, Fin = s
            x = inputs
        elif len(s) == 4:
            B, N, T, Fin = s
            x = torch.squeeze(inputs)

        x = self.layer(x, self.W)
        x = F.relu(self.fc1(x.view(B, -1)))
        x = self.fc2(x)

        return x


class DGCNN_V2(nn.Module):

    def __init__(self, F_in, F_hid, F_out, W_init, K, is_cuda=True,
                 train_graph=True, rho=0.5):
        super(DGCNN_V2, self).__init__()

        self.W = nn.Parameter(torch.FloatTensor(W_init),
                              requires_grad=train_graph)
        self.n_obj = self.W.size(0)
        self.is_cuda = is_cuda
        self.device = torch.device('cuda') if is_cuda else torch.device('cpu')
        self.rho = rho

        self.layer = ChebyshevLayer_DGCNN(K, F_in, F_hid, is_cuda)
        self.fc = nn.Linear(self.n_obj * F_hid, F_out)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                # m.bias.data.fill_(0.)

    def update_graph(self):
        W_grad = self.W.grad
        self.W.data *= (1 - self.rho)
        self.W.data += (self.rho * W_grad)
        self.W.data = F.relu(self.W.data)

    def forward(self, inputs):
        # Regularize the matrix W
        # x.shape = [B, N, Fin] (vector feature)
        # Chebyshev layer: B, N, Fin

        s = inputs.size()
        if len(s) == 3:
            B, N, Fin = s
            x = inputs
        elif len(s) == 4:
            B, N, T, Fin = s
            x = torch.squeeze(inputs)

        x = self.layer(x, self.W)
        x = x.view(B, -1)
        x = self.fc(x)

        return x


class DGCNN_V2_Reverse(nn.Module):

    def __init__(self, F_in, F_hid, F_out, W_init, K, is_cuda=True,
                 train_graph=True, rho=0.5):
        super(DGCNN_V2_Reverse, self).__init__()

        self.W = nn.Parameter(torch.FloatTensor(W_init),
                              requires_grad=train_graph)
        self.n_obj = self.W.size(0)
        self.is_cuda = is_cuda
        self.device = torch.device('cuda') if is_cuda else torch.device('cpu')
        self.rho = rho

        self.layer = ChebyshevLayer_DGCNN(K, F_in, F_hid, is_cuda)
        self.fc = nn.Linear(self.n_obj * F_hid, F_out)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                # m.bias.data.fill_(0.)

    def update_graph(self):
        W_grad = self.W.grad
        self.W.data *= (1 - self.rho)
        self.W.data -= (self.rho * W_grad)
        self.W.data = F.relu(self.W.data)

    def forward(self, inputs):
        # Regularize the matrix W
        # x.shape = [B, N, Fin] (vector feature)
        # Chebyshev layer: B, N, Fin

        s = inputs.size()
        if len(s) == 3:
            B, N, Fin = s
            x = inputs
        elif len(s) == 4:
            B, N, T, Fin = s
            x = torch.squeeze(inputs)

        x = self.layer(x, self.W)
        x = x.view(B, -1)
        x = self.fc(x)

        return x