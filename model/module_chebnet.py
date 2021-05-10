import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils_math import get_offdiag_indices, gumbel_softmax
from chebnet_lib import coarsening, graph
from deap_modules import Conv1DInception

_EPS = 1e-10


class ChebyshevLayer(nn.Module):

    def __init__(self, L, K, Fin, Fout, is_cuda):
        super(ChebyshevLayer, self).__init__()
        self.L = torch.FloatTensor(graph.rescale_L(L, lmax=2).todense())
        self.K = K
        self.Fin = Fin
        self.Fout = Fout

        self.layer = nn.Linear(Fin * K, Fout, bias=False)
        # self.bn = nn.BatchNorm1d(L.shape[-1])
        self.bn = nn.BatchNorm1d(Fin * K)

        if is_cuda:
            self.L = self.L.cuda()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                # m.bias.data.fill_(0.)

    def forward(self, inputs):
        B, N, Fin = inputs.size()  # should know the meaning of N, M, Fin
        # assert Fin == self.Fin, 'Given Fin of input is not same to self.Fin'

        x0 = inputs.permute(1, 2, 0).contiguous().view(
            N, -1)  # [N, Fin, B] => [N, Fin*B]

        x_list = [x0]
        for k in range(1, self.K):
            if k == 1:
                x1 = torch.matmul(self.L, x0)
                x_list.append(x1)
            else:
                x2 = 2 * torch.matmul(self.L, x1) - x0
                x_list.append(x2)
                x0, x1 = x1, x2
        x = torch.stack(x_list, dim=-1)  # [N, Fin*B, K]
        x = x.view(N, Fin, B, self.K).permute(2, 0, 1, 3).contiguous()
        # x = self.bn(x.view(B * N, Fin * self.K)).view(B, N, Fin * self.K)
        x = x.view(B, N, Fin * self.K)

        x = F.relu(self.layer(x))  # B*N x Fout

        # if self.K > 1:
        #     x1 = torch.matmul(self.L, x0)
        #     x = torch.cat([x0, x1], 0)
        # for k in range(2, self.K):
        #     x2 = 2 * torch.matmul(self.L, x1) - x0  # N x Fin*N
        #     x = torch.cat([x, x2], 0)
        #     x0, x1 = x1, x2
        # x = x.view(self.K, N, Fin, B).permute(3, 1, 2, 0)  # B x N x Fin x K
        # x = x.permute(3, 1, 2, 0)
        # x = x.view(B, N, Fin * self.K)  # N x M x Fin*K
        # # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        # x = self.bn(F.relu(self.layer(x)))  # B*N x Fout
        return x  # B x N x Fout


class DEAP_ChebNet(nn.Module):
    """ Virtual node-based output """

    def __init__(self, _L, _F, _K, _P, _M, Fin=1, is_cuda=True):
        super(DEAP_ChebNet, self).__init__()

        # assert (len(_F) == len(_K) ==
        #         len(_P)), 'List length of _F, _K, _P should be same'

        j = 0
        self._L = []
        j_former = 0
        for pp in _P:
            LL = graph.laplacian(_L[j].astype(np.float32), normalized=True)
            self._L.append(LL)
            j += int(np.log2(pp)) if pp > 1 else 0
            if j_former != j:
                # print('Graph index: {}'.format(j))
                # print(self._L[-1])
                j_former = j

        self._F, self._K, self._P, self._M = _F, _K, _P, _M
        self.n_layers = len(_P)

        self.gcn_layers = nn.ModuleList()
        self.pools = nn.ModuleList()

        for i, pp in enumerate(_P):
            self.gcn_layers.append(
                ChebyshevLayer(self._L[i], _K[i], Fin, _F[i], is_cuda=is_cuda))
            Fin = _F[i]

            if pp > 1:
                self.pools.append(
                    nn.MaxPool1d(kernel_size=pp, stride=pp, padding=0))
            else:
                self.pools.append(None)

        N_fc = _L[j].shape[-1]
        F_fc = self._F[-1]
        Fin = N_fc * F_fc
        self.fc_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dr_layers = nn.ModuleList()
        for i, mm in enumerate(self._M[:-1]):
            # self.bn_layers.append(nn.BatchNorm1d(Fin))
            self.fc_layers.append(nn.Linear(Fin, mm, bias=True))
            self.bn_layers.append(nn.BatchNorm1d(mm))
            self.dr_layers.append(nn.Dropout(p=0.))
            Fin = mm

        self.fc_logit = nn.Linear(Fin, self._M[-1], bias=True)
        # if is_cuda:
        #     self.fc_logit.cuda()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                # m.bias.data.fill_(0.)

    def forward(self, inputs):
        # Graph convolutional layers.
        # print('Input size: ', inputs.size())
        if len(inputs.size()) < 3:
            x = torch.unsqueeze(inputs, 2)  # N x M x F=1
        else:
            x = inputs

        def pool_func(x, pool):
            x = torch.transpose(x, 2, 1)  # [B, Fout, N]
            x = pool(x)
            x = torch.transpose(x, 2, 1)  # [B, N/p, Fout]
            return x

        for i in range(self.n_layers):
            # print('GCN layer: {}'.format(i), x)
            x = self.gcn_layers[i](x)
            pool = self.pools[i]
            x = pool_func(x, pool) if (pool is not None) else x

        # Fully connected hidden layers.
        # Start work from here
        B = inputs.size()[0]
        x = x.contiguous().view(int(B), -1)  # N x F_all
        # print('-----GCN output-----')
        # print(x)

        for i, fc_layer in enumerate(self.fc_layers):
            # print('Fully connected layer: {}'.format(i), x)
            bn_layer = self.bn_layers[i]
            dropout_layer = self.dr_layers[i]
            x = fc_layer(dropout_layer(x))
            x = F.relu(bn_layer(x))

        x = self.fc_logit(x)
        # print('Last output: ')
        # print(x)

        return x


class DEAP_ChebNet_V2(nn.Module):
    """ Virtual node-based output """

    def __init__(self, n_obj, n_time, _L, perm, _F, _K, _P, _M, Fin=1,
                 is_cuda=True):
        super(DEAP_ChebNet_V2, self).__init__()

        self.n_obj = n_obj
        self.n_time = n_time
        self.perm = perm
        self.is_cuda = is_cuda
        self.device = torch.device('cuda') if is_cuda else torch.device('cpu')

        assert (len(_F) == len(_K) ==
                len(_P)), 'List length of _F, _K, _P should be same'

        j = 0
        self._L = []
        for pp in _P:
            self._L.append(_L[j])
            j += int(np.log2(pp)) if pp > 1 else 0

        self._F, self._K, self._P, self._M = _F, _K, _P, _M
        self.n_layers = len(_F)

        ##### ConvInception Layers
        out_cnn = 8
        depth = 4
        self.cnn = nn.Sequential(
            Conv1DInception(1, out_cnn, 3, depth),
            nn.ReLU(),
            nn.MaxPool1d(4),  # 96
            Conv1DInception(out_cnn * depth, out_cnn, 3, depth),
            nn.ReLU(),
            nn.MaxPool1d(4),  # 24 
            Conv1DInception(out_cnn * depth, out_cnn, 3, depth),
            nn.ReLU(),
            nn.MaxPool1d(4),  # 6
        )
        self.out_channel = out_cnn * depth
        self.out_n_time = n_time // 64
        self.cheb_feat = self.out_n_time * self.out_channel

        indices = np.array(perm, dtype=np.int32)
        # indices[indices >= n_obj] = n_obj
        self.indices = torch.cuda.LongTensor(
            np.tile(indices, [self.cheb_feat, 1]).transpose())

        self.gcn_layers = nn.ModuleList()
        self.pools = nn.ModuleList()

        Fin = self.cheb_feat
        for i, pp in enumerate(_P):
            self.gcn_layers.append(
                ChebyshevLayer(self._L[i], _K[i], Fin, _F[i], is_cuda=is_cuda))
            Fin = _F[i]

            if pp > 1:
                self.pools.append(
                    nn.MaxPool1d(kernel_size=pp, stride=pp, padding=0))
            else:
                self.pools.append(None)

        N_fc = _L[j].shape[-1]
        F_fc = self._F[-1]
        Fin = N_fc * F_fc
        self.fc_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i, mm in enumerate(self._M[:-1]):
            self.fc_layers.append(nn.Linear(Fin, mm))
            self.bn_layers.append(nn.BatchNorm1d(mm))
            Fin = mm

        self.fc_logit = nn.Linear(Fin, self._M[-1])
        # if is_cuda:
        #     self.fc_logit.cuda()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.)

    def forward(self, inputs):
        # Graph convolutional layers.
        if len(inputs.size()) == 3:
            x = torch.unsqueeze(inputs, 3)  # N x M x F(T) x 1
        else:
            x = inputs

        x = x.view(-1, self.n_time, 1)
        x = x.transpose(1, 2).contiguous()
        reduced = self.cnn(x)  # [BxA, D', T']
        reduced = reduced.transpose(1, 2).contiguous()
        reduced = reduced.view(-1, self.n_obj, self.cheb_feat)  # [B, A, T*D]

        Nnew = self._L[0].shape[-1]
        Nold = self.n_obj

        zero_vec = torch.zeros(reduced.size(0), Nnew - Nold, reduced.size(2),
                               dtype=torch.float32, device=self.device)
        reduced_zv = torch.cat((reduced, zero_vec), dim=1)

        indices = self.indices.repeat([reduced.size(0), 1, 1])
        x = torch.zeros(reduced.size(0), Nnew, self.cheb_feat,
                        dtype=torch.float32, device=self.device)  # [B, P, T*D]

        x.scatter_(1, indices, reduced_zv)

        def pool_func(x, pool):
            x = torch.transpose(x, 2, 1)  # [B, Fout, N]
            x = pool(x)
            x = torch.transpose(x, 2, 1)  # [B, N/p, Fout]
            return x

        for i in range(self.n_layers):
            x = self.gcn_layers[i](x)
            pool = self.pools[i]
            x = pool_func(x, pool) if (pool is not None) else x

        # Fully connected hidden layers.
        # Start work from here
        B = inputs.size()[0]
        x = x.contiguous().view(int(B), -1)  # N x F_all

        for i, fc_layer in enumerate(self.fc_layers):
            bn_layer = self.bn_layers[i]
            x = bn_layer(F.relu(fc_layer(x)))

        x = self.fc_logit(x)

        return x