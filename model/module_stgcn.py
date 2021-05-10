import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class STGCN(nn.Module):
    "Code from yysijie/st-gcn"

    def __init__(self, in_channels, num_class, graph, edge_importance_weighting,
                 model_size='medium', **kwargs):
        super().__init__()

        # load graph
        self.graph = np.expand_dims(graph.astype(np.float32).toarray(), axis=0)
        A = torch.tensor(self.graph, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        if model_size == 'full':
            self.st_gcn_networks = nn.ModuleList((
                st_gcn_block(in_channels, 64, kernel_size, 1, residual=False,
                             **kwargs0),
                st_gcn_block(64, 64, kernel_size, 1, **kwargs),
                st_gcn_block(64, 64, kernel_size, 1, **kwargs),
                st_gcn_block(64, 64, kernel_size, 1, **kwargs),
                st_gcn_block(64, 128, kernel_size, 2, **kwargs),
                st_gcn_block(128, 128, kernel_size, 1, **kwargs),
                st_gcn_block(128, 128, kernel_size, 1, **kwargs),
                st_gcn_block(128, 256, kernel_size, 2, **kwargs),
                st_gcn_block(256, 256, kernel_size, 1, **kwargs),
                st_gcn_block(256, 256, kernel_size, 1, **kwargs),
            ))
            fcn_in = 256
        elif model_size == 'medium':
            self.st_gcn_networks = nn.ModuleList((
                st_gcn_block(in_channels, 64, kernel_size, 1, residual=False,
                             **kwargs0),
                st_gcn_block(64, 128, kernel_size, 2, **kwargs),
                st_gcn_block(128, 128, kernel_size, 1, **kwargs),
                st_gcn_block(128, 256, kernel_size, 2, **kwargs),
                st_gcn_block(256, 256, kernel_size, 1, **kwargs),
            ))
            fcn_in = 256
        elif model_size == 'small':
            self.st_gcn_networks = nn.ModuleList((
                st_gcn_block(in_channels, 32, kernel_size, 1, residual=False,
                             **kwargs0),
                st_gcn_block(32, 64, kernel_size, 2, **kwargs),
                st_gcn_block(64, 128, kernel_size, 2, **kwargs),
            ))
            fcn_in = 128

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(fcn_in, num_class, kernel_size=1)

    def forward(self, x):
        # Input shape: [batch, atoms, times, num_dims]

        # data normalization
        # []

        # N, C, T, V, M = x.size() # batch, feat, time, joint, body
        # x = x.permute(0, 4, 3, 1, 2).contiguous() # batch, body, joint, feat, time
        # x = x.view(N * M, V * C, T) # batch*body, joint*feat, time
        # x = self.data_bn(x)
        # x = x.view(N, M, V, C, T) # batch, body, joint, feat, time
        # x = x.permute(0, 1, 3, 4, 2).contiguous() # batch, body, feat, time, joint
        # x = x.view(N * M, C, T, V) # batch*body, feat, time, joint

        # Non-ChebNet transform function
        if len(x.size()) == 4:
            N, A, T, _ = x.size()  # batch, atom(joint), time, feat
            x = x.permute(0, 3, 2, 1).contiguous()
        elif len(x.size()) == 3:
            # for ChebNet transform function
            N, A, T = x.size()
            x = torch.unsqueeze(x.permute(0, 2, 1),
                                dim=1)  # batch, feat, time, atom

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, 1, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x


class st_gcn_block(nn.Module):
    "Code from yysijie/st-gcn"

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dropout=0, residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=(stride, 1)),  # N, C_in, H, W == N, C, T, A
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A


class ConvTemporalGraphical(nn.Module):
    "Code from yysijie/st-gcn"

    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1,
                 t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size,
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0), stride=(t_stride, 1),
                              dilation=(t_dilation, 1), bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))  # k, v

        return x.contiguous(), A
