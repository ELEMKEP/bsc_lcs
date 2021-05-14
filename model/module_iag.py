import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


class IAG(nn.Module):

    def __init__(self, N, Fin, Fout, K, coarsening_map=None):
        super(IAG, self).__init__()

        self.N = N
        self.Fin = Fin
        self.Fout = Fout
        self.K = K

        # X: [N, D] (D: #frequency band = Fin)
        # P: [N, N], O: [N, D] --> B: [N, 1]? [N, D]?
        self.P = nn.Parameter(torch.randn(N, N), requires_grad=True)
        self.B = nn.Parameter(torch.randn(N, 1), requires_grad=True)

        # G = Relu(OQT) - T: Theta
        self.Q = nn.Parameter(torch.randn(Fin, Fin), requires_grad=True)
        self.T = nn.Parameter(torch.randn(Fin, N * Fin), requires_grad=True)

        # Y = (Cat[AX])U
        self.U = nn.Parameter(torch.randn(Fin, Fout))

        self.A = None  # For loss computation

        self.Cmap = coarsening_map  # For graph coarsening

        # LSTM model definition
        '''
        The role of LSTM is to model dependencies of the features having
        sequential structure in graph coarsening layer to 
        an emotion-discriminative feature vector,
        which is helpful to provide a good performance on EEG emotion recognition.

        What's the meaning of this sentence?
        '''

    def forward(self, x):
        # x.shape = [B, N, Fin]
        B, N, Fin = x.size()
        assert (N == self.N) and (
            Fin == self.Fin), 'N and Fin of data and model must be same.'

        # Eq. 2
        O = torch.matmul(self.P, x) + self.B  # [B, N, Fin]

        # Eq. 3
        G = torch.matmul(O, self.Q)  # [B, N, Fin]
        G = F.relu(torch.matmul(G, self.T))  # [B, N, N*Fin]
        G = G.view(B, N, N, Fin).permute(0, 3, 1, 2)  # [B, Fin, N, N]

        # Degree normalization
        Dii = torch.sqrt(torch.sum(G, dim=-2))  # [B, Fin, (1,) N]
        Dii_sqinv = torch.diag_embed(1 / Dii)  # [B, Fin, N, N]
        Djj = torch.sqrt(torch.sum(G, dim=-1))  # [B, Fin, N (,1)]
        Djj_sqinv = torch.diag_embed(1 / Djj)  # [B, Fin, N, N]

        A = torch.matmul(Dii_sqinv, G)  # [B, Fin, N, N]
        A = torch.matmul(A, Djj_sqinv)  # [B, Fin, N, N]
        self.A = A  # For loss computation

        Asum = torch.zeros(B, Fin, N, N)
        AA = torch.eye(N).view(1, 1, N, N).repeat(B, Fin, 1,
                                                  1)  # [B, Fin, N, N]
        for i in range(self.K - 1):
            Asum += AA
            AA = torch.bmm(A, AA)
        Asum += AA  # A^(K-1) / Asum.shape = [B, Fin, N, N]

        # Graph convolution
        X = x.transpose(-1, -2).view(B, Fin, N, 1)
        X = torch.bmm(Asum,
                      X)  # [B, Fin, N, N] * [B, Fin, N, 1] => [B, Fin, N, 1]
        X = X.squeeze(-1).transpose(-1, -2)  # [B, N, Fin]

        Y = torch.bmm(X, self.U)

        # Graph coarsening
        # LSTM model

        return Y
