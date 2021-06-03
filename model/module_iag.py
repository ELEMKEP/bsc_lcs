import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-4

class IAG(nn.Module):

    def __init__(self, N, Fin, Fout, hid_gcn=32, hid_lstm=64, K=8,
                 alpha=[1e-4, 1e-5, 1e-5, 1e-5, 1e-5], coarsening_map=None):
        super(IAG, self).__init__()

        self.N = N
        self.Fin = Fin
        self.Fout = Fout  # Default: #class
        self.K = K  # Default: 8
        self.alpha = alpha

        assert Fin == len(alpha), "len(alpha) != Fin"

        # self.GCNhid = hid_gcn  # Default: 32, 64 for LSTM
        # self.LSTMhid = hid_lstm

        # X: [N, D] (D: #frequency band = Fin)
        # P: [N, N], O: [N, D] --> B: [N, 1]? [N, D]?
        self.P = nn.Parameter(torch.randn(N, N), requires_grad=True)
        self.B = nn.Parameter(torch.randn(N, 1), requires_grad=True)

        # G = Relu(OQT) - T: Theta
        self.Q = nn.Parameter(torch.randn(Fin, Fin), requires_grad=True)
        self.T = nn.Parameter(torch.randn(Fin, N * Fin), requires_grad=True)

        # Y = (Cat[AX])U
        self.U = nn.Parameter(torch.randn(Fin, hid_gcn))

        self.A = None  # For loss computation

        self.Cmap = coarsening_map  # For graph coarsening
        '''
        Original 62 nodes are clustered into 17 nodes.
        The dimensions of hidden state and memory cell in LSTM are both set to 64.
        '''

        # LSTM model definition for region dependency modeling
        # LSTM input: (seq_len, batch, input_size)
        # if batch_first=True, (batch, seq_len, feature)
        #   - seq_len: same to #node --> #region
        #   - batch: batch
        self.lstm = nn.LSTM(input_size=hid_gcn, hidden_size=hid_lstm,
                            num_layers=1, batch_first=True)
        '''
        input: (seq_len, batch, input_size)
        h_0: (num_layers * num_directions, batch, hidden_size): (initial) hidden state
        c_0: (num_layers * num_directions, batch, hidden_size): (initial) cell state

        output: (seq_len, batch, num_directions * hidden_size): h_t for each t.
                (batch, seq_len, num_directions * hidden_size) if batch_first=True?
        h_n: (num_layers * num_directions, batch, hidden_size): t=seq_len hidden state
        c_n: (num_layers * num_directions, batch, hidden_size): t=seq_len cell state
        '''
        # in_features = N * LSTM_output
        NN = len(self.Cmap) if self.Cmap else N
        self.fc = nn.Linear(in_features=NN * hid_lstm, out_features=Fout)

    def graph_l1_loss(self):
        self.A # self.A: [B, Fin, N, N]

        A_norm = self.A.norm(p=1, dim=-1).norm(p=1, dim=-1)
        alpha = torch.tensor(self.alpha).to(A_norm.device)
        band_norm = A_norm * alpha # [B, Fin] * [Fin] => [B * Fin]

        sum_norm = torch.sum(band_norm, dim=1)
        batch_mean_norm = torch.mean(sum_norm)
        
        return batch_mean_norm

    def _coarsen(self, y):
        # Graph coarsening operation
        # y.shape = [B, N, Fhid]
        B, N, F = y.size()

        if not self.Cmap:  # do not coarsen
            return y

        z_list = []
        for region in self.Cmap:
            rr = torch.LongTensor(region).view(1, len(region), 1).repeat(B, 1, F)
            rr = rr.to(y.device)

            y_src = torch.gather(y, dim=1,
                                 index=rr)  # [B, node_per_region, Fhid]
            z_i = torch.mean(y_src, dim=1, keepdim=True)
            z_list.append(z_i)
        z = torch.cat(z_list, dim=1)  # [B, #region, Fhid]

        return z

    def forward(self, x):
        # x.shape = [B, N, Fin]
        B, N, Fin = x.size()
        assert (N == self.N) and (
            Fin == self.Fin), 'N and Fin of data and model must be same.'
        # print(torch.mean(x), torch.std(x), torch.min(x), torch.max(x))

        # Eq. 2
        O = torch.matmul(self.P, x) + self.B  # [B, N, Fin]

        # Eq. 3
        G = torch.matmul(O, self.Q)  # [B, N, Fin]
        G = F.relu(torch.matmul(G, self.T))  # [B, N, N*Fin]
        G = G.view(B, N, N, Fin).permute(0, 3, 1, 2)  # [B, Fin, N, N]

        # Degree normalization
        Dii = torch.sqrt(torch.sum(G, dim=-2)) + EPS  # [B, Fin, (1,) N]
        Dii_sqinv = torch.diag_embed(1 / Dii)  # [B, Fin, N, N]
        Djj = torch.sqrt(torch.sum(G, dim=-1)) + EPS  # [B, Fin, N (,1)]
        Djj_sqinv = torch.diag_embed(1 / Djj)  # [B, Fin, N, N]

        A = torch.matmul(Dii_sqinv, G)  # [B, Fin, N, N]
        A = torch.matmul(A, Djj_sqinv)  # [B, Fin, N, N]
        self.A = A  # For loss computation
        # print(
        #     'Dii (min, max, mean, std): ', torch.min(Dii).item(), torch.max(Dii).item(), torch.mean(Dii).item(), torch.std(Dii).item(),
        #     'Djj (min, max, mean, std): ', torch.min(Djj).item(), torch.max(Djj).item(), torch.mean(Djj).item(), torch.std(Djj).item(),
        #     '\nDii_sqinv: ', Dii_sqinv.size(), torch.sum(Dii_sqinv), 
        #     '\nDjj_sqinv: ', Djj_sqinv.size(), torch.sum(Djj_sqinv),
        #     '\nA: ', A.size(), torch.sum(A))

        A = A.view(B*Fin, N, N)
        
        Asum = torch.zeros(B*Fin, N, N).to(A.device)
        AA = torch.eye(N).to(A.device)
        AA = AA.view(1, N, N).repeat(B*Fin, 1, 1)
        for i in range(self.K - 1):
            Asum += AA
            AA = torch.bmm(AA, A)
        Asum += AA  # A^(K-1) / Asum.shape = [B, Fin, N, N]
        # Asum = Asum.view(B, Fin, N, N)

        # Graph convolution and filtering
        X = x.transpose(-1, -2).contiguous().view(B*Fin, N, 1)
        X = torch.bmm(Asum,
                      X)  # [B*Fin, N, N] * [B*Fin, N, 1] => [B*Fin, N, 1]
        X = X.view(B, Fin, N).transpose(-1, -2)
        Y = torch.matmul(X, self.U)  # [B, N, Fin] * [Fin, Fhid] => [B, N, Fhid]

        # Graph coarsening
        Z = self._coarsen(Y)  # [B, #region, Fhid]

        # Region dependency modeling
        Z, _ = self.lstm(Z)

        # Fully-connected layer

        Z = Z.contiguous().view(B, -1)
        Z = self.fc(Z)

        return Z
