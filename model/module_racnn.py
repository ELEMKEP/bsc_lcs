import torch
import torch.nn as nn
import torch.nn.functional as F


class RACNN(nn.Module):

    def __init__(self, F_out, dataset):
        super(RACNN, self).__init__()

        self.dataset = dataset

        # Temporal feature
        self.t_conv1 = nn.Conv3d(1, 32, [1, 1, 5], padding=[0, 0, 2],
                                 stride=[1, 1, 2])
        self.t_conv2 = nn.Conv3d(32, 32, [1, 1, 3], padding=[0, 0, 1],
                                 stride=[1, 1, 2])
        self.t_conv3 = nn.Conv3d(32, 32, [1, 1, 3], padding=[0, 0, 1],
                                 stride=[1, 1, 2])
        self.t_conv4 = nn.Conv3d(32, 32, [1, 1, 16], padding=[0, 0, 2],
                                 stride=[1, 1, 16])

        if self.dataset.lower() == 'deap':
            r_kernel = [3, 3]
            r_padding = [1, 1]
            H, W = 9, 9
        elif self.dataset.lower() == 'dreamer':
            r_kernel = [1, 3]
            r_padding = [0, 1]
            H, W = 1, 14
        else:
            raise ValueError('dataset should be in [\'deap\', \'dreamer\']')

        # kernel_size: [3, 3] for DEAP, [1, 3] for DREAMER

        # Regional feature
        self.r_conv1 = nn.Conv2d(96, 32, r_kernel, padding=r_padding)
        self.r_conv2 = nn.Conv2d(32, 32, r_padding, padding=r_padding)

        # Asymmetric feature
        self.a_conv1 = nn.Conv2d(96, 64, [1, 1])

        # fr_flat = fr.view(B, -1)  # B, [32 * H * W]
        # fa_flat = fa.view(B, -1)  # B, [64 * H * W//2]
        fr_feat = 32 * H * W
        fa_feat = 64 * H * (W // 2)
        fc_in = fr_feat + fa_feat

        self.fc1 = nn.Linear(fc_in, 20)
        self.fc2 = nn.Linear(20, F_out)

    def _spatial_encode(self, x):
        B, N, T = x.size()  # x.shape = [Batch, n_obj, n_feat]

        if self.dataset == 'DEAP':
            output = torch.zeros(B, 9 * 9, T)
            # 0-based map (-1: not assigned)
            emap = torch.tensor([[-1, -1, -1, 0, -1, 16, -1, -1, -1],
                                 [-1, -1, -1, 1, -1, 17, -1, -1, -1],
                                 [3, -1, 2, -1, 18, -1, 19, -1, 20],
                                 [-1, 4, -1, 5, -1, 22, -1, 21, -1],
                                 [7, -1, 6, -1, 23, -1, 24, -1, 25],
                                 [-1, 8, -1, 9, -1, 27, -1, 26, -1],
                                 [11, -1, 10, -1, 15, -1, 28, -1, 29],
                                 [-1, -1, -1, 12, -1, 30, -1, -1, -1],
                                 [-1, -1, -1, 13, 14, 31, -1, -1, -1]])
            idx = torch.nonzero(emap != -1)
            idx = idx[:, 0] * 9 + idx[:, 1]
            idx = idx.view(1, N, 1).expand(B, -1, T)

            output.scatter_(dim=1, index=idx, src=x)  # [B, 9*9, F]

            # 5D tensor is needed for 3DConv
            output = output.view(B, 1, 9, 9, T)
            return output
        elif self.dataset == 'DREAMER':
            # 5D tensor is needed for 3DConv
            output = x.view(B, 1, 1, 14, T)

        return output

    def forward(self, x):
        x = self._spatial_encode(x)  # Get 5D tensor
        B, _, H, W, T = x.size()

        # Temporal feature
        t1 = F.relu(self.t_conv1(x))
        t2 = F.relu(self.t_conv2(t1))
        t3 = F.relu(self.t_conv3(t2))
        ft = F.relu(self.t_conv4(t3))
        # Last dimension is 3 - input length is 384
        # [B, C(32), 9, 9, 3] << DEAP
        # [B, C(32), 1, 14, 3] << DREAMER

        ft = ft.permute(0, 1, 4, 2, 3).contiguous()
        ft = ft.view(B, 32 * 3, H, W)

        # Regional feature
        r1 = F.relu(self.r_conv1(ft))
        fr = F.relu(self.r_conv2(r1))  # [B, C(32), H, W]

        # Asymmetric feature
        # ft.size() = [B, C, H, W]
        half = W // 2
        idx_l = torch.arange(0, half)
        idx_r = -torch.arange(1, half + 1)
        fa = ft[..., idx_l] - ft[..., idx_r]
        # DEAP: [B, C*F, 9, 4]
        # DREAMER: [B, C*F, 1, 7]
        fa = F.relu(self.a_conv1(fa))  # [B, 64, H, W//2]

        fr_flat = fr.view(B, -1)  # 32 * H * W
        fa_flat = fa.view(B, -1)  # 64 * H * W//2
        ff = torch.cat((fr_flat, fa_flat))

        f1 = F.relu(self.fc1(ff))
        f2 = F.relu(self.fc2(f1))

        return f2