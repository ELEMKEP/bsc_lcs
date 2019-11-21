import torch
import torch.nn as nn
import torch.nn.functional as F

# sys.path.append(os.path.abspath('../'))
from utils.utils_math import get_offdiag_indices, gumbel_softmax

_EPS = 1e-10

############################## Blocks ##############################


class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)


class DilatedInception(nn.Module):
    """ Inception module for Conv1D. for feature dimension reduction."""

    def __init__(self, F_in, F_out, kernel_size, depth):
        super(DilatedInception, self).__init__()

        self.F_in = F_in
        self.F_out = F_out
        self.kernel_size = kernel_size
        self.depth = depth

        dilation = 1
        self.conv_list = nn.ModuleList()
        for i in range(depth):
            conv = nn.Conv1d(
                in_channels=F_in,
                out_channels=F_out,
                kernel_size=kernel_size,
                padding=dilation,
                dilation=dilation,
            )
            self.conv_list.append(conv)
            dilation *= 2

    def forward(self, inputs):
        results = []
        for i in range(self.depth):
            results.append(self.conv_list[i](inputs))
            # shape = [N, Cout, Lout]

        output = torch.cat(results, dim=1)

        return output


############################## Main classes ##############################


class MLPEncoder(nn.Module):

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLPEncoder, self).__init__()

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)

        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)

        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([receivers, senders], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]

        x = self.mlp1(x)  # 2-layer ELU net per node

        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        x = self.edge2node(x, rel_rec, rel_send)
        x = self.mlp3(x)
        x = self.node2edge(x, rel_rec, rel_send)
        x = torch.cat((x, x_skip), dim=2)  # Skip connection
        x = self.mlp4(x)

        return self.fc_out(x)


class Feat_GNN(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_node, n_time, n_obj, edge_types, msg_hid, msg_out,
                 n_hid, n_out, do_prob=0., skip_first=False):
        super(Feat_GNN, self).__init__()
        self.n_in_node = n_in_node
        self.n_time = n_time
        self.n_obj = n_obj
        self.msg_out = msg_out
        self.skip_first_edge_type = skip_first

        # 1-D CNN for dimension reduction part
        out_cnn = 8
        depth = 4
        self.cnn = nn.Sequential(
            DilatedInception(n_in_node, out_cnn, 3, depth),
            nn.ReLU(),
            nn.MaxPool1d(4),  # 96
            DilatedInception(out_cnn * depth, out_cnn, 3, depth),
            nn.ReLU(),
            nn.MaxPool1d(4),  # 24 
            DilatedInception(out_cnn * depth, out_cnn, 3, depth),
            nn.ReLU(),
            nn.MaxPool1d(4),  # 6
        )
        self.out_channel = out_cnn * depth
        self.out_n_time = n_time // 64

        self.msg_fc1 = nn.ModuleList([
            nn.Linear(self.out_channel * 2, msg_hid) for _ in range(edge_types)
        ])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])

        self.out_fc_dim = n_obj * self.out_n_time * (msg_out + self.out_channel)

        self.out_fc1 = nn.Linear(self.out_fc_dim, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_out)
        self.dropout_prob = do_prob

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, inputs, rel_type, rel_rec, rel_send):
        # 1-D CNN for dimension reduction
        inputs = inputs.view(-1, self.n_time, self.n_in_node)
        inputs = inputs.transpose(1, 2).contiguous()
        reduced = self.cnn(inputs)  # [BxA, D', T']
        reduced = reduced.transpose(1, 2).view(-1, self.n_obj, self.out_n_time,
                                               self.out_channel)
        reduced = reduced.transpose(1, 2).contiguous()
        # reduced.shape = [batch_size, timesteps, atoms, dims]

        # inputs.shape = [batch_size, timesteps, atoms, dims=1]
        rel_type = rel_type.unsqueeze(1)
        receivers = torch.matmul(rel_rec, reduced)
        senders = torch.matmul(rel_send, reduced)
        pre_msg = torch.cat([receivers, senders], dim=-1)

        all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),
                               pre_msg.size(2), self.msg_out,
                               dtype=torch.float32, device=pre_msg.device)

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exclude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            # [batch_size, timesteps, A(A-1), dim]

            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))

            msg = torch.transpose(msg, 1, 2)
            temp_rel_type = torch.transpose(rel_type[:, 0:1, :, i:i + 1], 1, 2)

            msg = torch.mul(msg, temp_rel_type)
            msg = torch.transpose(msg, 1, 2).contiguous()

            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()
        # [B, A, T', D']

        aug_inputs = torch.cat([reduced, agg_msgs], dim=-1)

        aug_inputs = aug_inputs.view(-1, self.out_fc_dim)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = self.out_fc2(pred)

        return pred


if __name__ == "__main__":
    print('asdf')