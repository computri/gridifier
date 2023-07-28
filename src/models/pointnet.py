# from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_classification.py

import os.path as osp

import torch

from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius, knn
from torch_geometric.nn import MLP, knn_interpolate
from src import nn as src_nn



class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r

        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointNet2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, cfg):
        super().__init__()

        # Input channels account for both `pos` and node features.
        num_hidden = cfg.net.num_hidden
        connectivity = cfg.pointnet.connectivity
        self.dimensions = 3

        self.sa1_module = src_nn.SAModule(data_dim=self.dimensions,
                                          in_channels=in_channels,
                                          out_channels=num_hidden,
                                          mlp_num_hidden=int(num_hidden / 2),
                                          mlp_num_layers=2,
                                          mlp_nonlinear="relu",
                                          connectivity=connectivity,
                                          sampling_ratio=0.5,
                                          sampling_radius=0.2,
                                          max_num_neighbors=64,
                                          num_workers=1,
                                         )

        self.sa2_module = src_nn.SAModule(data_dim=self.dimensions,
                                          in_channels=num_hidden,
                                          out_channels=num_hidden * 2,
                                          mlp_num_hidden=num_hidden,
                                          mlp_num_layers=2,
                                          mlp_nonlinear="relu",
                                          connectivity=connectivity,
                                          sampling_ratio=0.25,
                                          sampling_radius=0.4,
                                          max_num_neighbors=64,
                                          num_workers=1,
                                         )

        # self.sa1_module = SAModule(0.5, 0.2, MLP([3 + 3, 64, 64, 128]))
        # self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([num_hidden * 2 + self.dimensions, num_hidden * 2, num_hidden * 4, num_hidden * 8]))

        self.mlp = MLP([num_hidden * 8, num_hidden * 4, num_hidden * 2, out_channels], dropout=0.5, norm=None)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        out = self.mlp(x)

        return out



class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class PointNet2Segmentation(torch.nn.Module):
    def __init__(self, in_channels, out_channels, cfg):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.2, 0.2, MLP([in_channels + 3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
        self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
        self.fp1_module = FPModule(3, MLP([128 + 3, 128, 128, 128]))

        self.mlp = MLP([128, 128, 128, out_channels], dropout=0.5, norm=None)


    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        return self.mlp(x)

