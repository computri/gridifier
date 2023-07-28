import torch
import torch.nn as nn
import torch_geometric.nn as tg_nn
from ml_collections import config_dict

from src import implicit_neural_reps as inr


class SAModule(torch.nn.Module):
    def __init__(
        self,
        data_dim: int,
        in_channels: int,
        out_channels: int,
        mlp_num_hidden: int,
        mlp_num_layers: int,
        mlp_nonlinear: str,
        sampling_ratio: float,
        sampling_radius: float,
        num_workers: int,
        connectivity: str = "radius",
        max_num_neighbors: int = 64,
    ):
        super().__init__()

        self.sampling_ratio = sampling_ratio
        self.sampling_radius = sampling_radius
        self.max_num_neighbors = max_num_neighbors
        self.num_workers = num_workers

        assert (connectivity == "radius" and sampling_radius is not None) or (connectivity == "knn" and max_num_neighbors is not None), "Radius needs r or knn needs k."

        if connectivity == "knn":
            self.connectivity = lambda x, y, batch_x, batch_y: tg_nn.knn(x, y, self.max_num_neighbors, batch_x, batch_y)
        else:
            self.connectivity = lambda x, y, batch_x, batch_y: tg_nn.radius(x, y, self.sampling_radius, batch_x, batch_y, max_num_neighbors=self.max_num_neighbors)

        # TODO(dwromero): Make the MLP a RFNet or smt like that.
        mlp_channels = (
            [in_channels + data_dim] + [mlp_num_hidden] * mlp_num_layers + [out_channels]
        )
        nn = tg_nn.MLP(mlp_channels, act=mlp_nonlinear)
        self.conv = tg_nn.PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = tg_nn.fps(pos, batch, ratio=self.sampling_ratio)

        row, col = self.connectivity(pos, pos[idx], batch, batch[idx])

        edge_index = torch.stack([col, row], dim=0)

        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]

        return x, pos, batch


def _create_embedding(embedding_cfg: config_dict.ConfigDict, in_channels: int, out_channels: int):
    """Create an embedding, i.e., a neural network or implicit neural repr from a config dict."""
    embedding_type = embedding_cfg.type
    embedding_num_hidden = embedding_cfg.num_hidden
    embedding_num_layers = embedding_cfg.num_layers
    embedding_omega_0 = embedding_cfg.omega_0
    embedding_use_bias = embedding_cfg.use_bias
    if embedding_cfg.norm == "BatchNorm":
        norm_type = f"{embedding_cfg.norm}1d"
    else:
        norm_type = embedding_cfg.norm
    embedding_norm_class = getattr(nn, norm_type)
    embedding_nonlinear_class = getattr(nn, embedding_cfg.nonlinearity)

    if embedding_type:  # i.e., not empty string
        embedding_class = getattr(inr, embedding_type)
        embedding = embedding_class(
            data_dim=in_channels,
            out_channels=out_channels,
            num_hidden=embedding_num_hidden,
            num_layers=embedding_num_layers,
            omega_0=embedding_omega_0,
            use_bias=embedding_use_bias,
            norm_class=embedding_norm_class,
            nonlinear_class=embedding_nonlinear_class,
        )
    else:
        embedding = nn.Identity()
    return embedding


class BipartiteConv(tg_nn.MessagePassing):
    """This layer builds knn connectivity between source and target, and propagates the feature
    values along the edges.

    Additionally, positional information can be embedded.
    """

    def __init__(
        self,
        data_dim: int,
        in_channels: int,
        out_channels: int,
        num_neighbors: int,
        conditioning: str,
        aggregation: str,
        use_target_positions: bool,
        use_normals: bool,
        position_embed_cfg: config_dict.ConfigDict,
        update_net_cfg: config_dict.ConfigDict,
        node_embed_cfg: config_dict.ConfigDict,
        message_net_cfg: config_dict.ConfigDict,
        **kwargs,
    ):
        if aggregation not in ["mean", "max", "add"]:
            raise ValueError(f"Unknown aggregation type: {aggregation}")

        super().__init__(flow="source_to_target", aggr=aggregation, **kwargs)

        self.data_dim = data_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_neighbors = num_neighbors
        self.conditioning = conditioning
        self.use_target_positions = use_target_positions
        self.use_normals = use_normals

        # Create positional embedding
        if self.conditioning == "distance":
            pos_emb_dim = 1
        elif self.conditioning in ["rel_pos", "absolute"]:
            pos_emb_dim = data_dim
        elif self.conditioning == "":
            pos_emb_dim = 0
        else:
            raise ValueError(f"Unknown conditioning type: {self.conditioning}")

        self.positional_embedding = _create_embedding(embedding_cfg=position_embed_cfg,
                                                      in_channels=pos_emb_dim,
                                                      out_channels=out_channels)
        self.update_net = _create_embedding(embedding_cfg=update_net_cfg,
                                            in_channels=out_channels,
                                            out_channels=out_channels)

        if in_channels > 0:
            self.node_embed_net = _create_embedding(embedding_cfg=node_embed_cfg,
                                                    in_channels=in_channels,
                                                    out_channels=out_channels)
            in_channels = out_channels
        else:
            self.node_embed_net = nn.Identity()

        self.message_net = _create_embedding(embedding_cfg=message_net_cfg,
                                             in_channels=in_channels + out_channels + (self.use_target_positions * 3),
                                             out_channels=out_channels)

    def forward(self, edge_index, x_in, f_in, batch_x, x_out, batch_y, normals=None):
        # retrieve features at the target positions
        return self.propagate(
            edge_index=edge_index,
            x_in=x_in,
            f_in=f_in,
            batch_x=batch_x,
            x_out=x_out,
            batch_y=batch_y,
            normals=normals,
        )

    def message(self, f_in_j, x_in_j, x_out_i, normals_j):
        if self.conditioning == "rel_pos":
            pos_embedding = x_out_i - x_in_j
        elif self.conditioning == "absolute":
            pos_embedding = (x_out_i - x_in_j) ** 2
        elif self.conditioning == "distance":
            pos_embedding = torch.sqrt(((x_out_i - x_in_j) ** 2).sum(dim=-1)).unsqueeze(-1)
        elif self.conditioning == "":
            pos_embedding = None
        else:
            raise ValueError(f"Unknown conditioning type: {self.conditioning}")

        if self.positional_embedding is not None and pos_embedding is not None:
            pos_embedding = self.positional_embedding(pos_embedding)

        if pos_embedding is not None:
            message_in = pos_embedding
        if self.use_target_positions:
            message_in = torch.cat((x_out_i, message_in), dim=-1)
        if f_in_j is not None:
            node_embedding = self.node_embed_net(f_in_j)
            # If the message net is an identity, we multiply the node embedding and the positional embedding
            if pos_embedding is not None:
                if isinstance(self.message_net, nn.Identity):
                    message_in = message_in * node_embedding
                else:
                    message_in = torch.cat((node_embedding, message_in), dim=-1)
            else:
                message_in = node_embedding

        result = self.message_net(message_in)
        return result

    def update(self, message, f_in, x_in, x_out, batch_x, batch_y):
        f_out = self.update_net(message)
        return (
            f_out,
            x_out,
            batch_y,
        )  # return the target batch to use it as the input batch for the next layer

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"data_dim={self.data_dim}, "
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"num_neighbors={self.num_neighbors}, "
            f"conditioning={self.conditioning})\n"
            f"positional_embedding={self.positional_embedding}\n"
            f"update_net={self.update_net}\n"
            f"node_embed_net={self.node_embed_net}\n"
            f"message_net={self.message_net}\n"
        )
