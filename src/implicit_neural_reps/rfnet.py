import torch

from src.implicit_neural_reps.mlp import MLPBase


class RFNet(MLPBase):
    def __init__(
        self,
        data_dim: int,
        out_channels: int,
        num_hidden: int,
        num_layers: int,
        omega_0: float,
        nonlinear_class: type[torch.nn.Module],
        norm_class: type[torch.nn.Module],
        use_bias: bool,
        **kwargs,
    ):
        # construct the hidden and out layers of the network
        super().__init__(
            out_channels=out_channels,
            num_hidden=num_hidden,
            num_layers=num_layers,
            nonlinear_class=nonlinear_class,
            norm_class=norm_class,
            use_bias=use_bias,
        )
        # Construct input embedding
        self.input_layers = RandomFourierEmbedding(
            data_dim=data_dim, out_channels=num_hidden, omega_0=omega_0, use_bias=use_bias
        )


class RandomFourierEmbedding(torch.nn.Module):
    def __init__(
        self,
        data_dim: int,
        out_channels: int,
        omega_0: float,
        use_bias: bool,
    ):
        if out_channels % 2 != 0:
            raise ValueError(f"out_channels must be even. Current {out_channels}")

        super().__init__()

        linear_out_channels = out_channels // 2
        self.linear = torch.nn.Linear(
            in_features=data_dim, out_features=linear_out_channels, bias=use_bias
        )
        # Initialize:
        self.linear.weight.data.normal_(0.0, 2 * torch.pi * omega_0)
        if use_bias:
            torch.nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        out = self.linear(x)
        return torch.cat([torch.cos(out), torch.sin(out)], dim=-1)
