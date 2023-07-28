import torch
from ml_collections import config_dict
from torch import nn

from src import implicit_neural_reps as inr
from src import utils as src_utils


class CKConv(torch.nn.Module):
    """Depth separable CKConv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        data_dim: int,
        kernel_cfg: config_dict.ConfigDict,
        kernel_size: int,
        groups: int,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.data_dim = data_dim
        self.groups = groups

        # Create kernel
        kernel_type = kernel_cfg.type
        kernel_num_hidden = kernel_cfg.num_hidden
        kernel_num_layers = kernel_cfg.num_layers
        kernel_omega_0 = kernel_cfg.omega_0
        kernel_use_bias = kernel_cfg.use_bias
        if kernel_cfg.norm == "BatchNorm":
            norm_type = f"{kernel_cfg.norm}{data_dim}d"
        else:
            norm_type = kernel_cfg.norm
        kernel_norm_class = getattr(nn, norm_type)
        kernel_nonlinear_class = getattr(nn, kernel_cfg.nonlinearity)

        kernel_class = getattr(inr, kernel_type)
        self.kernel_generator = kernel_class(
            data_dim=data_dim,
            out_channels=in_channels,
            num_hidden=kernel_num_hidden,
            num_layers=kernel_num_layers,
            omega_0=kernel_omega_0,
            use_bias=kernel_use_bias,
            norm_class=kernel_norm_class,
            nonlinear_class=kernel_nonlinear_class,
        )
        # Use chang initialize:
        norm_factor = 1.0 / torch.sqrt(torch.tensor(kernel_size**3))
        self.kernel_generator.output_linear.weight.data *= norm_factor

        grid = src_utils.create_coordinate_grid(
            size=kernel_size, dimensions=data_dim, domain=(-1.0, 1.0), as_list=False
        )
        self.register_buffer("grid", grid)

        self.conv_fn = getattr(torch.nn.functional, f"conv{self.data_dim}d")

    def forward(self, x):
        # Compute kernel
        # The kernel must be permuted to be of shape [in_channels, 1, kernel_size, kernel_size, kernel_size] for it to
        # be compatible with a depth-wise convolution implemented with groups=in_channels.
        kernel = self.kernel_generator(self.grid).permute(3, 0, 1, 2).unsqueeze(1)
        # Convolve
        out = self.conv_fn(input=x, weight=kernel, bias=None, padding="same", groups=self.groups)
        return out
