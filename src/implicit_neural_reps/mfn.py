import numpy as np
import torch


def gaussian_window(x, gamma, mu):
    return torch.exp(-0.5 * ((gamma * (x.unsqueeze(dim=-2) - mu)) ** 2).sum(dim=-1))


class FourierLayer(torch.nn.Module):
    """Sine filter as used in FourierNet."""

    def __init__(
        self,
        data_dim: int,
        num_hidden: int,
        omega_0: float,
    ):
        super().__init__()

        self.linear = torch.nn.Linear(in_features=data_dim, out_features=num_hidden)
        # Initialize
        w_std = 1.0 / self.linear.weight.shape[1]  # Following Sitzmann et al. 2020
        init_value = 2.0 * np.pi * omega_0 * w_std
        self.linear.weight.data.uniform_(-init_value, init_value)
        if self.linear.bias is not None:
            torch.nn.init.constant_(self.linear.bias, 0.0)
        # Save params in self
        self.omega_0 = omega_0

    def forward(self, x):
        return torch.sin(self.linear(x))

    def extra_repr(self):
        return f"omega_0={self.omega_0}"


class GaborLayer(torch.nn.Module):
    """Gabor-like filter as used in GaborNet."""

    def __init__(
        self,
        data_dim: int,
        num_hidden: int,
        omega_0: float,
        alpha: float,
        beta: float,
        use_bias: bool,
        init_spatial_value: float,
    ):
        super().__init__()

        # Construct & initialize parameters
        # """
        # Background:
        # If we use a 2D mask, we must initialize the mask around 0. If we use a 1D
        # mask, however, we initialize the mean to 1. That is, the last sequence
        # element. As a result, we must also recenter the mu-values around 1.0.
        # In addition, the elements at the positive size are not used at the beginning.
        # Hence, we are only interested in elements on the negative size of the line.
        # """
        mu = init_spatial_value * (2 * torch.rand(num_hidden, data_dim) - 1)
        gamma = torch.distributions.gamma.Gamma(alpha, beta).sample((num_hidden, 1))  # Isotropic
        self.mu = torch.nn.Parameter(mu)
        self.gamma = torch.nn.Parameter(gamma)

        self.linear = torch.nn.Linear(in_features=data_dim, out_features=num_hidden, bias=use_bias)
        # Initialize
        w_std = 1.0 / self.linear.weight.shape[1]  # Following Sitzmann et al. 2020
        init_value = 2.0 * np.pi * omega_0 * w_std
        self.linear.weight.data.uniform_(-init_value, init_value)
        if self.linear.bias is not None:
            torch.nn.init.constant_(self.linear.bias, 0.0)

        self.omega_0 = omega_0
        self.alpha = alpha
        self.beta = beta
        self.init_spatial_value = init_spatial_value

    def forward(self, x):
        return gaussian_window(
            x,
            self.gamma.view(*(1,) * self.data_dim, *self.gamma.shape),
            self.mu.view(*(1,) * self.data_dim, *self.mu.shape),
        ) * torch.sin(self.linear(x))

    def extra_repr(self):
        return (
            f"omega_0={self.omega_0}, alpha={self.alpha}, beta={self.beta}, "
            f"init_spatial_value={self.init_spatial_value}"
        )


class MAGNetLayer(torch.nn.Module):
    """MAGNet layer as used in MAGNets (FlexConv)."""

    def __init__(
        self,
        data_dim: int,
        num_hidden: int,
        omega_0: float,
        alpha: float,
        beta: float,
        use_bias: bool,
        init_spatial_value: float,
    ):
        super().__init__()

        # Construct & initialize parameters
        # """
        # Background:
        # If we use a 2D mask, we must initialize the mask around 0. If we use a 1D
        # mask, however, we initialize the mean to 1. That is, the last sequence
        # element. As a result, we must also recenter the mu-values around 1.0.
        # In addition, the elements at the positive size are not used at the beginning.
        # Hence, we are only interested in elements on the negative size of the line.
        # """
        mu = init_spatial_value * (2 * torch.rand(num_hidden, data_dim) - 1)
        gamma = torch.distributions.gamma.Gamma(alpha, beta).sample((num_hidden, data_dim))
        self.mu = torch.nn.Parameter(mu)
        self.gamma = torch.nn.Parameter(gamma)

        self.linear = torch.nn.Linear(in_features=data_dim, out_features=num_hidden, bias=use_bias)
        # Initialize
        w_std = 1.0 / self.linear.weight.shape[1]  # Following Sitzmann et al. 2020
        init_value = 2 * np.pi * omega_0 * self.gamma * w_std
        self.linear.weight.data.uniform_(-init_value, init_value)
        if self.linear.bias is not None:
            torch.nn.init.constant_(self.linear.bias, 0.0)

        self.omega_0 = omega_0
        self.alpha = alpha
        self.beta = beta
        self.init_spatial_value = init_spatial_value

    def forward(self, x):
        return gaussian_window(
            x,
            self.gamma.view(*(1,) * self.data_dim, *self.gamma.shape),
            self.mu.view(*(1,) * self.data_dim, *self.mu.shape),
        ) * torch.sin(self.linear(x))


class MFNBase(torch.nn.Module):
    """Multiplicative filter network base class.

    Expects the child class to define the 'filters' attribute, which should be a nn.ModuleList of
    n_layers+1 filters with output equal to hidden_size.
    """

    def __init__(
        self,
        out_channels: int,
        num_hidden: int,
        num_layers: int,
        use_bias: bool,
    ):
        super().__init__()

        self.linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(in_features=num_hidden, out_features=num_hidden, bias=use_bias)
                for _ in range(num_layers)
            ]
        )
        self.output_linear = torch.nn.Linear(
            in_features=num_hidden, out_features=out_channels, bias=use_bias
        )

    def forward(self, x):
        out = self.filters[0](x)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x) * self.linears[i - 1](out)
        out = self.output_linear(out)
        return out


class FourierNet(MFNBase):
    def __init__(
        self,
        data_dim: int,
        out_channels: int,
        num_hidden: int,
        num_layers: int,
        omega_0: float,
        use_bias: bool,
        **kwargs,
    ):
        super().__init__(
            out_channels=out_channels,
            num_hidden=num_hidden,
            num_layers=num_layers,
            use_bias=use_bias,
        )

        self.filters = torch.nn.ModuleList(
            [
                FourierLayer(data_dim=data_dim, num_hidden=num_hidden, omega_0=omega_0)
                for _ in range(num_layers + 1)
            ]
        )
        # Initialize
        var_g = 0.5
        with torch.no_grad():
            # Init so that all freq. components have the same amplitude on initialization
            accumulated_weight = None
            for idx, lin in enumerate(self.linears):
                layer = idx + 1
                torch.nn.init.orthogonal_(lin.weight)
                lin.weight.data *= np.sqrt(1.0 / var_g)

                # Get norm of weights so far.
                if accumulated_weight is None:
                    accumulated_weight = lin.weight.data.clone()
                else:
                    accumulated_weight = torch.einsum(
                        "ab,bc->ac", lin.weight.data, accumulated_weight
                    )
                accumulated_value = accumulated_weight.sum(dim=-1)

                # Initialize the bias
                if lin.bias is not None:
                    lin.bias.data = (accumulated_value / (num_hidden * 2.0**layer)).flatten()

        # Initialize output_linear layer
        torch.nn.init.kaiming_uniform_(self.output_linear.weight, nonlinearity="linear")
        self.output_linear.bias.data.fill_(0.0)


class GaborNet(MFNBase):
    def __init__(
        self,
        data_dim: int,
        out_channels: int,
        num_hidden: int,
        num_layers: int,
        omega_0: float,
        use_bias: bool,
        alpha: float = 6.0,
        beta: float = 1.0,
        init_spatial_value: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            out_channels=out_channels,
            num_hidden=num_hidden,
            num_layers=num_layers,
            use_bias=use_bias,
        )

        self.filters = torch.nn.ModuleList(
            [
                GaborLayer(
                    data_dim=data_dim,
                    num_hidden=num_hidden,
                    omega_0=omega_0,
                    alpha=alpha / (num_hidden + 1),
                    beta=beta,
                    use_bias=use_bias,
                    init_spatial_value=init_spatial_value,
                )
                for _ in range(num_layers + 1)
            ]
        )
        # Initialize
        for idx, lin in enumerate(self.linears):
            torch.nn.init.kaiming_uniform_(lin.weight, nonlinearity="linear")
            if lin.bias is not None:
                torch.nn.init.constant_(lin.bias, 1.0)
        torch.nn.init.kaiming_uniform_(self.output_linear.weight, nonlinearity="linear")
        if self.output_linear.bias is not None:
            torch.nn.init.constant_(self.output_linear.bias, 0.0)


class MAGNet(MFNBase):
    """Multiplicative Anisotropic Gabor Network."""

    def __init__(
        self,
        data_dim: int,
        out_channels: int,
        num_hidden: int,
        num_layers: int,
        omega_0: float,
        use_bias: bool,
        alpha: float = 6.0,
        beta: float = 1.0,
        init_spatial_value: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            out_channels=out_channels,
            num_hidden=num_hidden,
            num_layers=num_layers,
            use_bias=use_bias,
        )

        self.filters = torch.nn.ModuleList(
            [
                MAGNetLayer(
                    data_dim=data_dim,
                    num_hidden=num_hidden,
                    omega_0=omega_0,
                    alpha=alpha / (layer + 1),
                    beta=beta,
                    use_bias=use_bias,
                    init_spatial_value=init_spatial_value,
                )
                for layer in range(num_layers + 1)
            ]
        )
        # Initialize
        for idx, lin in enumerate(self.linears):
            torch.nn.init.kaiming_uniform_(lin.weight, nonlinearity="linear")
            if lin.bias is not None:
                torch.nn.init.constant_(lin.bias, 1.0)
        torch.nn.init.kaiming_uniform_(self.output_linear.weight, nonlinearity="linear")
        if self.output_linear.bias is not None:
            torch.nn.init.constant_(self.output_linear.bias, 0.0)
