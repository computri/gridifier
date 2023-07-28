import torch


class WeightedSin(torch.nn.Module):
    """Wraps torch.sin in a torch.nn.Module."""

    def __init__(self, omega_0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x):
        return torch.sin(self.omega_0 * x)


class SIREN(torch.nn.Module):
    def __init__(
        self,
        data_dim: int,
        out_channels: int,
        num_hidden: int,
        num_layers: int,
        omega_0: float,
        use_bias: bool,
        bias_init_type: str,
        **kwargs,
    ):
        super().__init__()

        # 1st layer:
        input_layers = [
            torch.nn.Linear(in_features=data_dim, out_features=num_hidden, bias=use_bias),
            WeightedSin(omega_0=omega_0),
        ]
        self.input_layers = torch.nn.Sequential(*input_layers)
        # Hidden layers
        hidden_layers = []
        for _ in range(num_layers - 2):
            hidden_layers.extend(
                [
                    torch.nn.Linear(in_features=data_dim, out_features=num_hidden, bias=use_bias),
                    WeightedSin(omega_0=omega_0),
                ]
            )
        self.hidden_layers = torch.nn.Sequential(*hidden_layers)
        # Out layer
        self.output_linear = torch.nn.Linear(
            in_features=num_hidden, out_features=out_channels, bias=use_bias
        )
        self.initialize_siren(omega_0=omega_0, bias_init_type=bias_init_type)

    def forward(self, x):
        out = self.input_layers(x)
        out = self.hidden_layers(out)
        return self.output_linear(out)

    def initialize_siren(self, omega_0, bias_init_type):
        def initialize_bias(tensor, value):
            if tensor is not None:
                if bias_init_type == "zero":
                    torch.nn.init.constant_(tensor, val=value)
                elif bias_init_type == "uniform":
                    torch.nn.init.uniform_(tensor, a=-value, b=value)
                else:
                    raise ValueError(
                        f"bias_init_type must be 'zero' or 'uniform'. Current: {bias_init_type}"
                    )

        # Initialize first layers
        w_std = 1.0 / self.hidden_layers[0].weight.shape[1]
        torch.nn.init.uniform_(self.hidden_layers[0].weight, -w_std, w_std)
        initialize_bias(self.hidden_layers[0].bias, w_std)

        # Initialize middle layers
        for m in self.hidden_layers.modules():
            if isinstance(m, torch.nn.Linear):
                w_std = torch.sqrt(6.0 / m.weight.shape[1]) / omega_0
                torch.nn.init.uniform_(m.weight, -w_std, w_std)
                initialize_bias(m.bias, w_std)

        # Initialize final layer
        torch.nn.init.kaiming_uniform_(self.output_linear.weight, nonlinearity="linear")
        # TODO(dwromero): Define based on the nonlin used in the main network.
        if self.output_linear.bias is not None:
            torch.nn.init.constant_(self.output_linear.bias, 0.0)
