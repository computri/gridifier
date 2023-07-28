import torch


class MLPBase(torch.nn.Module):
    def __init__(
        self,
        out_channels: int,
        num_hidden: int,
        num_layers: int,
        nonlinear_class: type[torch.nn.Module],
        norm_class: type[torch.nn.Module],
        use_bias: bool,
    ):
        super().__init__()

        hidden_layers = []
        for _ in range(num_layers - 2):
            hidden_layers.extend(
                [
                    torch.nn.Linear(
                        in_features=num_hidden, out_features=num_hidden, bias=use_bias
                    ),
                    norm_class(num_hidden),
                    nonlinear_class(),
                ]
            )
        self.hidden_layers = torch.nn.Sequential(*hidden_layers)
        self.output_linear = torch.nn.Linear(
            in_features=num_hidden, out_features=out_channels, bias=use_bias
        )
        self.initialize(nonlinear_class)

    def forward(self, x):
        out = self.input_layers(x)
        out = self.hidden_layers(out)
        return self.output_linear(out)

    def initialize(self, nonlinear_class: type[torch.nn.Module]):
        # Define the gain
        if nonlinear_class == torch.nn.ReLU:
            nonlin = "relu"
        elif nonlinear_class == torch.nn.LeakyReLU:
            nonlin = "leaky_relu"
        else:
            nonlin = "linear"
        # Initialize hidden layers
        for i, m in enumerate(self.hidden_layers.modules()):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlin)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)
        # Initialize output layer
        torch.nn.init.kaiming_uniform_(self.output_linear.weight, nonlinearity="linear")
        # TODO(dwromero): Define based on the nonlin used in the main network.
        if self.output_linear.bias is not None:
            self.output_linear.bias.data.fill_(0.0)


class MLP(MLPBase):
    def __init__(
        self,
        data_dim: int,
        out_channels: int,
        num_hidden: int,
        num_layers: int,
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
        # Construct 1st layers (Sequence of modules, e.g., input embeddings)
        input_layers = [
            torch.nn.Linear(in_features=data_dim, out_features=num_hidden, bias=use_bias),
            norm_class(num_hidden),
            nonlinear_class(),
        ]
        self.input_layers = torch.nn.Sequential(*input_layers)
        self.initialize_input_layers(nonlinear_class)

    def initialize_input_layers(self, nonlinear_class: type[torch.nn.Module]):
        # Define the gain
        if nonlinear_class == torch.nn.ReLU:
            nonlin = "relu"
        elif nonlinear_class == torch.nn.LeakyReLU:
            nonlin = "leaky_relu"
        else:
            nonlin = "linear"

        for i, m in enumerate(self.input_layers.modules()):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlin)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
