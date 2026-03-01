
import torch
import torch.nn as nn


def get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    if name == "tanh":
        return nn.Tanh
    if name == "gelu":
        return nn.GELU
    return nn.ReLU


class MLPClassifier(nn.Module):
    """
    MLP (Multi-Layer Perceptron) pour classification multi-classes.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int = 3,
        hidden_units=(128, 64),
        activation: str = "relu",
        dropout: float = 0.3,
        batch_norm: bool = True,
    ):
        super().__init__()

        Act = get_activation(activation)

        layers = []
        in_dim = input_dim

        for units in hidden_units:
            layers.append(nn.Linear(in_dim, units))
            if batch_norm:
                layers.append(nn.BatchNorm1d(units))
            layers.append(Act())
            layers.append(nn.Dropout(dropout))
            in_dim = units

        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

        self._init_weights(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _init_weights(self, activation: str):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if activation.lower() in {"relu", "gelu"}:
                    nn.init.kaiming_normal_(m.weight)
                else:
                    nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)