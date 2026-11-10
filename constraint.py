from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn


ResidualFunction = Callable[[Tensor], Tensor]
LossFunction = Callable[[Tensor], Tensor]


def mean_squared_error(residual: Tensor) -> Tensor:
    return torch.mean(residual**2)


@dataclass
class Constraint:
    name: str
    residual_function: ResidualFunction
    loss_function: LossFunction = mean_squared_error
    weight: float = 1.0

    def residual(self, model: nn.Module, points: Tensor) -> Tensor:
        return self.residual_function(model, points)

    def loss(self, model: nn.Module, points: Tensor) -> Tensor:
        residual = self.residual(model, points)
        return self.loss_function(residual)
