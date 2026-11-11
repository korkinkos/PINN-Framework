from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor, nn

from sampler import Sampler
from loss import *

ResidualFunction = Callable[[Tensor], Tensor]
LossFunction = Callable[[Tensor], Tensor]


def mean_squared_error(residual: Tensor) -> Tensor:
    return torch.mean(residual**2)


@dataclass
class Constraint:
    name: str
    residual_function: ResidualFunction
    loss_function: Loss = field(default_factory=MSELoss)
    weight: float = 1.0
    sampler: Sampler = None

    def residual(self, model: nn.Module, points: Tensor) -> Tensor:
        return self.residual_function(model, points)

    def compute_loss(self, model: nn.Module, points: Tensor) -> Tensor:
        residual = self.residual(model, points)
        return self.loss_function(residual)
