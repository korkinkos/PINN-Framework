from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Loss(ABC):
    @abstractmethod
    def forward(self, residual: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, residual: Tensor) -> Tensor:
        return self.forward(residual)


class MSELoss(Loss):
    def forward(self, residual: Tensor) -> Tensor:
        return torch.mean(residual**2)


class MAELoss(Loss):
    def forward(self, residual: Tensor) -> Tensor:
        return torch.mean(torch.abs(residual))


class LossAggregator:
    def compute(self, problem, model, points_by_constraint):
        losses = {}
        total = None

        for name, constraint in problem.constraints.items():
            points = points_by_constraint[name]
            raw_loss = constraint.loss(model, points)
            weighted_loss = constraint.weight * raw_loss

            losses[name] = raw_loss
            total = (
                weighted_loss
                if total is None
                else total + weighted_loss
            )

        if total is None:
            raise ValueError("Problem has no constraints")

        losses["total"] = total
        return losses