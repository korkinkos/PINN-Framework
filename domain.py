import torch
from dataclasses import dataclass
from torch import Tensor


@dataclass
class Domain:
    lower: Tensor
    upper: Tensor

    def __post_init__(self):
        if self.lower.ndim != 1 or self.upper.ndim != 1:
            raise ValueError("Domain bounds must be 1D tensors")

        if self.lower.shape != self.upper.shape:
            raise ValueError("Domain bounds must have identical shapes")

        if not torch.all(self.lower < self.upper):
            raise ValueError("Lower bounds must be smaller than upper bounds")

    @property
    def dim(self) -> int:
        return self.lower.numel()