from abc import ABC, abstractmethod

import torch
from torch import Tensor

from domain import Domain


class Sampler(ABC):
    @abstractmethod
    def sample(self, n: int) -> Tensor:
        raise NotImplementedError


class UniformSampler(Sampler):
    def __init__(self, domain: Domain):
        self.domain = domain

    def sample(self, n: int) -> Tensor:
        random_points = torch.rand(n, self.domain.dim)

        return (
                self.domain.lower
                + random_points * (self.domain.upper - self.domain.lower)
        )