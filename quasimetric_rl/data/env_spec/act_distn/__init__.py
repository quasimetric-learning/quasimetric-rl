from typing import *
from typing_extensions import Protocol, Final

import abc

import gym.spaces

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
import torch.distributions.constraints



#-----------------------------------------------------------------------------#
#-------------------------------- output API ---------------------------------#
#-----------------------------------------------------------------------------#


class TensorDistributionProtocol(Protocol):
    batch_shape: torch.Size
    event_shape: torch.Size

    mode: torch.Tensor
    mean: torch.Tensor

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        pass

    def rsample(self, sample_shape=torch.Size()) -> torch.Tensor:
        pass

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        pass

    def entropy(self) -> torch.Tensor:
        pass


class ActionOutputConverter(nn.Module, metaclass=abc.ABCMeta):
    input_size: int

    def __init__(self, action_space: gym.spaces.Box) -> None:
        super().__init__()

    @abc.abstractmethod
    def forward(self, feature: torch.Tensor) -> TensorDistributionProtocol:
        pass

    def __call__(self, feature: torch.Tensor) -> TensorDistributionProtocol:
        return super().__call__(feature)


#-----------------------------------------------------------------------------#
#----------------------------------- impls -----------------------------------#
#-----------------------------------------------------------------------------#


class DiscreteOutputOneHot(ActionOutputConverter):
    input_size: Final[int]
    num_actions: Final[int]

    def __init__(self, action_space: gym.spaces.Discrete) -> None:
        super().__init__(action_space)
        self.input_size = self.num_actions = int(action_space.n)

    def forward(self, feature: torch.Tensor) -> torch.distributions.Distribution:
        return torch.distributions.Categorical(logits=feature)


class BoxOutputLinearNormalization(ActionOutputConverter):
    input_size: Final[int]

    kind: str
    mean: torch.Tensor
    half_len: torch.Tensor

    @property
    def output_distn_ty(self) -> str:
        return self.kind

    def __init__(self, action_space: gym.spaces.Box) -> None:
        super().__init__(action_space)
        self.input_size = torch.Size(action_space.shape).numel() * 2
        high = torch.as_tensor(action_space.high, dtype=torch.float32)
        low = torch.as_tensor(action_space.low, dtype=torch.float32)
        self.register_buffer('mean', (high + low) / 2)
        self.register_buffer('half_len', ((high - low) / 2).clamp_min(1e-3))
        assert torch.as_tensor(action_space.bounded_above & action_space.bounded_below).all(), "Must have bounded action space"

    def forward(self, feature: torch.Tensor) -> torch.distributions.Distribution:
        gmean, grawstd = feature.view(*feature.shape[:-1], 2, *self.mean.shape).unbind(dim=-self.mean.ndim - 1)
        distn = torch.distributions.Normal(
            loc=gmean,
            scale=F.softplus(grawstd) + 1e-4,
        )

        # Acme (CRL) Tanh Normal
        from .utils import AcmeTanhTransformedDistribution, SampleDist
        distn = AcmeTanhTransformedDistribution(
            distn,
        )
        distn = torch.distributions.TransformedDistribution(
            distn,
            torch.distributions.AffineTransform(loc=self.mean, scale=self.half_len),
        )
        distn = torch.distributions.Independent(
            distn,
            reinterpreted_batch_ndims=1,
        )

        distn = SampleDist(distn)

        return distn
