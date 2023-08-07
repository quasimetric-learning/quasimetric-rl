from typing import *

import attrs

import torch

from ....data import BatchData

from ...utils import LossResult
from ..model import Actor
from ...quasimetric_critic import CriticBatchInfo

from . import ActorLossBase



class BCLoss(ActorLossBase):
    @attrs.define(kw_only=True)
    class Conf:
        # config / argparse uses this to specify behavior

        weight: float = attrs.field(default=0, validator=attrs.validators.ge(0))

        def make(self) -> 'BCLoss':
            return BCLoss(weight=self.weight)

    weight: float

    def __init__(self, *, weight: float):
        super().__init__()
        self.weight = weight

    def forward(self, actor: Actor, critic_batch_infos: Collection[CriticBatchInfo], data: BatchData) -> LossResult:
        if self.weight == 0:
            return LossResult(loss=0, info={})
        actor_distn = actor(data.observations, data.future_observations)
        log_prob: torch.Tensor = actor_distn.log_prob(data.actions).mean()
        loss = -log_prob * self.weight
        return LossResult(loss=loss, info=dict(log_prob=log_prob))

    def extra_repr(self) -> str:
        return f"weight={self.weight:g}"
