from typing import *

import attrs

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....data import BatchData

from ...utils import LatentTensor, LossResult, grad_mul, softplus_inv_float

from . import CriticLossBase, CriticBatchInfo



class LocalConstraintLoss(CriticLossBase):
    @attrs.define(kw_only=True)
    class Conf:
        # config / argparse uses this to specify behavior

        epsilon: float = attrs.field(default=0.25, validator=attrs.validators.gt(0))

        # Cost per step. If environment has variable costs, this can be changed
        # to load from data, and QRL will still have guarantees.
        step_cost: float = attrs.field(default=1, validator=attrs.validators.gt(0))

        init_lagrange_multiplier: float = attrs.field(default=0.01, validator=attrs.validators.gt(0))

        def make(self) -> 'LocalConstraintLoss':
            return LocalConstraintLoss(
                epsilon=self.epsilon,
                step_cost=self.step_cost,
                init_lagrange_multiplier=self.init_lagrange_multiplier,
            )

    epsilon: float
    step_cost: float
    init_lagrange_multiplier: float

    raw_lagrange_multiplier: nn.Parameter  # for the QRL constrained optimization

    def __init__(self, *, epsilon: float, step_cost: float, init_lagrange_multiplier: float):
        super().__init__()
        self.epsilon = epsilon
        self.step_cost = step_cost
        self.init_lagrange_multiplier = init_lagrange_multiplier
        self.raw_lagrange_multiplier = nn.Parameter(
            torch.tensor(softplus_inv_float(init_lagrange_multiplier), dtype=torch.float32))

    def forward(self, data: BatchData, critic_batch_info: CriticBatchInfo) -> LossResult:

        dist = critic_batch_info.critic.quasimetric_model(critic_batch_info.zx, critic_batch_info.zy)

        lagrange_mult = F.softplus(self.raw_lagrange_multiplier)  # make positive
        # lagrange multiplier is minimax training, so grad_mul -1
        lagrange_mult = grad_mul(lagrange_mult, -1)

        sq_deviation = (dist - self.step_cost).relu().square().mean()
        violation = (sq_deviation - self.epsilon ** 2)
        loss = violation * lagrange_mult

        return LossResult(
            loss=loss,
            info=dict(dist=dist.mean(), sq_deviation=sq_deviation, violation=violation, lagrange_mult=lagrange_mult),
        )

    def extra_repr(self) -> str:
        return f"epsilon={self.epsilon:g}, step_cost={self.step_cost:g}"
