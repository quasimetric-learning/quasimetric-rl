from typing import *

import attrs

import torch
import torch.nn.functional as F

from ....data import BatchData

from ...utils import LatentTensor, LossResult
from ..models import QuasimetricCritic

from . import CriticLossBase, CriticBatchInfo



class GlobalPushLoss(CriticLossBase):
    @attrs.define(kw_only=True)
    class Conf:
        # config / argparse uses this to specify behavior

        # smaller => smoother loss
        softplus_beta: float = attrs.field(default=0.1, validator=attrs.validators.gt(0))

        # should be greater than most GT distances between sampled pairs
        softplus_offset: float = attrs.field(default=15, validator=attrs.validators.ge(0))

        def make(self) -> 'GlobalPushLoss':
            return GlobalPushLoss(
                softplus_beta=self.softplus_beta,
                softplus_offset=self.softplus_offset,
            )

    softplus_beta: float
    softplus_offset: float

    def __init__(self, *, softplus_beta: float, softplus_offset: float):
        super().__init__()
        self.softplus_beta = softplus_beta
        self.softplus_offset = softplus_offset

    def forward(self, data: BatchData, critic_batch_info: CriticBatchInfo) -> LossResult:
        # To randomly pair zx, zy, we just roll over zy by 1, because zx and zy
        # are latents of randomly ordered random batches.
        dists = critic_batch_info.critic.quasimetric_model(critic_batch_info.zx, torch.roll(critic_batch_info.zy, 1, dims=0))
        # Sec 3.2. Transform so that we penalize large distances less.
        tsfm_dist: torch.Tensor = F.softplus(self.softplus_offset - dists, beta=self.softplus_beta)
        tsfm_dist = tsfm_dist.mean()
        return LossResult(loss=tsfm_dist, info=dict(dist=dists.mean(), tsfm_dist=tsfm_dist))

    def extra_repr(self) -> str:
        return f"softplus_beta={self.softplus_beta:g}, softplus_offset={self.softplus_offset:g}"
