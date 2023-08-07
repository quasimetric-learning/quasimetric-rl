from typing import *

import abc
import attrs

import torch

from ....data import BatchData
from ...utils import LossBase, LossResult, LatentTensor
from ..models import QuasimetricCritic
from ...optim import OptimWrapper, AdamWSpec


@attrs.define(kw_only=True)
class CriticBatchInfo:
    r"""
    All critic outputs needed to compute losses for a single critic over a batch.
    """
    critic: QuasimetricCritic
    zx: LatentTensor
    zy: LatentTensor


class CriticLossBase(LossBase):
    @abc.abstractmethod
    def forward(self, data: BatchData, critic_batch_info: CriticBatchInfo) -> LossResult:
        pass

    # for type hints
    def __call__(self, data: BatchData, critic_batch_info: CriticBatchInfo) -> LossResult:
        return super().__call__(data, critic_batch_info)


from .global_push import GlobalPushLoss
from .local_constraint import LocalConstraintLoss
from .latent_dynamics import LatentDynamicsLoss


class QuasimetricCriticLosses(CriticLossBase):
    @attrs.define(kw_only=True)
    class Conf:
        global_push: GlobalPushLoss.Conf = GlobalPushLoss.Conf()
        local_constraint: LocalConstraintLoss.Conf = LocalConstraintLoss.Conf()
        latent_dynamics: LatentDynamicsLoss.Conf = LatentDynamicsLoss.Conf()

        critic_optim: AdamWSpec.Conf = AdamWSpec.Conf(lr=1e-4)
        lagrange_mult_optim: AdamWSpec.Conf = AdamWSpec.Conf(lr=1e-2)

        def make(self, critic: QuasimetricCritic, total_optim_steps: int) -> 'QuasimetricCriticLosses':
            return QuasimetricCriticLosses(
                critic,
                total_optim_steps=total_optim_steps,
                global_push=self.global_push.make(),
                local_constraint=self.local_constraint.make(),
                latent_dynamics=self.latent_dynamics.make(),
                critic_optim_spec=self.critic_optim.make(),
                lagrange_mult_optim_spec=self.lagrange_mult_optim.make(),
            )

    global_push: GlobalPushLoss
    local_constraint: LocalConstraintLoss
    latent_dynamics: LatentDynamicsLoss

    critic_optim: OptimWrapper
    critic_sched: torch.optim.lr_scheduler._LRScheduler
    lagrange_mult_optim: OptimWrapper
    lagrange_mult_sched: torch.optim.lr_scheduler._LRScheduler

    def __init__(self, critic: QuasimetricCritic, *, total_optim_steps: int, global_push: GlobalPushLoss,
                 local_constraint: LocalConstraintLoss, latent_dynamics: LatentDynamicsLoss,
                 critic_optim_spec: AdamWSpec, lagrange_mult_optim_spec: AdamWSpec):
        super().__init__()
        self.global_push = global_push
        self.local_constraint = local_constraint
        self.latent_dynamics = latent_dynamics

        self.critic_optim, self.critic_sched = critic_optim_spec.create_optim_scheduler(
            critic.parameters(), total_optim_steps)
        self.lagrange_mult_optim, self.lagrange_mult_sched = lagrange_mult_optim_spec.create_optim_scheduler(
            local_constraint.parameters(), total_optim_steps)
        assert len(list(local_constraint.parameters())) == 1


    def forward(self, data: BatchData, critic_batch_info: CriticBatchInfo, *,
                optimize: bool = True) -> LossResult:
        with self.critic_optim.update_context(optimize=optimize), \
                self.lagrange_mult_optim.update_context(optimize=optimize):

            result = LossResult.combine(dict(
                global_push=self.global_push(data, critic_batch_info),
                local_constraint=self.local_constraint(data, critic_batch_info),
                latent_dynamics=self.latent_dynamics(data, critic_batch_info),
            ))
            result.loss.backward()

        if optimize:
            self.critic_sched.step()
            self.lagrange_mult_sched.step()
        return result

    # for type hints
    def __call__(self, data: BatchData, critic_batch_info: CriticBatchInfo, *,
                 optimize: bool = True) -> LossResult:
        return torch.nn.Module.__call__(self, data, critic_batch_info, optimize=optimize)
