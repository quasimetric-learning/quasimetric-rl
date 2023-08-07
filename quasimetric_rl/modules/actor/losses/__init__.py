from typing import *

import abc
import attrs

import torch

from ...utils import LossBase, LossResult
from ....data import BatchData, EnvSpec
from ..model import Actor
from ...quasimetric_critic.losses import CriticBatchInfo
from ...optim import OptimWrapper, AdamWSpec


class ActorLossBase(LossBase):
    @abc.abstractmethod
    def forward(self, actor: Actor, critic_batch_infos: Collection[CriticBatchInfo], data: BatchData) -> LossResult:
        pass

    # for type hints
    def __call__(self, actor: Actor, critic_batch_infos: Collection[CriticBatchInfo], data: BatchData) -> LossResult:
        return super().__call__(actor, critic_batch_infos, data)


from .min_dist import MinDistLoss
from .behavior_cloning import BCLoss


class ActorLosses(ActorLossBase):
    @attrs.define(kw_only=True)
    class Conf:
        # config / argparse uses this to specify behavior

        min_dist: MinDistLoss.Conf = MinDistLoss.Conf()
        behavior_cloning: BCLoss.Conf = BCLoss.Conf()

        actor_optim: AdamWSpec.Conf = AdamWSpec.Conf(lr=3e-5)
        entropy_weight_optim: AdamWSpec.Conf = AdamWSpec.Conf(lr=3e-4)

        def make(self, actor: Actor, total_optim_steps: int, env_spec: EnvSpec) -> 'ActorLosses':
            return ActorLosses(
                actor,
                total_optim_steps=total_optim_steps,
                min_dist=self.min_dist.make(env_spec=env_spec),
                behavior_cloning=self.behavior_cloning.make(),
                actor_optim_spec=self.actor_optim.make(),
                entropy_weight_optim_spec=self.entropy_weight_optim.make(),
            )

    min_dist: MinDistLoss
    behavior_cloning: BCLoss

    actor_optim: OptimWrapper
    actor_sched: torch.optim.lr_scheduler._LRScheduler
    entropy_weight_optim: OptimWrapper
    entropy_weight_sched: torch.optim.lr_scheduler._LRScheduler

    def __init__(self, actor: Actor, *, total_optim_steps: int,
                 min_dist: MinDistLoss, behavior_cloning: BCLoss,
                 actor_optim_spec: AdamWSpec, entropy_weight_optim_spec: AdamWSpec):
        super().__init__()
        self.min_dist = min_dist
        self.behavior_cloning = behavior_cloning

        self.actor_optim, self.actor_sched = actor_optim_spec.create_optim_scheduler(
            actor.parameters(), total_optim_steps)
        self.entropy_weight_optim, self.entropy_weight_sched = entropy_weight_optim_spec.create_optim_scheduler(
            min_dist.parameters(), total_optim_steps)
        assert len(list(min_dist.parameters())) <= 1

    def forward(self, actor: Actor, critic_batch_infos: Collection[CriticBatchInfo], data: BatchData, *,
                optimize: bool = True) -> LossResult:
        with self.actor_optim.update_context(optimize=optimize), \
                self.entropy_weight_optim.update_context(optimize=optimize):

            result = LossResult.combine(dict(
                min_dist=self.min_dist(actor, critic_batch_infos, data),
                behavior_cloning=self.behavior_cloning(actor, critic_batch_infos, data),
            ))
            result.loss.backward()

        if optimize:
            self.actor_sched.step()
            self.entropy_weight_sched.step()
        return result

    # for type hints
    def __call__(self, actor: Actor, critic_batch_infos: Collection[CriticBatchInfo], data: BatchData, *,
                 optimize: bool = True) -> LossResult:
        return torch.nn.Module.__call__(self, actor, critic_batch_infos, data, optimize=optimize)
