from typing import *

import attrs

from .models import QuasimetricCritic
from .losses import QuasimetricCriticLosses, CriticBatchInfo

from ...data import EnvSpec


@attrs.define(kw_only=True)
class QuasimetricCriticConf:
    model: QuasimetricCritic.Conf = QuasimetricCritic.Conf()
    losses: QuasimetricCriticLosses.Conf = QuasimetricCriticLosses.Conf()

    def make(self, *, env_spec: EnvSpec, total_optim_steps: int) -> Tuple[QuasimetricCritic, QuasimetricCriticLosses]:
        critic = self.model.make(env_spec=env_spec)
        return critic, self.losses.make(critic, total_optim_steps)


__all__ = ['QuasimetricCritic', 'QuasimetricCriticLosses', 'CriticBatchInfo', 'QuasimetricCriticConf']
