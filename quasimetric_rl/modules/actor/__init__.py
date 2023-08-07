from typing import *

import attrs

from .model import Actor
from .losses import ActorLosses

from ...data import EnvSpec


@attrs.define(kw_only=True)
class ActorConf:
    # config / argparse uses this to specify behavior

    model: Actor.Conf = Actor.Conf()
    losses: ActorLosses.Conf = ActorLosses.Conf()

    def make(self, *, env_spec: EnvSpec, total_optim_steps: int) -> Tuple[Actor, ActorLosses]:
        actor = self.model.make(env_spec=env_spec)
        return actor, self.losses.make(actor, total_optim_steps=total_optim_steps, env_spec=env_spec)


__all__ = ['Actor', 'ActorLosses', 'ActorConf']
