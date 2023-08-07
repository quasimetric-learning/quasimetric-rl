from typing import *
from typing import Any, Mapping

import attrs

import torch

from . import actor, quasimetric_critic

from ..data import EnvSpec, BatchData
from .utils import LossResult, Module, InfoT


class QRLAgent(Module):
    actor: Optional[actor.Actor]
    critics: Collection[quasimetric_critic.QuasimetricCritic]

    def __init__(self, actor: Optional['actor.Actor'], critics: Collection[quasimetric_critic.QuasimetricCritic]):
        super().__init__()
        self.add_module('actor', actor)
        self.critics = torch.nn.ModuleList(critics)


class QRLLosses(Module):
    actor_loss: Optional[actor.ActorLosses]
    critic_losses: Collection[quasimetric_critic.QuasimetricCriticLosses]

    def __init__(self, actor_loss: Optional['actor.ActorLosses'],
                 critic_losses: Collection[quasimetric_critic.QuasimetricCriticLosses]):
        super().__init__()
        self.add_module('actor_loss', actor_loss)
        self.critic_losses = torch.nn.ModuleList(critic_losses)

    def forward(self, agent: QRLAgent, data: BatchData, *, optimize: bool = True) -> LossResult:
        # compute CriticBatchInfo
        critic_batch_infos = []
        loss_results: Dict[str, LossResult] = {}

        for idx, (critic, critic_loss) in enumerate(zip(agent.critics, self.critic_losses)):
            zx, zy = critic.encoder(torch.stack([data.observations, data.next_observations], dim=0)).unbind(0)
            critic_batch_info = quasimetric_critic.CriticBatchInfo(
                critic=critic,
                zx=zx,
                zy=zy,
            )
            critic_batch_infos.append(critic_batch_info)
            loss_results[f"critic_{idx:02d}"] = critic_loss(data, critic_batch_info, optimize=optimize)

        if self.actor_loss is not None:
            loss_results['actor'] = self.actor_loss(agent.actor, critic_batch_infos, data, optimize=optimize)

        return LossResult.combine(loss_results)

    # for type hints
    def __call__(self, agent: QRLAgent, data: BatchData, *, optimize: bool = True) -> LossResult:
        return super().__call__(agent, data, optimize=optimize)

    def state_dict(self):
        optim_scheds = {}
        if self.actor_loss is not None:
            optim_scheds['actor'] = dict(
                actor_optim=self.actor_loss.actor_optim.state_dict(),
                actor_sched=self.actor_loss.actor_sched.state_dict(),
                entropy_weight_optim=self.actor_loss.entropy_weight_optim.state_dict(),
                entropy_weight_sched=self.actor_loss.entropy_weight_sched.state_dict(),
            )
        for idx, critic_loss in enumerate(self.critic_losses):
            optim_scheds[f"critic_{idx:02d}"] = dict(
                critic_optim=critic_loss.critic_optim.state_dict(),
                critic_sched=critic_loss.critic_sched.state_dict(),
                lagrange_mult_optim=critic_loss.lagrange_mult_optim.state_dict(),
                lagrange_mult_sched=critic_loss.lagrange_mult_sched.state_dict(),
            )
        return dict(
            module=super().state_dict(),
            optim_scheds=optim_scheds,
        )

    def load_state_dict(self, state_dict: Mapping[str, Any]):
        super().load_state_dict(state_dict['module'])
        optim_scheds = state_dict['optim_scheds']
        if self.actor_loss is not None:
            self.actor_loss.actor_optim.load_state_dict(optim_scheds['actor']['actor_optim'])
            self.actor_loss.actor_sched.load_state_dict(optim_scheds['actor']['actor_sched'])
            self.actor_loss.entropy_weight_optim.load_state_dict(optim_scheds['actor']['entropy_weight_optim'])
            self.actor_loss.entropy_weight_sched.load_state_dict(optim_scheds['actor']['entropy_weight_sched']),
        for idx, critic_loss in enumerate(self.critic_losses):
            critic_loss.critic_optim.load_state_dict(optim_scheds[f"critic_{idx:02d}"]['critic_optim'])
            critic_loss.critic_sched.load_state_dict(optim_scheds[f"critic_{idx:02d}"]['critic_sched'])
            critic_loss.lagrange_mult_optim.load_state_dict(optim_scheds[f"critic_{idx:02d}"]['lagrange_mult_optim'])
            critic_loss.lagrange_mult_sched.load_state_dict(optim_scheds[f"critic_{idx:02d}"]['lagrange_mult_sched'])


@attrs.define(kw_only=True)
class QRLConf:
    actor: Optional['actor.ActorConf'] = actor.ActorConf()
    quasimetric_critic: 'quasimetric_critic.QuasimetricCriticConf' = quasimetric_critic.QuasimetricCriticConf()
    num_critics: int = attrs.field(default=2, validator=attrs.validators.gt(0))

    def make(self, *, env_spec: EnvSpec, total_optim_steps: int) -> Tuple[QRLAgent, QRLLosses]:
        if self.actor is None:
            actor = actor_losses = None
        else:
            actor, actor_losses = self.actor.make(env_spec=env_spec, total_optim_steps=total_optim_steps)
        critics, critic_losses = zip(*[
            self.quasimetric_critic.make(env_spec=env_spec, total_optim_steps=total_optim_steps) for _ in range(self.num_critics)
        ])
        return QRLAgent(actor=actor, critics=critics), QRLLosses(actor_loss=actor_losses, critic_losses=critic_losses)

__all__ = ['QRLAgent', 'QRLLosses', 'QRLConf', 'InfoT']
