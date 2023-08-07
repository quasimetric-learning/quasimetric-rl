from typing import *

import attrs

import torch
import torch.nn as nn

from ....data import BatchData, EnvSpec

from ...utils import LatentTensor, LossResult, grad_mul
from ..model import Actor
from ...quasimetric_critic import QuasimetricCritic, CriticBatchInfo

from . import ActorLossBase



@attrs.define(kw_only=True)
class ActorObsGoalCriticInfo:
    r"""
    Similar to CriticBatchInfo, but does not store the latents for the data batch.

    Instead, for the batch of observation and goal pairs which the actor is activated with,
    this stores the latents for them.
    """
    critic: QuasimetricCritic
    zo: LatentTensor
    zg: LatentTensor


class MinDistLoss(ActorLossBase):
    @attrs.define(kw_only=True)
    class Conf:
        # config / argparse uses this to specify behavior

        adaptive_entropy_regularizer: bool = True

        # If set, in addition to use random goals, also use future state in the same trajectory as goals.
        # We enable this for online settings, following Contrastive RL.
        add_goal_as_future_state: bool = True

        def make(self, env_spec: EnvSpec) -> 'MinDistLoss':
            return MinDistLoss(
                env_spec=env_spec,
                adaptive_entropy_regularizer=self.adaptive_entropy_regularizer,
                add_goal_as_future_state=self.add_goal_as_future_state,
            )

    add_goal_as_future_state: bool
    raw_entropy_weight: Optional[nn.Parameter]  # set if using adaptive entropy regularization
    target_entropy: Optional[float] = None  # set if using adaptive entropy regularization

    def __init__(self, *, env_spec: EnvSpec,
                 adaptive_entropy_regularizer: bool,
                 add_goal_as_future_state: bool):
        super().__init__()
        self.add_goal_as_future_state = add_goal_as_future_state
        if adaptive_entropy_regularizer:
            self.raw_entropy_weight = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            self.target_entropy = env_spec.get_action_entropy_reg_target()
        else:
            self.register_parameter('raw_entropy_weight', None)
            self.target_entropy = None

    def gather_obs_goal_pairs(self, critic_batch_infos: Collection[CriticBatchInfo],
                                data: BatchData) -> Tuple[torch.Tensor, torch.Tensor, Collection[ActorObsGoalCriticInfo]]:
        r"""
        Returns (
            obs,
            goal,
            [ (latent_obs, latent_goal) for each critic ],
        )
        """

        obs = data.observations
        goal = torch.roll(data.next_observations, 1, dims=0)  # randomize :)
        if self.add_goal_as_future_state:
            # add future_observations
            goal = torch.stack([goal, data.future_observations], 0)
            obs = obs.expand_as(goal)

        actor_obs_goal_critic_infos: List[ActorObsGoalCriticInfo] = []

        for critic_batch_info in critic_batch_infos:
            zo = critic_batch_info.zx
            zg = torch.roll(critic_batch_info.zy, 1, dims=0)  # randomize in the same way:)

            if self.add_goal_as_future_state:
                # add future_observations
                zg = torch.stack([
                    zg,
                    critic_batch_info.critic.encoder(data.future_observations),
                ], 0)
                zo = zo.expand_as(zg)

            actor_obs_goal_critic_infos.append(ActorObsGoalCriticInfo(
                critic=critic_batch_info.critic,
                zo=zo,
                zg=zg,
            ))

        return obs, goal, actor_obs_goal_critic_infos

    def forward(self, actor: Actor, critic_batch_infos: Collection[CriticBatchInfo], data: BatchData) -> LossResult:
        with torch.no_grad():
            obs, goal, actor_obs_goal_critic_infos = self.gather_obs_goal_pairs(critic_batch_infos, data)

        actor_distn = actor(obs, goal)
        action = actor_distn.rsample()

        info: Dict[str, torch.Tensor] = {}

        dists: List[torch.Tensor] = []

        for idx, actor_obs_goal_critic_info in enumerate(actor_obs_goal_critic_infos):
            critic = actor_obs_goal_critic_info.critic
            with critic.requiring_grad(False):
                zp = critic.latent_dynamics(actor_obs_goal_critic_info.zo.detach(), action)
                dist = critic.quasimetric_model(zp, actor_obs_goal_critic_info.zg.detach())
            info[f'dist_{idx:02d}'] = dist.mean()
            dists.append(dist)

        max_dist = info['dist_max'] = torch.stack(dists, -1).max(-1).values.mean()
        loss = max_dist  # pick the most pessimistic

        if self.target_entropy is not None:
            # add entropy regularization

            info['target_entropy'] = self.target_entropy
            entropy = info['entropy'] = actor_distn.entropy().mean()

            alpha = info['entropy_alpha'] = grad_mul(self.raw_entropy_weight.exp(), -1)  # minimax :)
            loss += alpha * (self.target_entropy - entropy)

        return LossResult(loss=loss, info=info)

    def extra_repr(self) -> str:
        return f"add_goal_as_future_state={self.add_goal_as_future_state}, target_entropy={self.target_entropy}"
