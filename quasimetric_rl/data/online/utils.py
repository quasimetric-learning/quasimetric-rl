from __future__ import annotations
from typing import *

import numpy as np
import torch
import torch.utils.data

from .. import EnvSpec
from ..base import (
    EpisodeData, MultiEpisodeData,
)


def get_empty_episode(env_spec: EnvSpec, episode_length: int) -> EpisodeData:
    r'''
    episode_lengths: torch.Tensor                                          # [1]
    all_observations: torch.Tensor                                        # [L + 1, *observation_shape]
    actions: torch.Tensor                                                 # [L, *action_shape]
    rewards: torch.Tensor                                                 # [L]
    terminals: torch.Tensor                                               # [L]
    timeouts: torch.Tensor                                                # [L]
    observation_infos: Mapping[str, torch.Tensor] = attrs.Factory(dict)   # [L + 1]
    transition_infos: Mapping[str, torch.Tensor] = attrs.Factory(dict)    # [L]
    can_generate_goal_transition: torch.Tensor                            # [L + 1]

    Only needs to fill in
        all_observations
        actions
        rewards
        can_generate_goal_transition
        is_success
        desired_goals
    afterwards.
    '''
    all_observations = torch.empty(episode_length + 1, *env_spec.observation_shape, dtype=env_spec.observation_dtype)
    timeouts = torch.zeros(episode_length, dtype=torch.bool)
    timeouts[-1] = True
    return EpisodeData(
        episode_lengths=torch.tensor(episode_length).view(-1),
        all_observations=all_observations,
        actions=torch.empty(episode_length, *env_spec.action_shape, dtype=env_spec.action_dtype),
        rewards=torch.empty(episode_length),
        terminals=torch.zeros(episode_length, dtype=torch.bool),
        timeouts=timeouts,
        observation_infos=dict(
            achieved_goals=torch.empty_like(all_observations),
            desired_goals=torch.empty_like(all_observations),
        ),
        transition_infos=dict(
            is_success=torch.empty(episode_length, dtype=torch.bool),
        ),
    )


def get_empty_episodes(env_spec: EnvSpec, episode_length: int, num_episodes: int) -> MultiEpisodeData:
    all_observations = torch.empty(episode_length * num_episodes + num_episodes, *env_spec.observation_shape, dtype=env_spec.observation_dtype)
    timeouts = torch.zeros(num_episodes, episode_length, dtype=torch.bool)
    timeouts[:, -1] = True
    return MultiEpisodeData(
        episode_lengths=torch.full([num_episodes], episode_length, dtype=torch.int64),
        all_observations=all_observations,
        actions=torch.empty(episode_length * num_episodes, *env_spec.action_shape, dtype=env_spec.action_dtype),
        rewards=torch.empty(episode_length * num_episodes),
        terminals=torch.zeros(episode_length * num_episodes, dtype=torch.bool),
        timeouts=timeouts.flatten(),
        observation_infos=dict(
            achieved_goals=torch.empty_like(all_observations),
            desired_goals=torch.empty_like(all_observations),
        ),
        transition_infos=dict(
            is_success=torch.empty(episode_length * num_episodes, dtype=torch.bool),
        ),
    )
