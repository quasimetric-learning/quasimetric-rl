from __future__ import annotations
from typing import *

import attrs
import logging

import numpy as np
import torch
import torch.utils.data
import gym
import gym.spaces

from quasimetric_rl.data.env_spec import EnvSpec

from .. import EnvSpec
from ..base import (
    EpisodeData, MultiEpisodeData, Dataset, BatchData,
    register_offline_env,
)
from .utils import get_empty_episode, get_empty_episodes


#-----------------------------------------------------------------------------#
#------------------------------ replay buffer --------------------------------#
#-----------------------------------------------------------------------------#

# `ReplayBuffer` is an extended `Dataset``, that
#
#   1. supports sampling a batch of valid transitions
#
#   2. *does not* support dataloader access (which may be multiprocessing)
#
#   3. supports adding new rollouts (episodes)
#
#   4. to avoid constantly expanding the tensors after each new episode, we
#      expand `increment_num_episodes` episodes in the `Dataset` at each time,
#      and keep track of which episodes in `Dataset` contains valid data.
#      (similar to how vectors grows in chunks in c++ to have amortized constant
#      time complexity, although we grow by a constant number of episodes rather
#      than exponentially)
#
#   5. as a result of 4, each episode must contain a fixed number of transitions,
#      which we access by requiring all online env to
#        i.  use an observation dict with keys ['observation', 'achived_goal', 'desired_goal'],
#            all of the same dtype and shape
#        ii  exposes 'is_success' as a bool in `info`
#        ii. be wrapped with `FixedLengthEnvWrapper`.
#
#      (i) and (ii) are checked in collecting rollouts and creating `ReplayBuffer`.
#      (iii) is done by the `register_online_env` function below.
#


class FixedLengthEnvWrapper(gym.Wrapper):
    episode_length: int
    current_episode_num_transitions: int

    def __init__(self, env, episode_length: int):
        super().__init__(env)
        assert episode_length > 0
        self.episode_length = episode_length
        self.current_episode_num_transitions = 0

    def reset(self, *args, **kwargs):
        self.current_episode_num_transitions = 0
        return super().reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        self.current_episode_num_transitions += 1
        assert self.current_episode_num_transitions <= self.episode_length
        return super().step(*args, **kwargs)


def load_episode_error():
    raise NotImplementedError()


def register_online_env(kind: str, spec: str, *,
                 load_episodes_fn=load_episode_error,
                 create_env_fn, episode_length: int):
    r"""
    Similar to `register_offline_env`, but
      1. has a default `load_episodes_fn` that errors out.
      2. requires each episode to have a fixed length specified by `episode_length`.
    """
    register_offline_env(
        kind, spec,
        load_episodes_fn=load_episodes_fn,
        create_env_fn=lambda: FixedLengthEnvWrapper(create_env_fn(), episode_length))


class ReplayBuffer(Dataset):
    @attrs.define(kw_only=True)
    class Conf(Dataset.Conf):
        r"""
        Adds a few more configs to `Dataset.Conf`, but they generally should not
        be changed. See below for details.
        """

        # Whether to load the offline data by calling registered `load_episode_fn`
        # In general, for online settings, this should not be used and `load_episode_fn`
        # will through an error.  But useful for debugging and analysis, where you
        # can fix the training data across methods.
        load_offline_data: bool = False

        # Below are options for growing the tensors. Only tune they if you think
        # that speed / memory is affected by the current settings. They *DO NOT*
        # affect behavior. See above note.
        init_num_transitions: int = attrs.field(default=int(1e6), validator=attrs.validators.gt(0))
        increment_num_transitions: int = attrs.field(default=int(0.5e6), validator=attrs.validators.gt(0))

        def make(self) -> 'ReplayBuffer':
            return ReplayBuffer(
                self.kind, self.name,
                future_observation_discount=self.future_observation_discount,
                load_offline_data=self.load_offline_data,
                init_num_transitions=self.init_num_transitions,
                increment_num_transitions=self.increment_num_transitions,
            )

    load_offline_data: bool
    init_num_transitions: int
    increment_num_transitions: int

    env: FixedLengthEnvWrapper
    num_episodes_realized: int  # track valid episodes
    num_successful_episodes: int  # stats
    num_successful_transitions: int  # stats

    @property
    def episode_length(self) -> int:
        return self.env.episode_length

    @property
    def num_transitions_realized(self) -> int:
        return self.num_episodes_realized * self.episode_length

    @property
    def episodes_capacity(self) -> int:
        return self.raw_data.num_episodes

    def create_env(self) -> FixedLengthEnvWrapper:  # type hint
        env = super().create_env()
        assert isinstance(env, FixedLengthEnvWrapper), "not online env"
        return env

    def load_episodes(self) -> Iterator[EpisodeData]:
        r'''
        Used to initialize `ReplayBuffer`
        '''
        init_num_episodes = int(np.ceil(self.init_num_transitions / self.episode_length))
        for _ in range(init_num_episodes):
            yield get_empty_episode(self.env_spec, self.episode_length)

    def __init__(self, kind: str, name: str, *, future_observation_discount: float,
                 load_offline_data: bool, init_num_transitions: int, increment_num_transitions: int,
                 dummy: bool = False,  # when you don't want to load data, e.g., in analysis
                 ):
        self.kind = kind
        self.name = name
        self.load_offline_data = load_offline_data
        self.init_num_transitions = init_num_transitions
        self.increment_num_transitions = increment_num_transitions
        self.env = self.create_env()
        super().__init__(
            kind, name, future_observation_discount=future_observation_discount,
            dummy=dummy)
        self.num_episodes_realized = 0
        if load_offline_data and not dummy:  # load data if required.
            for episode in super().load_episodes():
                self.add_rollout(episode)
                self.num_episodes_realized += 1
        self.num_successful_episodes = self.raw_data.transition_infos['is_success'].unflatten(
            0, [self.episodes_capacity, self.episode_length],
        )[:self.num_episodes_realized].any(-1).sum(dtype=torch.int64).item()
        self.num_successful_transitions = self.raw_data.transition_infos['is_success'].unflatten(
            0, [self.episodes_capacity, self.episode_length],
        )[:self.num_episodes_realized].sum(dtype=torch.int64).item()

    def _expand(self):
        original_capacity: int = self.episodes_capacity

        # Update
        #   # Data
        #   raw_data: MultiEpisodeData  # only episodes in split
        #   # Auxiliary structures that helps fetching transitions of specific kinds
        self.raw_data = MultiEpisodeData.cat(
            [self.raw_data, get_empty_episodes(self.env_spec, self.episode_length, self.increment_num_episodes)],
            dim=0,
        )
        new_capacity = self.episodes_capacity

        #   # Stats
        #   indices_to_episode_indices: torch.Tensor  # episode indices refers to indices in this split
        #   indices_to_episode_timesteps: torch.Tensor
        self.indices_to_episode_indices = torch.cat([
            self.indices_to_episode_indices,
            torch.repeat_interleave(torch.arange(original_capacity, new_capacity), self.episode_length),
        ], dim=0)
        self.indices_to_episode_timesteps = torch.cat([
            self.indices_to_episode_timesteps,
            torch.arange(self.episode_length).repeat(new_capacity - original_capacity),
        ], dim=0)

        logging.info(f'ReplayBuffer: Expanded from capacity={original_capacity} to {new_capacity} episodes')

    def collect_rollout(self, actor: Callable[[torch.Tensor, torch.Tensor, gym.Space], np.ndarray], *,
                        env: Optional[FixedLengthEnvWrapper] = None) -> EpisodeData:
        if env is None:
            env = self.env

        epi = get_empty_episode(self.env_spec, self.episode_length)

        # check observation space
        obs_dict_keys = {'observation', 'achieved_goal', 'desired_goal'}
        WRONG_OBS_ERR_MESSAGE = (
            f"{self.__class__.__name__} collect_rollout only supports Dict "
            f"observation space with keys {obs_dict_keys}, but got {env.observation_space}"
        )
        assert isinstance(env.observation_space, gym.spaces.Dict), WRONG_OBS_ERR_MESSAGE
        assert set(env.observation_space.spaces.keys()) == {'observation', 'achieved_goal', 'desired_goal'}, WRONG_OBS_ERR_MESSAGE

        observation_dict = env.reset()
        observation: torch.Tensor = torch.as_tensor(observation_dict['observation'])

        goal: torch.Tensor = torch.as_tensor(observation_dict['desired_goal'])
        agoal: torch.Tensor = torch.as_tensor(observation_dict['achieved_goal'])
        epi.all_observations[0] = observation
        epi.observation_infos['desired_goals'][0] = goal
        epi.observation_infos['achieved_goals'][0] = agoal

        t = 0
        timeout = False
        while not timeout:
            action = actor(
                observation,
                goal,
                self.env_spec.action_space,
            )
            observation_dict, reward, terminal, info = env.step(np.asarray(action))

            observation = torch.tensor(observation_dict['observation'])  # copy just in case

            goal: torch.Tensor = torch.as_tensor(observation_dict['desired_goal'])
            agoal: torch.Tensor = torch.as_tensor(observation_dict['achieved_goal'])

            is_success: bool = info['is_success']

            epi.all_observations[t + 1] = observation
            epi.actions[t] = torch.as_tensor(action)
            epi.rewards[t] = reward
            epi.transition_infos['is_success'][t] = is_success
            epi.observation_infos['desired_goals'][t + 1] = goal
            epi.observation_infos['achieved_goals'][t + 1] = agoal

            t += 1
            timeout = info.get('TimeLimit.truncated', False)
            assert (timeout or terminal) == (t == self.episode_length)
        return epi

    def add_rollout(self, episode: EpisodeData):
        if self.num_episodes_realized == self.episodes_capacity:
            self._expand()

        self.raw_data.all_observations.unflatten(
            0, [self.episodes_capacity, self.episode_length + 1],
        )[self.num_episodes_realized] = episode.all_observations

        self.raw_data.actions.unflatten(
            0, [self.episodes_capacity, self.episode_length],
        )[self.num_episodes_realized] = episode.actions

        self.raw_data.rewards.unflatten(
            0, [self.episodes_capacity, self.episode_length],
        )[self.num_episodes_realized] = episode.rewards

        self.raw_data.observation_infos['achieved_goals'].unflatten(
            0, [self.episodes_capacity, self.episode_length + 1],
        )[self.num_episodes_realized] = episode.observation_infos['achieved_goals']

        self.raw_data.observation_infos['desired_goals'].unflatten(
            0, [self.episodes_capacity, self.episode_length + 1],
        )[self.num_episodes_realized] = episode.observation_infos['desired_goals']

        self.raw_data.transition_infos['is_success'].unflatten(
            0, [self.episodes_capacity, self.episode_length],
        )[self.num_episodes_realized] = episode.transition_infos['is_success']

        num_successful_transitions = episode.transition_infos['is_success'].sum(dtype=torch.int64).item()
        self.num_successful_transitions += num_successful_transitions
        self.num_successful_episodes += int(num_successful_transitions > 0)

        self.num_episodes_realized += 1

    def sample(self, batch_size: int) -> BatchData:
        indices = torch.as_tensor(
            np.random.choice(self.num_transitions_realized, size=[batch_size])
        )
        return self[indices]

    def get_dataloader(self, *args, **kwargs):
        raise RuntimeError('Online data cannot be loaded as a dataloader')

    def __repr__(self):
        lines = super().__repr__().split('\n')
        assert lines[-1] == ')'
        lines = lines[:-1] + [
            rf"""
    episode_length={self.episode_length!r},
    load_offline_data={self.load_offline_data!r},
    num_episodes_realized={self.num_episodes_realized!r},
    num_successful_episodes={self.num_successful_episodes!r},
    num_successful_transitions={self.num_successful_transitions!r},
""".strip('\n'),
        ] + lines[-1:]
        return '\n'.join(lines)


from . import gcrl  # register
