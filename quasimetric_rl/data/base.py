from __future__ import annotations
from typing import *

import attrs

import numpy as np
import torch
import torch.utils.data
import gym

from omegaconf import MISSING

from .utils import TensorCollectionAttrsMixin
from .env_spec import EnvSpec



#-----------------------------------------------------------------------------#
#-------------------------------- Batch data ---------------------------------#
#-----------------------------------------------------------------------------#

# What should be in a batch


@attrs.define(kw_only=True)
class BatchData(TensorCollectionAttrsMixin):  # TensorCollectionAttrsMixin has some util methods
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    rewards: torch.Tensor
    terminals: torch.Tensor
    timeouts: torch.Tensor

    future_observations: torch.Tensor  # sampled!

    @property
    def device(self) -> torch.device:
        return self.observations.device

    @property
    def batch_shape(self) -> torch.Size:
        return self.terminals.shape

    @property
    def num_transitions(self) -> int:
        return self.terminals.numel()



#-----------------------------------------------------------------------------#
#------------------------------- Episode data --------------------------------#
#-----------------------------------------------------------------------------#


@attrs.define(kw_only=True)
class MultiEpisodeData(TensorCollectionAttrsMixin):
    r"""
    The DATASET of MULTIPLE episodes
    """


    # For each episode, L: number of (s, a, s', r, d, to) pairs, so number of transitions (not observations)
    episode_lengths: torch.Tensor
    # cat all states from all episodes, where the last s' is added. I.e., each episode has L+1 states
    all_observations: torch.Tensor
    # cat all actions from all episodes. Each episode has L actions.
    actions: torch.Tensor
    # cat all rewards from all episodes. Each episode has L rewards.
    rewards: torch.Tensor
    # cat all terminals from all episodes. Each episode has L terminals.
    terminals: torch.Tensor
    # cat all timeouts from all episodes. Each episode has L timeouts.
    timeouts: torch.Tensor
    # cat all observation infos from all episodes. Each episode has L + 1 elements.
    observation_infos: Mapping[str, torch.Tensor] = attrs.Factory(dict)
    # cat all transition infos from all episodes. Each episode has L elements.
    transition_infos: Mapping[str, torch.Tensor] = attrs.Factory(dict)

    @property
    def num_episodes(self) -> int:
        return self.episode_lengths.shape[0]

    @property
    def num_transitions(self) -> int:
        return self.rewards.shape[0]

    def __attrs_post_init__(self):
        assert self.episode_lengths.ndim == 1
        N = self.num_transitions
        assert N > 0
        assert self.all_observations.ndim >= 1 and self.all_observations.shape[0] == (N + self.num_episodes), self.all_observations.shape
        assert self.actions.ndim >= 1 and self.actions.shape[0] == N
        assert self.rewards.ndim == 1 and self.rewards.shape[0] == N
        assert self.terminals.ndim == 1 and self.terminals.shape[0] == N
        assert self.timeouts.ndim == 1 and self.timeouts.shape[0] == N
        for k, v in self.observation_infos.items():
            assert v.shape[0] == N + self.num_episodes, k
        for k, v in self.transition_infos.items():
            assert v.shape[0] == N, k



@attrs.define(kw_only=True)
class EpisodeData(MultiEpisodeData):
    r"""
    A SINGLE episode
    """

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        assert self.num_episodes == 1

    @classmethod
    def from_simple_trajectory(cls,
                               observations: Union[np.ndarray, torch.Tensor],
                               actions: Union[np.ndarray, torch.Tensor],
                               next_observations: Union[np.ndarray, torch.Tensor],
                               rewards: Union[np.ndarray, torch.Tensor],
                               terminals: Union[np.ndarray, torch.Tensor],
                               timeouts: Union[np.ndarray, torch.Tensor]):
        observations = torch.tensor(observations)
        next_observations=torch.tensor(next_observations)
        all_observations = torch.cat([observations, next_observations[-1:]], dim=0)
        return cls(
            episode_length=torch.tensor([observations.shape[0]]),
            all_observations=all_observations,
            actions=torch.tensor(actions),
            rewards=torch.tensor(rewards),
            terminals=torch.tensor(terminals),
            timeouts=torch.tensor(timeouts),
        )


#-----------------------------------------------------------------------------#
#--------------------------------- dataset -----------------------------------#
#-----------------------------------------------------------------------------#


# Each env is specified with two strings:
#   + kind  # d4rl, gcrl, etc.
#   + spec  # maze2d-umaze-v1, FetchPushImage, etc.


LOAD_EPISODES_REGISTRY: Mapping[Tuple[str, str], Callable[[], Iterator[EpisodeData]]] = {}
CREATE_ENV_REGISTRY: Mapping[Tuple[str, str], Callable[[], gym.Env]] = {}


def register_offline_env(kind: str, spec: str, *, load_episodes_fn, create_env_fn):
    r"""
    Each specific env (e.g., an offline env from d4rl) just needs to register

        1. how to load the episodes
        (this is optional in online settings. see ReplayBuffer)

        load_episodes_fn() -> Iterator[EpisodeData]

        2. how to create an env

        create_env_fn() -> gym.Env

     See d4rl/maze2d.py for example
    """
    assert (kind, spec) not in LOAD_EPISODES_REGISTRY
    LOAD_EPISODES_REGISTRY[(kind, spec)] = load_episodes_fn
    CREATE_ENV_REGISTRY[(kind, spec)] = create_env_fn


class Dataset:
    @attrs.define(kw_only=True)
    class Conf:
        # config / argparse uses this to specify behavior

        kind: str = MISSING  # d4rl, gcrl, etc.
        name: str = MISSING  # maze2d-umaze-v1, etc.

        # Defines how to fetch the future observation. smaller -> more recent
        future_observation_discount: float = attrs.field(default=0.99, validator=attrs.validators.and_(
            attrs.validators.ge(0.0),
            attrs.validators.le(1.0),
        ))

        def make(self, *, dummy: bool = False) -> 'Dataset':
            return Dataset(self.kind, self.name,
                           future_observation_discount=self.future_observation_discount,
                           dummy=dummy)

    kind: str
    name: str
    future_observation_discount: float

    # Computed Attributes::

    # Data
    raw_data: MultiEpisodeData  # will contain all episodes

    # Env info
    env_spec: EnvSpec

    # Defines how to fetch the future observation. smaller -> more recent
    future_observation_discount: float

    # Auxiliary structures that helps fetching transitions of specific kinds
    # -----
    obs_indices_to_obs_index_in_episode: torch.Tensor
    indices_to_episode_indices: torch.Tensor  # episode indices refers to indices in this split
    indices_to_episode_timesteps: torch.Tensor
    max_episode_length: int
    # -----

    def create_env(self) -> gym.Env:
        return CREATE_ENV_REGISTRY[self.kind, self.name]()

    def load_episodes(self) -> Iterator[EpisodeData]:
        return LOAD_EPISODES_REGISTRY[self.kind, self.name]()

    def __init__(self, kind: str, name: str, *,
                 future_observation_discount: float,
                 dummy: bool = False,  # when you don't want to load data, e.g., in analysis
                 ) -> None:
        self.kind = kind
        self.name = name
        self.future_observation_discount = future_observation_discount

        self.env_spec = EnvSpec.from_env(self.create_env())

        assert 0 <= future_observation_discount
        self.future_observation_discount = future_observation_discount

        if not dummy:
            episodes = tuple(self.load_episodes())
        else:
            from .online.utils import get_empty_episode
            episodes = (get_empty_episode(self.env_spec, episode_length=1),)

        obs_indices_to_obs_index_in_episode = []
        indices_to_episode_indices = []
        indices_to_episode_timesteps = []
        for eidx, episode in enumerate(episodes):
            l = episode.num_transitions
            obs_indices_to_obs_index_in_episode.append(torch.arange(l + 1, dtype=torch.int64))
            indices_to_episode_indices.append(torch.full([l], eidx, dtype=torch.int64))
            indices_to_episode_timesteps.append(torch.arange(l, dtype=torch.int64))

        assert len(episodes) > 0, "must have at least one episode"
        self.raw_data = MultiEpisodeData.cat(episodes)

        self.obs_indices_to_obs_index_in_episode = torch.cat(obs_indices_to_obs_index_in_episode, dim=0)
        self.indices_to_episode_indices = torch.cat(indices_to_episode_indices, dim=0)
        self.indices_to_episode_timesteps = torch.cat(indices_to_episode_timesteps, dim=0)
        self.max_episode_length = self.raw_data.episode_lengths.max().item()

    def get_observations(self, obs_indices: torch.Tensor):
        return self.raw_data.all_observations[obs_indices]

    def __getitem__(self, indices: torch.Tensor) -> BatchData:
        indices = torch.as_tensor(indices)
        eindices = self.indices_to_episode_indices[indices]
        obs_indices = indices + eindices  # index for `observation`: skip the s_last from previous episodes
        obs = self.get_observations(obs_indices)
        nobs = self.get_observations(obs_indices + 1)

        terminals = self.raw_data.terminals[indices]

        tindices = self.indices_to_episode_timesteps[indices]
        epilengths = self.raw_data.episode_lengths[eindices]  # max idx is this
        deltas = torch.arange(self.max_episode_length)
        pdeltas = torch.where(
            # test tidx + 1 + delta <= max_idx = epi_length
            (tindices[:, None] + deltas) < epilengths[:, None],
            self.future_observation_discount ** deltas,
            0,
        )
        deltas = torch.distributions.Categorical(
            probs=pdeltas,
        ).sample()
        future_observations = self.get_observations(obs_indices + 1 + deltas)

        return BatchData(
            observations=obs,
            actions=self.raw_data.actions[indices],
            next_observations=nobs,
            future_observations=future_observations,
            rewards=self.raw_data.rewards[indices],
            terminals=terminals,
            timeouts=self.raw_data.timeouts[indices],
        )

    def __len__(self):
        return self.raw_data.num_transitions

    def __repr__(self):
        return rf"""
{self.__class__.__name__}(
    kind={self.kind!r},
    name={self.name!r},
    future_observation_discount={self.future_observation_discount!r},
    env_spec={self.env_spec!r},
)""".lstrip('\n')

    def get_dataloader(self, *,
                       batch_size: int, shuffle: bool = False,
                       drop_last: bool = False,
                       pin_memory: bool = False,
                       num_workers: int = 0, persistent_workers: bool = False,
                       **kwargs) -> torch.utils.data.DataLoader:
        sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(self),
            batch_size=batch_size,
            drop_last=drop_last,
        )
        return torch.utils.data.DataLoader(
            self,
            batch_size=None,
            sampler=sampler,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            **kwargs,
        )


def seed_worker(_):
    worker_seed = torch.utils.data.get_worker_info().seed % (2 ** 32)
    np.random.seed(worker_seed)


from . import d4rl  # register
