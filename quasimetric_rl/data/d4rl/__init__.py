r"""
Adapted from
https://github.com/jannerm/diffuser/blob/2c496e055c0c036a66f653752e96a0f7c66fdcda/diffuser/datasets/d4rl.py
https://github.com/jannerm/diffuser/blob/2c496e055c0c036a66f653752e96a0f7c66fdcda/diffuser/datasets/preprocessing.py
"""
from typing import *

import os
import collections
import logging
import attrs
import abc
from tqdm.auto import tqdm

import numpy as np
import torch
import gym

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


d4rl = None
OfflineEnv = None

def lazy_init_d4rl():
    # d4rl requires mujoco_py, which has a range of installation issues.
    # do not load until needed.

    global d4rl, OfflineEnv

    if d4rl is None:
        import importlib
        with suppress_output():
            ## d4rl prints out a variety of warnings
            d4rl = __import__('d4rl')
        OfflineEnv = d4rl.offline_env.OfflineEnv


if TYPE_CHECKING:
    import d4rl
    import d4rl.offline_env
    import d4rl.pointmaze

    class OfflineEnv(d4rl.offline_env.OfflineEnv):  # give it better type annotation
        name: str
        max_episode_steps: int


#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def load_environment(name: Union[str, gym.Env]) -> 'OfflineEnv':
    if type(name) != str:
        ## name is already an environment
        return name
    with suppress_output():
        wrapped_env: gym.Wrapper = gym.make(name)
    env: 'OfflineEnv' = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    env.reset()
    env.step(env.action_space.sample())  # sometimes stepping is needed to initialize internal
    env.reset()
    return env


def sequence_dataset(env: 'OfflineEnv', dataset: Mapping[str, np.ndarray]) -> Generator[Mapping[str, np.ndarray], None, None]:
    """
    Returns an *ordered* iterator through trajectories.
    Args:
        env: `OfflineEnv`
        dataset: `d4rl` dataset with keys:
            observations
            next_observations
            actions
            rewards
            terminals
            timeouts (optional)
            ...
    Returns:
        An iterator through dictionaries with keys:
            all_observations
            actions
            rewards
            terminals
            timeouts (optional)
            ...
    """

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatibility.
    use_timeouts = 'timeouts' in dataset

    episode_step = 0
    for i in tqdm(range(N), desc=f"{env.name} dataset timesteps"):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env.max_episode_steps - 1)

        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep or i == N - 1:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            assert 'all_observations' not in episode_data
            episode_data['all_observations'] = np.concatenate(
                [episode_data['observations'], episode_data['next_observations'][-1:]], axis=0)
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1


from ..base import EpisodeData


def convert_dict_to_EpisodeData_iter(sequence_dataset_episodes: Iterator[Mapping[str, np.ndarray]]):
    for episode in sequence_dataset_episodes:
        episode_dict = dict(
            episode_lengths=torch.as_tensor([len(episode['all_observations']) - 1], dtype=torch.int64),
            all_observations=torch.as_tensor(episode['all_observations'], dtype=torch.float32),
            actions=torch.as_tensor(episode['actions'], dtype=torch.float32),
            rewards=torch.as_tensor(episode['rewards'], dtype=torch.float32),
            terminals=torch.as_tensor(episode['terminals'], dtype=torch.bool),
            timeouts=(
                torch.as_tensor(episode['timeouts'], dtype=torch.bool) if 'timeouts' in episode else
                torch.zeros(episode['terminals'].shape, dtype=torch.bool)
            ),
            observation_infos={},
            transition_infos={},
        )
        for k, v in episode.items():
            if k.startswith('observation_infos/'):
                episode_dict['observation_infos'][k.split('/', 1)[1]] = v
            elif k.startswith('transition_infos/'):
                episode_dict['transition_infos'][k.split('/', 1)[1]] = v
        yield EpisodeData(**episode_dict)


from . import maze2d  # register

__all__ = ['D4RLDataset']
