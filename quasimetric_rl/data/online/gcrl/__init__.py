from typing import *

import gym
import gym.spaces
import numpy as np

from ..memory import register_online_env
from . import fetch_envs


class GoalCondEnvWrapper(gym.ObservationWrapper):
    r"""
    Convert the concatenated observation space in GCRL into a better format with
    dict observations.
    """

    episode_length: int
    is_image_based: bool
    create_kwargs: Mapping[str, Any]

    def __init__(self, env: gym.Env, episode_length: int, is_image_based: bool):
        super().__init__(gym.wrappers.TimeLimit(env.unwrapped, episode_length))
        if is_image_based:
            single_ospace = gym.spaces.Box(
                low=np.full((64, 64, 3), 0),
                high=np.full((64, 64, 3), 255),
                dtype=np.uint8,
            )
        else:
            assert isinstance(env.observation_space, gym.spaces.Box)
            ospace: gym.spaces.Box = env.observation_space
            assert len(ospace.shape) == 1
            single_ospace = gym.spaces.Box(
                low=np.split(ospace.low, 2)[0],
                high=np.split(ospace.high, 2)[0],
                dtype=ospace.dtype,
            )
        self.observation_space = gym.spaces.Dict(dict(
            observation=single_ospace,
            achieved_goal=single_ospace,
            desired_goal=single_ospace,
        ))
        self.episode_length = episode_length
        self.is_image_based = is_image_based

    def observation(self, observation):
        o, g = np.split(observation, 2)
        if self.is_image_based:
            o = o.reshape(64, 64, 3)
            g = g.reshape(64, 64, 3)
        odict = dict(
            observation=o,
            achieved_goal=o,
            desired_goal=g,
        )
        return odict



name_img_env = [
    ('FetchReach', False, fetch_envs.FetchReachEnv),
    ('FetchReachImage', True, fetch_envs.FetchReachImage),
    ('FetchPush', False, fetch_envs.FetchPushEnv),
    ('FetchPushImage', True, fetch_envs.FetchPushImage),
    ('FetchSlide', False, fetch_envs.FetchSlideEnv),
]


for name, is_image_based, env_ty in name_img_env:
    register_online_env(
        'gcrl', name,
        create_env_fn=(
            lambda env_ty, is_image_based: lambda: GoalCondEnvWrapper(env_ty(), 50, is_image_based)
        )(env_ty, is_image_based),  # capture in scope!
        episode_length=50,
    )
