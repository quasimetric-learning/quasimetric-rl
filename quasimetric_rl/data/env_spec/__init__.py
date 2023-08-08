r"""
Entry points for
1. accessing info about observation space and action space
2. creating module for processing input observations and actions
3. creating module for generate output actions
"""

import attrs

import gym
import gym.spaces
import torch

from . import input_encoding
from . import act_distn


@attrs.define(kw_only=True)
class EnvSpec:
    observation_space: gym.Space  # always the observation of a single tensor, even if the env gives a dict
    observation_space_is_dict: bool  # whether a dict with key {'observation', 'achieved_goal', 'desired_goal'}
    action_space: gym.Space

    @classmethod
    def from_env(self, env: gym.Env) -> 'EnvSpec':
        ospace = env.observation_space
        observation_space_is_dict = False
        if isinstance(ospace, gym.spaces.Dict):
            # support the goal-cond gym format
            assert set(ospace.spaces.keys()) == {'observation', 'achieved_goal', 'desired_goal'}
            ospace = ospace['observation']
            observation_space_is_dict = True
        return EnvSpec(
            observation_space=ospace,
            action_space=env.action_space,
            observation_space_is_dict=observation_space_is_dict,
        )

    @property
    def action_shape(self) -> torch.Size:
        return torch.Size(self.action_space.shape)

    @property
    def action_dtype(self) -> torch.dtype:
        return convert_to_pytorch_dtype(self.action_space.dtype)

    @property
    def observation_shape(self) -> torch.Size:
        return torch.Size(self.observation_space.shape)

    @property
    def observation_dtype(self) -> torch.dtype:
        return convert_to_pytorch_dtype(self.observation_space.dtype)

    def make_observation_input(self) -> input_encoding.InputEncoding:
        assert isinstance(self.observation_space, gym.spaces.Box)
        if len(self.observation_shape) == 3:
            return input_encoding.AtariTorso(input_shape=self.observation_shape)
        elif len(self.observation_shape) == 1:
            return input_encoding.Identity(input_shape=self.observation_shape)
        else:
            raise NotImplementedError(self.observation_space)

    def make_action_input(self) -> input_encoding.InputEncoding:
        if isinstance(self.action_space, gym.spaces.Discrete):
            assert len(self.action_shape) == 0
            return input_encoding.OneHot(input_shape=torch.Size([]), num_classes=self.action_space.n)
        elif isinstance(self.action_space, gym.spaces.Box):
            return input_encoding.Identity(input_shape=self.action_shape)
        else:
            raise NotImplementedError(self.action_space)

    def get_action_entropy_reg_target(self) -> float:
        if isinstance(self.action_space, gym.spaces.Box):
            assert self.action_space.is_bounded()
            range = torch.as_tensor(self.action_space.high - self.action_space.low)
            return -float(range.sum()) / 2
        elif isinstance(self.action_space, gym.spaces.Discrete):
            raise RuntimeError("Discrete action spaces don't usually require entropy regularization")
        else:
            raise NotImplementedError(self.action_space)

    def make_action_output_distn(self) -> act_distn.ActionOutputConverter:
        if isinstance(self.action_space, gym.spaces.Box):
            return act_distn.BoxOutputLinearNormalization(action_space=self.action_space)
        elif isinstance(self.action_space, gym.spaces.Discrete):
            return act_distn.DiscreteOutputOneHot(action_space=self.action_space)
        else:
            raise NotImplementedError(self.action_space)



def convert_to_pytorch_dtype(np_dtype):
    if hasattr(np_dtype, 'name'):
        name = np_dtype.name
    else:
        name = np_dtype.__name__
    return {
        'bool_'      : torch.bool,
        'uint8'      : torch.uint8,
        'int8'       : torch.int8,
        'int16'      : torch.int16,
        'int32'      : torch.int32,
        'int64'      : torch.int64,
        'float16'    : torch.float16,
        'float32'    : torch.float32,
        'float64'    : torch.float64,
        'complex64'  : torch.complex64,
        'complex128' : torch.complex128
    }[name]
