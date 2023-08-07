from typing import *

import attrs

import torch
import torch.nn as nn

from ..utils import MLP

from ...data import EnvSpec
from ...data.env_spec.input_encoding import InputEncoding
from ...data.env_spec.act_distn import ActionOutputConverter


class Actor(nn.Module):
    @attrs.define(kw_only=True)
    class Conf:
        # config / argparse uses this to specify behavior

        arch: Tuple[int, ...] = (512, 512)

        def make(self, *, env_spec: EnvSpec) -> 'Actor':
            return Actor(
                env_spec=env_spec,
                arch=self.arch,
            )

    observation_shape: torch.Size
    observation_encoding: InputEncoding
    backbone: MLP
    action_output: ActionOutputConverter

    def __init__(self, *, env_spec: EnvSpec, arch: Tuple[int, ...], **kwargs):
        super().__init__(**kwargs)
        self.observation_shape = env_spec.observation_shape
        self.observation_encoding = env_spec.make_observation_input()

        self.action_output = env_spec.make_action_output_distn()
        backbone_input_size = self.observation_encoding.output_size * 2  # add goal
        self.backbone = MLP(backbone_input_size, self.action_output.input_size, hidden_sizes=arch,
                            zero_init_last_fc=True)

    def forward(self, o: torch.Tensor, g: torch.Tensor) -> torch.distributions.Distribution:
        og = torch.stack([o, g], dim=-len(self.observation_shape) - 1)
        return self.action_output(self.backbone(self.observation_encoding(og).flatten(-2, -1)))

    # for type hint
    def __call__(self, o: torch.Tensor, g: torch.Tensor) -> torch.distributions.Distribution:
        return super().__call__(o, g)

    def extra_repr(self) -> str:
        return f"observation_shape={self.observation_shape}"
