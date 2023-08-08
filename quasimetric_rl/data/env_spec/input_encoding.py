from typing import *

import abc

import torch
import torch.nn as nn
import torch.nn.functional as F


class InputEncoding(nn.Module, metaclass=abc.ABCMeta):
    r"""
    Maps input to a flat vector to be fed into neural networks.

    Supports arbitrary batching.
    """
    input_shape: torch.Size
    output_size: int

    def __init__(self, input_shape: torch.Size, output_size: int) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.output_size = output_size

    @abc.abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pass

    # for type hints
    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        return super().__call__(input)


class Identity(InputEncoding):
    def __init__(self, input_shape: torch.Size) -> None:
        assert len(input_shape) == 1
        super().__init__(input_shape, input_shape.numel())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input


class OneHot(InputEncoding):
    def __init__(self, input_shape: torch.Size, num_classes: int) -> None:
        assert len(input_shape) == 0, 'we only support single scalar discrete action'
        super().__init__(input_shape, num_classes)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.dtype == torch.int64
        return F.one_hot(input, self.output_size).to(torch.float32)


class AtariTorso(InputEncoding):
    r'''
    https://github.com/deepmind/acme/blob/3cf8495da73fb64cc09f272b7d99e52da4a3f082/acme/tf/networks/atari.py

    class AtariTorso(base.Module):
        """Simple convolutional stack commonly used for Atari."""

        def __init__(self):
            super().__init__(name='atari_torso')
            self._network = snt.Sequential([
                snt.Conv2D(32, [8, 8], [4, 4]),
                tf.nn.relu,
                snt.Conv2D(64, [4, 4], [2, 2]),
                tf.nn.relu,
                snt.Conv2D(64, [3, 3], [1, 1]),
                tf.nn.relu,
                snt.Flatten(),
            ])

        def __call__(self, inputs: Images) -> tf.Tensor:
            return self._network(inputs)
    '''

    torso: Callable[[torch.Tensor], torch.Tensor]

    def __init__(self, input_shape: torch.Size, activate_last: bool = True) -> None:
        assert tuple(input_shape) in [(64, 64, 3), (64, 64, 4)]
        super().__init__(input_shape, 1024)
        self.torso = nn.Sequential(
            nn.Conv2d(self.input_shape[-1], 32, (8, 8), (4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), (2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 1)),
            nn.ReLU() if activate_last else nn.Identity(),
            nn.Flatten(),
        )

    def permute(self, s: torch.Tensor) -> torch.Tensor:
        assert s.dtype == torch.uint8
        return s.div(255).permute(list(range(s.ndim - 3)) + [-1, -3, -2]) - 0.5

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.torso(self.permute(
            input.flatten(0, -4),
        )).unflatten(0, input.shape[:-3])
