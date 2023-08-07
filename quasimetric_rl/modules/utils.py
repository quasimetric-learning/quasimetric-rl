from typing import *

import abc
import attrs
import contextlib

import torch
import torch.nn as nn

import numpy as np
from ..data.utils import NestedMapping


LatentTensor = torch.Tensor  # alias to make type hints more readable



#-----------------------------------------------------------------------------#
#--------------------------------- LossBase ----------------------------------#
#-----------------------------------------------------------------------------#

InfoT = NestedMapping[Union[float, torch.Tensor]]


@attrs.define(kw_only=True)
class LossResult:
    loss: Union[torch.Tensor, float]
    info: InfoT

    def __attrs_post_init__(self):
        assert isinstance(self.loss, (int, float)) or self.loss.numel() == 1
        # detach info tensors

        def detach(d: InfoT) -> InfoT:
            if isinstance(d, torch.Tensor):
                return d.detach()
            elif isinstance(d, (float, int)):
                return d
            else:
                return {k: detach(v) for k, v in d.items()}

        object.__setattr__(self, "info", detach(self.info))

    @classmethod
    def combine(cls, results: Mapping[str, 'LossResult']) -> 'LossResult':
        return LossResult(
            loss=sum(r.loss for r in results.values()),
            info={k: r.info for k, r in results.items()},
        )


class LossBase(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> LossResult:
        pass

    # for type hints
    def __call__(self, *args, **kwargs) -> LossResult:
        return super().__call__(*args, **kwargs)


#-----------------------------------------------------------------------------#
#------------------------------------ MLP ------------------------------------#
#-----------------------------------------------------------------------------#


class MLP(nn.Module):
    input_size: int
    output_size: int
    zero_init_last_fc: bool
    module: nn.Sequential

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 *,
                 hidden_sizes: Collection[int],
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 zero_init_last_fc: bool = False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.zero_init_last_fc = zero_init_last_fc

        layer_in_size = input_size
        modules: List[nn.Module] = []
        for sz in hidden_sizes:
            modules.extend([
                nn.Linear(layer_in_size, sz),
                activation_fn(),
            ])
            layer_in_size = sz
        modules.append(
            nn.Linear(layer_in_size, output_size),
        )

        # initialize with glorot_uniform
        with torch.no_grad():
            def init_(m: nn.Module):
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            for m in modules:
                m.apply(init_)
            if zero_init_last_fc:
                last_fc = cast(nn.Linear, modules[-1])
                last_fc.weight.zero_()
                last_fc.bias.zero_()

        self.module = torch.jit.script(nn.Sequential(*modules))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.module(input)

    # for type hints
    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        return super().__call__(input)

    def extra_repr(self) -> str:
        return "zero_init_last_fc={}".format(
            self.zero_init_last_fc,
        )



#-----------------------------------------------------------------------------#
#-------------------------------- Module abc ---------------------------------#
#-----------------------------------------------------------------------------#
# Makes it easier to switch train/eval modes, or detach gradients. It is
# recommended to use this at the top-level modules (e.g., actors), which often
# need such switches.

class Module(nn.Module):
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device  # a bit inaccurate, but should be fine

    @contextlib.contextmanager
    def requiring_grad(self, flag=True):
        rgs = []
        for p in self.parameters():
            rgs.append(p.requires_grad)
            p.requires_grad_(flag)
        yield
        for rg, p in zip(rgs, self.parameters()):
            p.requires_grad_(rg)

    @contextlib.contextmanager
    def mode(module, mode=True):
        orig = module.training
        module.train(mode)
        yield
        module.train(orig)


#-----------------------------------------------------------------------------#
#------------------------------ softplus_inv ---------------------------------#
#-----------------------------------------------------------------------------#


def softplus_inv_float(y: float) -> float:
    threshold: float = 20.  # https://pytorch.org/docs/stable/generated/torch.nn.functional.softplus.html#torch-nn-functional-softplus
    if y > threshold:
        return y
    else:
        return np.log(np.expm1(y))



#-----------------------------------------------------------------------------#
#------------------------------- grad_mul ------------------------------------#
#-----------------------------------------------------------------------------#


class GradMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, mult: Union[float, torch.Tensor]) -> torch.Tensor:
        ctx.mult_is_tensor = isinstance(mult, torch.Tensor)
        if ctx.mult_is_tensor:
            assert not mult.requires_grad
            ctx.save_for_backward(mult)
        else:
            ctx.mult = mult
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.mult_is_tensor:
            mult, = ctx.saved_tensors
        else:
            mult = ctx.mult
        return grad_output * mult, None


def grad_mul(x: torch.Tensor, mult: Union[float, torch.Tensor]) -> torch.Tensor:
    if not isinstance(mult, torch.Tensor) and mult == 0:
        return x.detach()
    return GradMul.apply(x, mult)
