from typing import *

import attrs
import contextlib

import torch



class OptimWrapper(object):
    def __init__(self, optim: torch.optim.Optimizer, *, grad_clip_norm: Optional[float] = None):
        self.optim = optim
        self.grad_clip_norm = grad_clip_norm
        if grad_clip_norm is not None:
            assert len(optim.param_groups) == 1

    @contextlib.contextmanager
    def update_context(self, optimize: bool = True):
        r"""
        useful context manager for wrapping the forward pass of a model & loss backward :)
        """
        if optimize:
            self.optim.zero_grad()
            yield
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.param_groups[0]['params'],
                    self.grad_clip_norm,
                    norm_type=2,
                )
            self.optim.step()
        else:
            yield

    def zero_grad(self):
        return self.optim.zero_grad()

    def step(self):
        return self.optim.step()

    @property
    def param_groups(self):
        return self.optim.param_groups

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict: dict):
        return self.optim.load_state_dict(state_dict)


@attrs.define(kw_only=True)
class AdamWSpec:
    @attrs.define(kw_only=True)
    class Conf:
        alg: str = 'adamw'
        lr: float = attrs.field(default=1e-3, validator=attrs.validators.gt(0))
        betas: Tuple[float, float] = attrs.field(
            converter=tuple,
            default=(0.9, 0.999),
        )
        weight_decay: float = attrs.field(default=0, validator=attrs.validators.ge(0))
        grad_clip_norm: Optional[float] = attrs.field(
            default=None, validator=attrs.validators.optional(attrs.validators.gt(0))
        )
        cosine_lr_decay_final_mul: float = attrs.field(default=1, validator=attrs.validators.and_(
            attrs.validators.ge(0),
            attrs.validators.le(1),
        ))  # 1 means no decay

        def make(self) -> 'AdamWSpec':
            return AdamWSpec(**attrs.asdict(self))

    alg: str
    lr: float
    betas: Tuple[float, float]
    weight_decay: float
    grad_clip_norm: Optional[float]
    cosine_lr_decay_final_mul: float

    def __attrs_post_init__(self):
        assert self.alg == 'adamw', 'Only AdamW is supported.'

    def create_optim(self, params) -> OptimWrapper:
        params = list(params)
        if len(params) == 0:
            params = [dict(params=[])]  # dummy param group so pytorch doesn't complain
        return OptimWrapper(
            torch.optim.AdamW(params, lr=self.lr, betas=self.betas, weight_decay=self.weight_decay),
            grad_clip_norm=self.grad_clip_norm,
        )

    def create_scheduler(self, optim: OptimWrapper, epochs: int) -> torch.optim.lr_scheduler._LRScheduler:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optim.optim, T_max=epochs, eta_min=self.lr * self.cosine_lr_decay_final_mul)

    def create_optim_scheduler(self, params, epochs) -> Tuple[OptimWrapper, torch.optim.lr_scheduler._LRScheduler]:
        optim = self.create_optim(params)
        return optim, self.create_scheduler(optim, epochs)
