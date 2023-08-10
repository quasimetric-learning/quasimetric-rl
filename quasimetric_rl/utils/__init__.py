# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import *

import os
import shutil
import functools
import numpy as np

from tqdm.auto import tqdm
tqdm = functools.partial(tqdm, dynamic_ncols=True)


T = TypeVar('T')

def singleton(cls: Type[T]) -> Type[T]:
    instance = None

    def __new__(subcls, *args, **kwargs):
        nonlocal instance
        if instance is None:
            instance = super(cls, subcls).__new__(subcls, *args, **kwargs)
        return instance

    return type(cls.__name__, (cls,), {'__new__': __new__})


def mkdir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass


def rm_if_exists(filename, maybe_dir=False) -> bool:
    r"""
    Returns whether removed
    """
    if os.path.isfile(filename):
        os.remove(filename)
        return True
    elif maybe_dir and os.path.isdir(filename):
        shutil.rmtree(filename)
        return True
    assert not os.path.exists(filename)
    return False


T = TypeVar('T')


class lazy_property(Generic[T]):
    r"""
    Used as a decorator for lazy loading of class attributes. This uses a
    non-data descriptor that calls the wrapped method to compute the property on
    first call; thereafter replacing the wrapped method into an instance
    attribute.
    Derived from:
      https://github.com/pytorch/pytorch/blob/556c8a300b5b062f3429dfac46f6def372bd22fc/torch/distributions/utils.py#L92
    TODO: replace with `functools.cached_property` in py3.8.
    """

    def __init__(self, wrapped: Callable[[Any], T]):
        self.wrapped = wrapped
        functools.update_wrapper(self, wrapped)

    def __get__(self, instance: Any, obj_type: Any = None) -> T:
        if instance is None:
            return self  # typing: ignore
        value = self.wrapped(instance)
        setattr(instance, self.wrapped.__name__, value)
        return value


def as_SeedSequence(seed: Union[np.random.SeedSequence, int, None]) -> np.random.SeedSequence:
    if isinstance(seed, int) or seed is None:
        seed = np.random.SeedSequence(seed)
    return seed


def split_seed(seed: Union[np.random.SeedSequence, int, None], n) -> List[np.random.SeedSequence]:
    return as_SeedSequence(seed).spawn(n)


from . import logging


__all__ = ['mkdir', 'tqdm', 'rm_if_exists', 'lazy_property', 'logging', 'split_seed']