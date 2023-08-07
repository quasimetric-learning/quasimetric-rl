from typing import Callable

import sys
import os
import traceback
import pdb
import distutils.util

import attrs
import torch
import functools


@attrs.define(kw_only=True)
class FlagsDefinition():
    DEBUG: bool = attrs.field(
        default=distutils.util.strtobool(os.environ.get('QRL_DEBUG', 'False')),
        on_setattr=lambda self, field, val: (torch.autograd.set_detect_anomaly(val), val)[1],
    )


FLAGS = FlagsDefinition()


def pdb_if_DEBUG(fn: Callable):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            fn(*args, **kwargs)
        except:
            # follow ABSL:
            # https://github.com/abseil/abseil-py/blob/a0ae31683e6cf3667886c500327f292c893a1740/absl/app.py#L311-L327

            exc = sys.exc_info()[1]
            if isinstance(exc, KeyboardInterrupt):
                raise

            # Don't try to post-mortem debug successful SystemExits, since those
            # mean there wasn't actually an error. In particular, the test framework
            # raises SystemExit(False) even if all tests passed.
            if isinstance(exc, SystemExit) and not exc.code:
                raise

            # Check the tty so that we don't hang waiting for input in an
            # non-interactive scenario.
            if FLAGS.DEBUG:
                traceback.print_exc()
                print()
                print(' *** Entering post-mortem debugging ***')
                print()
                pdb.post_mortem()
            raise

    return wrapped


__all__ = ['FLAGS', 'pdb_if_DEBUG']
