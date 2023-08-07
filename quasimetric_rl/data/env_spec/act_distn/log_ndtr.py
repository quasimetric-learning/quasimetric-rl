# author: vlad niculae <vlad@vene.ro>
# license: simplified BSD
# adapted from jax.scipy.special
# <https://github.com/google/jax/blob/master/jax/_src/scipy/special.py>,
# in turn based on cephes

# https://github.com/probabll/mixed-rv-vae/blob/fe809beb42f3c4d0d388ccd534cdba800d4d0a72/torch_log_ndtr.py

import numpy as np
import torch
from scipy.special import ndtr as reference_ndtr
from scipy.special import log_ndtr as reference_log_ndtr

from torch.autograd import Function


_LOGNDTR_FLOAT64_LOWER = torch.tensor(-20, dtype=torch.float64)
_LOGNDTR_FLOAT32_LOWER = torch.tensor(-10, dtype=torch.float32)
_LOGNDTR_FLOAT64_UPPER = torch.tensor(8, dtype=torch.float64)
_LOGNDTR_FLOAT32_UPPER = torch.tensor(5, dtype=torch.float32)


# make this longer
DBL_FAC = [1, 1, 2, 3, 8, 15, 48, 105, 384, 945]
HALF_SQRT2 = np.sqrt(2) / 2
LOG2PI = np.log(2 * np.pi)


def _nrm_logpdf(x):  # is this stable?
    return -(LOG2PI + (x ** 2)) / 2


def _ndtr(x):
    # just using erf is very bad
    # ndtr = (1 + torch.erf(x / SQRT2)) / 2

    w = x * HALF_SQRT2
    z = torch.abs(w)
    y = torch.where(torch.lt(z, HALF_SQRT2),
                   torch.erf(w) + 1,
                   torch.where(torch.gt(w, 0),
                               -torch.erfc(z) + 2,
                               torch.erfc(z)))
    ndtr = y / 2

    return ndtr


def _log_ndtr_lower(x, series_order):
    """Asymptotic expansion version of `Log[cdf(x)]`, appropriate for `x<<-1`."""
    x_2 = x.square()
    log_scale = -(x_2 / 2) - torch.log(-x) - 0.5 * np.log(2. * np.pi)
    return log_scale + torch.log(_log_ndtr_asymptotic_series(x, series_order))


def _log_ndtr_asymptotic_series(x, series_order):
    """Calculates the asymptotic series used in log_ndtr."""
    dtype = x.dtype

    if series_order <= 0:
        return torch.tensor(1, dtype)

    x_2 = x.square()
    even_sum = torch.zeros_like(x)
    odd_sum = torch.zeros_like(x)

    x_2n = x_2  # Start with x^{2*1} = x^{2*n} with n = 1.

    for n in range(1, series_order + 1):
        y = DBL_FAC[2 * n - 1] / x_2n
        if n % 2:
            odd_sum += y
        else:
            even_sum += y
        x_2n *= x_2

    return 1 + even_sum - odd_sum


def _log_ndtr(x, series_order=3):
    dtype = x.dtype

    if dtype == torch.float64:
        lower_segment = _LOGNDTR_FLOAT64_LOWER
        upper_segment = _LOGNDTR_FLOAT64_UPPER
    elif dtype == torch.float32:
        lower_segment = _LOGNDTR_FLOAT32_LOWER
        upper_segment = _LOGNDTR_FLOAT32_UPPER

    # The basic idea here was ported from:
    #   https://root.cern.ch/doc/v608/SpecFuncCephesInv_8cxx_source.html
    # We copy the main idea, with a few changes
    # * For x >> 1, and X ~ Normal(0, 1),
    #     Log[P[X < x]] = Log[1 - P[X < -x]] approx -P[X < -x],
    #     which extends the range of validity of this function.
    # * We use one fixed series_order for all of 'x', rather than adaptive.
    # * Our docstring properly reflects that this is an asymptotic series, not a
    #   Taylor series. We also provided a correct bound on the remainder.
    # * We need to use the max/min in the _log_ndtr_lower arg to avoid nan when
    #   x=0. This happens even though the branch is unchosen because when x=0
    #   the gradient of a select involves the calculation 1*dy+0*(-inf)=nan
    #   regardless of whether dy is finite. Note that the minimum is a NOP if
    #   the branch is chosen.
    # (vlad's note: does the last bullet point matter if using custom backward?)

    # return torch.log(_ndtr(x))

    return torch.where(
        torch.gt(x, upper_segment),
        -_ndtr(-x),
        torch.where(
            torch.gt(x, lower_segment),
            torch.log(_ndtr(torch.maximum(x, lower_segment))),
            _log_ndtr_lower(torch.minimum(x, lower_segment), series_order)
        )
    )


class LogNdtr(Function):

    @staticmethod
    def forward(ctx, x):
        with torch.no_grad():
            y = _log_ndtr(x)
        ctx.save_for_backward(x, y)
        return y

    @staticmethod
    def backward(ctx, dy):
        dx = None
        x, y = ctx.saved_tensors
        return dy * torch.exp(_nrm_logpdf(x) - y)


log_ndtr = LogNdtr.apply

def log_ndtr_general(x: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor):
    return log_ndtr((x - mean) / scale)


def test_gradcheck():
    from torch.autograd import gradcheck
    x = 200 * torch.rand(100, dtype=torch.double) - 100
    x.requires_grad_()
    assert gradcheck(log_ndtr, x)


def main():

    # x = torch.tensor([-9.8099, -1.0396e+01, -1.1412e+01, -6.1407e+02, -131])
    x = torch.tensor([-100, -21, -19, -11, -9], dtype=torch.float64, requires_grad=True)

    x1 = x.to(dtype=torch.float32)
    print(log_ndtr(x1))
    # print(log_ndtr(x1).numpy())
    print(reference_log_ndtr(x1.detach().numpy()))
    print()

    x2 = x.to(dtype=torch.float64)
    print(log_ndtr(x2))
    # print(log_ndtr(x2).numpy())
    print(reference_log_ndtr(x2.detach().numpy()))

    test_gradcheck()



if __name__ == '__main__':
    main()
